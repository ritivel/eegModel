"""`eegfm-eval` CLI — one command, dispatches to one or many task atoms.

Examples:

    # One cell — fast LP on TUAB
    eegfm-eval --checkpoint runs/v2/.../ckpt_final.pt --task tuab --strategy lp

    # A subset
    eegfm-eval --checkpoint ... --tasks tuab,tuev,physionet_mi --strategies lp,ft

    # Everything (runs from a YAML profile under `profiles/`)
    eegfm-eval --checkpoint ... --profile paper_grade

    # Random-init floor on every task
    eegfm-eval --random-init --profile sanity_floor

    # Reproduce a published number to validate the harness
    eegfm-eval --reproduce labram_base --task tuab --strategy ft \\
        --reproduce-checkpoint /path/to/labram-base.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .runner import list_tasks, run

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # Encoder
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="eegfm checkpoint .pt path"),
    random_init: bool = typer.Option(False, "--random-init",
                                     help="Use random-init encoder (no pretraining); ignores --checkpoint"),
    arch: str = typer.Option("mamba2_d256_l6", "--arch",
                             help="Architecture string for --random-init"),
    reproduce: Optional[str] = typer.Option(None, "--reproduce",
                                            help="Reproduce mode: 'labram_base' | 'cbramod' | 'eegpt' | 'biot' | 'reve_base'"),
    reproduce_checkpoint: Optional[Path] = typer.Option(None, "--reproduce-checkpoint",
                                                        help="Path to the third-party pretrained weights"),
    # What to run
    task: Optional[str] = typer.Option(None, "--task", help="Single task name"),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma-separated task names"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="lp | ft"),
    strategies: Optional[str] = typer.Option(None, "--strategies",
                                             help="Comma-separated: lp,ft"),
    profile: Optional[Path] = typer.Option(None, "--profile",
                                           help="Path to a YAML profile (or short name from profiles/)"),
    # Inputs / outputs
    derived_root: Path = typer.Option(Path("/opt/dlami/nvme/eeg/derived"), "--derived-root",
                                      help="Where preprocessed parquet shards live"),
    output_dir: Path = typer.Option(Path("./eval_runs"), "--output-dir",
                                    help="Where to write per-atom JSON results"),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("cuda", "--device", help="cuda | cpu"),
    # Inspection
    show_tasks: bool = typer.Option(False, "--list-tasks", help="Print all registered tasks and exit"),
):
    """Run one or more (task, strategy) atoms against an encoder."""
    if show_tasks:
        for name in list_tasks():
            print(name)
        raise typer.Exit(0)

    # Resolve which tasks/strategies to run -----------------------------------
    if profile is not None:
        prof_data = _load_profile(profile)
        task_strategy_pairs = [(t, s) for t in prof_data["tasks"] for s in prof_data["strategies"]]
    else:
        if not (task or tasks):
            typer.echo("ERROR: provide --task / --tasks / --profile", err=True)
            raise typer.Exit(2)
        task_list = [task] if task else [t.strip() for t in tasks.split(",")]
        strat_list = ([strategy] if strategy
                      else [s.strip() for s in (strategies or "lp").split(",")])
        task_strategy_pairs = [(t, s) for t in task_list for s in strat_list]

    # Resolve encoder ---------------------------------------------------------
    if reproduce:
        encoder_kind = reproduce
        ckpt = reproduce_checkpoint
        if ckpt is None:
            typer.echo("ERROR: --reproduce requires --reproduce-checkpoint", err=True)
            raise typer.Exit(2)
    elif random_init:
        encoder_kind = "random_init"
        ckpt = None
    else:
        if checkpoint is None:
            typer.echo("ERROR: provide --checkpoint or --random-init or --reproduce", err=True)
            raise typer.Exit(2)
        encoder_kind = "eegfm"
        ckpt = checkpoint

    # Run all atoms -----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for t_name, s_name in task_strategy_pairs:
        console.print(f"\n[cyan bold]►[/cyan bold] task=[cyan]{t_name}[/cyan] "
                      f"strategy=[cyan]{s_name}[/cyan] "
                      f"encoder=[cyan]{encoder_kind}[/cyan]")
        out_file = output_dir / f"{t_name}__{s_name}.json"
        try:
            r = run(
                task=t_name, strategy=s_name,
                checkpoint=ckpt, encoder_kind=encoder_kind,
                derived_root=derived_root, output=out_file,
                device=device, seed=seed,
                arch=arch if encoder_kind == "random_init" else None,
            )
            results.append(r)
        except Exception as e:                                  # noqa: BLE001
            console.print(f"[red]FAIL[/red] {t_name}/{s_name}: {type(e).__name__}: {e}")
            results.append({"task": t_name, "strategy": s_name,
                            "error": f"{type(e).__name__}: {e}"})

    _print_scoreboard(results)


def _load_profile(profile: Path) -> dict:
    """Load a YAML profile. If `profile` doesn't exist as a path, look it up under
    `eegfm_eval/profiles/<name>.yaml`."""
    if profile.exists():
        return yaml.safe_load(profile.read_text())
    # short-name lookup
    pkg_profile = Path(__file__).parent / "profiles" / f"{profile}.yaml"
    if pkg_profile.exists():
        return yaml.safe_load(pkg_profile.read_text())
    raise FileNotFoundError(f"profile {profile!r} not found")


def _print_scoreboard(results: list[dict]) -> None:
    table = Table(title="eegfm-eval scoreboard", show_lines=False)
    table.add_column("task", style="cyan", no_wrap=True)
    table.add_column("strat", justify="center")
    table.add_column("primary metric", justify="right", style="bold")
    table.add_column("vs lit best", justify="right")
    table.add_column("CI", justify="right")
    table.add_column("runtime", justify="right")

    for r in results:
        if "error" in r:
            table.add_row(r["task"], r["strategy"], "—", "—", "—", f"[red]{r['error'][:40]}[/red]")
            continue
        metrics = r.get("metrics", {})
        primary_name = next(iter(metrics)) if metrics else "?"
        primary = metrics.get(primary_name, {})
        point = primary.get("point", float("nan"))
        ci = (f"[{primary.get('ci_low_95', float('nan')):.3f}, "
              f"{primary.get('ci_high_95', float('nan')):.3f}]"
              if "ci_low_95" in primary else "—")
        anchors = r.get("lit_anchors", {})
        best_lit = max(anchors.values()) if anchors else None
        delta = (f"{point - best_lit:+.3f}" if best_lit is not None else "—")
        runtime = f"{r.get('wallclock_s', 0):.1f}s"
        table.add_row(r["task"], r["strategy"],
                      f"{primary_name}={point:.4f}", delta, ci, runtime)

    console.print(table)


if __name__ == "__main__":
    app()
