"""``eeg-ops`` — top-level CLI.

Subcommand groups:

* ``capacity`` — find / buy / status of AWS Capacity Blocks
* ``cluster``  — up / status / exec / down (delegates to SkyPilot)
* ``data``     — pre-warm a regional S3 mirror of preprocessed shards
* ``iam``      — create the instance profile the box uses for S3 RW + CW
* ``alarm``    — create the CloudWatch billing alarm for runaway spend
* ``checkpoint`` — list / sync the runs/ tree to/from S3
* ``config``   — print the resolved state, paths, and IDs
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import aws as aws_mod
from . import prewarm
from . import sky as sky_mod
from .config import (
    DEFAULT_STATE_PATH,
    WAREHOUSE_BUCKET,
    WAREHOUSE_REGION,
    CapacityState,
    State,
    region_config,
)

app = typer.Typer(
    name="eeg-ops",
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Cluster-lifecycle CLI for the eegModel project. Wraps SkyPilot for "
        "short-term GPU rentals (AWS Capacity Blocks today; Lambda / GCP / "
        "Azure later via SkyPilot's auto-failover) and handles the AWS chores "
        "SkyPilot doesn't: capacity-block purchase, IAM instance profile, "
        "CloudWatch alarms, and the cross-region warehouse mirror."
    ),
    rich_markup_mode="rich",
)
console = Console()


# ---------------------------------------------------------------------------
# capacity — describe / buy / status of AWS Capacity Blocks
# ---------------------------------------------------------------------------

capacity_app = typer.Typer(no_args_is_help=True, help="AWS Capacity Block lifecycle.")
app.add_typer(capacity_app, name="capacity")


@capacity_app.command("find")
def capacity_find(
    region: str = typer.Option(..., "--region", help="AWS region, e.g. ap-south-1"),
    instance_type: str = typer.Option("p5.48xlarge", "--instance-type"),
    instance_count: int = typer.Option(1, "--count"),
    duration_hours: int = typer.Option(168, "--duration-hours",
                                       help="One of {24,48,…,168,…} as accepted by AWS."),
    search_window_days: int = typer.Option(56, "--search-window-days",
                                           help="Look for slots up to N days in the future."),
):
    """List Capacity Block offerings sorted by start time. Cheapest first slot wins."""
    offerings = aws_mod.find_capacity_offerings(
        region=region,
        instance_type=instance_type,
        instance_count=instance_count,
        duration_hours=duration_hours,
        search_window_days=search_window_days,
    )
    if not offerings:
        console.print(f"[yellow]no offerings[/yellow] for {instance_type} × {instance_count} "
                      f"× {duration_hours}h in {region}")
        raise typer.Exit(2)
    t = Table(title=f"Capacity Block offerings - {region}")
    t.add_column("Offering ID", style="cyan")
    t.add_column("AZ")
    t.add_column("Instances", justify="right")
    t.add_column("Start (UTC)")
    t.add_column("End (UTC)")
    t.add_column("Hours", justify="right")
    t.add_column("Upfront $", justify="right")
    for o in offerings:
        t.add_row(o.offering_id, o.az, str(o.instance_count),
                  o.start_date, o.end_date, str(o.duration_hours),
                  f"{o.upfront_fee_usd:,.2f}")
    console.print(t)


@capacity_app.command("buy")
def capacity_buy(
    offering_id: str = typer.Argument(..., help="from `eeg-ops capacity find`"),
    region: str = typer.Option(..., "--region"),
    instance_type: str = typer.Option("p5.48xlarge", "--instance-type"),
    duration_hours: int = typer.Option(168, "--duration-hours"),
    expected_fee_usd: float = typer.Option(..., "--expected-fee",
                                           help="Hard-fail if AWS price drifted >$0.01."),
    yes: bool = typer.Option(False, "--yes", help="Skip the type-PURCHASE confirmation."),
):
    """Re-verify the offering and commit. **Irreversible** — charges immediately."""
    state = State.load()

    console.print("\n[bold]Confirm purchase[/bold]")
    console.print(f"  region:        {region}")
    console.print(f"  offering:      [cyan]{offering_id}[/cyan]")
    console.print(f"  instance_type: {instance_type}")
    console.print(f"  duration:      {duration_hours} h")
    console.print(f"  upfront:       [yellow]${expected_fee_usd:,.2f}[/yellow] (NON-REFUNDABLE)\n")

    if not yes:
        token = typer.prompt("Type PURCHASE to commit")
        if token != "PURCHASE":
            console.print("[red]aborted[/red]")
            raise typer.Exit(4)

    cr = aws_mod.purchase_capacity_block(
        region=region, offering_id=offering_id, expected_fee_usd=expected_fee_usd,
    )
    cap = CapacityState(
        reservation_id=cr["CapacityReservationId"],
        offering_id=offering_id,
        region=region,
        az=cr.get("AvailabilityZone", ""),
        instance_type=cr.get("InstanceType", instance_type),
        duration_hours=duration_hours,
        upfront_fee_usd=expected_fee_usd,
        start_date=cr["StartDate"].isoformat() if hasattr(cr["StartDate"], "isoformat") else str(cr["StartDate"]),
        end_date=cr["EndDate"].isoformat() if hasattr(cr["EndDate"], "isoformat") else str(cr["EndDate"]),
        state=cr.get("State", "payment-pending"),
    )
    state.upsert_capacity(cap, set_active=True)
    state.save()

    # Wire SkyPilot to use this reservation automatically.
    sky_mod.register_capacity_reservation(cap.reservation_id)

    console.print(f"[green]✓[/green] purchased [cyan]{cap.reservation_id}[/cyan]")
    console.print(f"  state:    {cap.state}")
    console.print(f"  start:    {cap.start_date}")
    console.print(f"  end:      {cap.end_date}")
    console.print(f"  state.toml updated: {DEFAULT_STATE_PATH}")
    console.print(f"  ~/.sky/config.yaml updated (aws.specific_reservations += {cap.reservation_id})")


@capacity_app.command("status")
def capacity_status(
    reservation_id: str = typer.Option(None, "--id",
                                       help="Default: the active reservation in state.toml"),
):
    """Show live state, hours elapsed, hours remaining, $/hr equivalent."""
    state = State.load()
    rid = reservation_id or state.active_reservation_id
    if rid is None:
        console.print("[yellow]no active reservation in state.toml[/yellow]")
        raise typer.Exit(1)
    rec = state.capacities.get(rid)
    if rec is None:
        console.print(f"[yellow]{rid} not in local state[/yellow]")
        raise typer.Exit(1)

    live = aws_mod.describe_reservation(region=rec.region, reservation_id=rid)
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(rec.start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(rec.end_date.replace("Z", "+00:00"))
    h_elapsed = max(0, int((now - start).total_seconds() // 3600))
    h_remaining = max(0, int((end - now).total_seconds() // 3600))
    per_hour = rec.upfront_fee_usd / max(rec.duration_hours, 1)
    value_left = per_hour * h_remaining

    t = Table(title=f"Reservation {rid}")
    t.add_column("Field", style="cyan")
    t.add_column("Value")
    t.add_row("region", rec.region)
    t.add_row("instance_type", rec.instance_type)
    t.add_row("state (live)", live.get("State", "?"))
    t.add_row("available_instances", str(live.get("AvailableInstanceCount", "?")))
    t.add_row("start (UTC)", rec.start_date)
    t.add_row("end (UTC)", rec.end_date)
    t.add_row("h_elapsed / h_remaining", f"{h_elapsed} / {h_remaining}")
    t.add_row("$/hr equivalent", f"${per_hour:.2f}")
    t.add_row("value left on meter", f"${value_left:,.2f}")
    console.print(t)


# ---------------------------------------------------------------------------
# cluster — wraps SkyPilot
# ---------------------------------------------------------------------------

cluster_app = typer.Typer(no_args_is_help=True, help="SkyPilot cluster lifecycle.")
app.add_typer(cluster_app, name="cluster")


@cluster_app.command("up")
def cluster_up(
    yaml_path: Path = typer.Option(
        Path("infrastructure/aws-mumbai/eeg.sky.yaml"), "--yaml",
        help="SkyPilot task YAML."),
    cluster_name: str = typer.Option("eeg-mumbai-2026w19", "--name"),
    retry_until_up: bool = typer.Option(True, "--retry-until-up/--no-retry"),
    forward_envs: list[str] = typer.Option(
        ["WANDB_API_KEY", "HF_TOKEN"], "--env",
        help="Local env-var names to forward into the cluster."),
):
    """Launch the cluster. SkyPilot finds the registered reservation and
    waits for it to flip to 'active' if needed (--retry-until-up)."""
    import os
    state = State.load()
    if state.active_reservation_id is None:
        console.print("[yellow]warn[/yellow] no active reservation in state.toml — "
                      "SkyPilot will fall back to on-demand pricing.")
    extra_envs = {k: os.environ[k] for k in forward_envs if os.environ.get(k)}
    rc = sky_mod.launch(
        cluster_name=cluster_name, yaml_path=yaml_path,
        retry_until_up=retry_until_up, extra_envs=extra_envs,
    )
    if rc == 0:
        state.cluster_name = cluster_name
        state.save()
    raise typer.Exit(rc)


@cluster_app.command("status")
def cluster_status(
    name: str = typer.Option(None, "--name"),
):
    state = State.load()
    raise typer.Exit(sky_mod.status(cluster_name=name or state.cluster_name))


@cluster_app.command("exec")
def cluster_exec(
    command: str = typer.Argument(..., help="Quoted shell command to run on the cluster."),
    name: str = typer.Option(None, "--name"),
):
    state = State.load()
    cn = name or state.cluster_name
    if not cn:
        console.print("[red]no cluster name in state.toml[/red]")
        raise typer.Exit(1)
    raise typer.Exit(sky_mod.exec_remote(cluster_name=cn, command=command))


@cluster_app.command("down")
def cluster_down(
    name: str = typer.Option(None, "--name"),
    sync_runs: bool = typer.Option(True, "--sync-runs/--no-sync",
                                   help="rsync $EXP03_DATA_ROOT/runs/ → warehouse before terminate"),
):
    """Terminate the cluster, optionally pushing local runs/ back to the warehouse first."""
    state = State.load()
    cn = name or state.cluster_name
    if cn is None:
        console.print("[red]no cluster name in state.toml[/red]")
        raise typer.Exit(1)

    if sync_runs:
        cmd = (
            "source ~/sky_workdir/.venv/bin/activate 2>/dev/null; "
            "rclone copy /opt/dlami/nvme/eeg/runs/ s3w:runs/exp03/ "
            "--transfers 32 --checkers 32 --quiet"
        )
        console.print(f"[cyan]sync runs/ → warehouse on {cn}[/cyan]")
        sky_mod.exec_remote(cluster_name=cn, command=cmd)

    rc = sky_mod.down(cluster_name=cn)
    if rc == 0:
        # Unregister the now-spent reservation from sky config so future launches
        # don't slow down on a dead ID.
        active = state.get_active()
        if active is not None:
            sky_mod.unregister_capacity_reservation(active.reservation_id)
        state.cluster_name = None
        state.save()
    raise typer.Exit(rc)


# ---------------------------------------------------------------------------
# data — pre-warm regional S3 mirror
# ---------------------------------------------------------------------------

data_app = typer.Typer(no_args_is_help=True, help="Pre-warm a regional S3 mirror.")
app.add_typer(data_app, name="data")


@data_app.command("prewarm")
def data_prewarm(
    region: str = typer.Option(..., "--region"),
    pipelines: str = typer.Option(",".join(prewarm.DEFAULT_PIPELINES), "--pipelines",
                                  help="Comma-separated pipeline names."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Replicate ``s3://<warehouse>/derived/`` → ``s3://<region-cache>/derived/``.

    Idempotent. Re-runs skip objects with matching ETags. Useful again on the
    first day of every new rental in a different region."""
    rc = region_config(region)
    bucket = prewarm.ensure_cache_bucket(region_cfg=rc)
    console.print(f"[cyan]pre-warming[/cyan] {WAREHOUSE_BUCKET} ({WAREHOUSE_REGION}) "
                  f"→ {bucket} ({region})")
    pls = [p.strip() for p in pipelines.split(",") if p.strip()]
    rc_code = prewarm.sync_pipelines(region_cfg=rc, pipelines=pls, dry_run=dry_run)
    n_obj, total = prewarm.measure_cache_size(region_cfg=rc)
    console.print(f"[green]done[/green]: {n_obj:,} objects, {total / 2**30:.1f} GiB in cache")
    raise typer.Exit(rc_code)


# ---------------------------------------------------------------------------
# iam — instance profile creation
# ---------------------------------------------------------------------------

iam_app = typer.Typer(no_args_is_help=True, help="IAM instance profile management.")
app.add_typer(iam_app, name="iam")


@iam_app.command("create")
def iam_create(
    region: str = typer.Option(..., "--region"),
    extra_buckets: list[str] = typer.Option([], "--extra-bucket",
                                            help="Additional S3 buckets to grant RW to."),
):
    """Create the instance profile the cluster will assume.

    Idempotent — re-runs update the inline policy in place."""
    rc = region_config(region)
    buckets = [WAREHOUSE_BUCKET, rc.cache_bucket, *extra_buckets]
    arns = aws_mod.ensure_instance_profile(
        profile_name=rc.instance_profile_name, buckets=buckets,
    )
    console.print(f"[green]✓[/green] role: [cyan]{arns['role_arn']}[/cyan]")
    console.print(f"[green]✓[/green] instance profile: [cyan]{arns['instance_profile_arn']}[/cyan]")


# ---------------------------------------------------------------------------
# alarm — CloudWatch billing alarm
# ---------------------------------------------------------------------------

alarm_app = typer.Typer(no_args_is_help=True, help="CloudWatch billing alarms.")
app.add_typer(alarm_app, name="alarm")


@alarm_app.command("create")
def alarm_create(
    region: str = typer.Option(..., "--region"),
    daily_budget_usd: float = typer.Option(..., "--daily-budget-usd"),
    alarm_name: str = typer.Option("eeg-ops-budget-alarm", "--name"),
    sns_topic_arn: str = typer.Option(None, "--sns-topic-arn",
                                       help="If set, alarm publishes to this SNS topic."),
):
    """Create or update a CloudWatch billing alarm. AWS billing metrics are
    reported every ~6h, so this is coarse — useful as a backstop, not a tripwire."""
    out = aws_mod.ensure_billing_alarm(
        region=region, alarm_name=alarm_name,
        daily_budget_usd=daily_budget_usd, sns_topic_arn=sns_topic_arn,
    )
    console.print(f"[green]✓[/green] alarm '{out['alarm']}' threshold ${out['threshold']}/day")


# ---------------------------------------------------------------------------
# checkpoint — runs/ ↔ warehouse
# ---------------------------------------------------------------------------

ckpt_app = typer.Typer(no_args_is_help=True, help="Checkpoint utilities.")
app.add_typer(ckpt_app, name="checkpoint")


@ckpt_app.command("sync-runs")
def checkpoint_sync_runs(
    local_dir: Path = typer.Option(Path("/opt/dlami/nvme/eeg/runs"), "--local-dir"),
    bucket: str = typer.Option(WAREHOUSE_BUCKET, "--bucket"),
    prefix: str = typer.Option("runs/exp03", "--prefix"),
    region: str = typer.Option(WAREHOUSE_REGION, "--region"),
):
    """One-shot ``aws s3 sync`` of the local runs tree to the warehouse.

    Run this near the end of the reservation window so nothing in NVMe is
    lost when AWS force-stops at EndDate."""
    from .checkpoint import sync_runs_dir
    rc = sync_runs_dir(local_runs_dir=local_dir, bucket=bucket, prefix=prefix, region=region)
    raise typer.Exit(rc)


@ckpt_app.command("latest")
def checkpoint_latest(
    bucket: str = typer.Option(..., "--bucket"),
    prefix: str = typer.Option(..., "--prefix"),
    region: str = typer.Option(..., "--region"),
):
    """Print the most-recent step number whose checkpoint exists under prefix."""
    from .checkpoint import S3DCPCheckpointSink
    sink = S3DCPCheckpointSink(bucket=bucket, prefix=prefix, region=region)
    n = sink.latest_step()
    console.print(n if n is not None else "<none>")


# ---------------------------------------------------------------------------
# config — show resolved state
# ---------------------------------------------------------------------------


@app.command("config")
def config_show(
    region: str = typer.Option(None, "--region"),
):
    """Print resolved per-region defaults plus mutable state."""
    state = State.load()
    if region:
        rc = region_config(region)
        t = Table(title=f"region_config({region})")
        t.add_column("key", style="cyan")
        t.add_column("value")
        for k, v in rc.__dict__.items():
            t.add_row(k, str(v))
        console.print(t)

    t2 = Table(title=f"state ({DEFAULT_STATE_PATH})")
    t2.add_column("key", style="cyan")
    t2.add_column("value")
    t2.add_row("aws_account_id", str(state.aws_account_id))
    t2.add_row("active_reservation_id", str(state.active_reservation_id))
    t2.add_row("cluster_name", str(state.cluster_name))
    t2.add_row("known_reservations", ", ".join(state.capacities) or "—")
    console.print(t2)


def main(argv: list[str] | None = None) -> int:
    """Entry point declared in pyproject.toml's ``[project.scripts]``."""
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
