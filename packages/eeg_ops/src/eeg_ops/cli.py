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
from . import launcher as launcher_mod
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
    via: str = typer.Option(
        "boto3", "--via",
        help="'boto3' (default, fast, deterministic) or 'skypilot' (multi-cloud, "
             "requires every AWS regional endpoint reachable from this network)."),
    cluster_name: str = typer.Option("eeg-mumbai-2026w19", "--name"),
    instance_type: str = typer.Option("p5.48xlarge", "--instance-type"),
    key_name: str = typer.Option("eeg-mumbai-2026w19", "--key-name"),
    security_group_id: str = typer.Option("sg-0a7017c9b48c300ca", "--sg"),
    repo_url: str = typer.Option(
        "https://github.com/ritivel/eegModel.git", "--repo-url"),
    repo_branch: str = typer.Option("main", "--repo-branch"),
    yaml_path: Path = typer.Option(
        Path("infrastructure/aws-mumbai/eeg.sky.yaml"), "--yaml",
        help="(SkyPilot path only) task YAML."),
    retry_until_up: bool = typer.Option(True, "--retry-until-up/--no-retry",
                                        help="(SkyPilot path only)"),
    forward_envs: list[str] = typer.Option(
        ["WANDB_API_KEY", "HF_TOKEN"], "--env",
        help="Local env-var names to forward into the cluster."),
):
    """Launch the cluster against the active reservation.

    Default ``--via boto3`` runs ``ec2:RunInstances`` directly with the
    instance profile, security group, AMI, and reservation we already have
    set up. Returns in ~3-5 min (instance pending → running → SSH ready).

    ``--via skypilot`` delegates to ``sky launch``; nicer long-term
    (multi-cloud failover, recipes, managed jobs) but currently fails on
    networks that can't reach all AWS regional endpoints (it scans every
    region for capacity reservations).
    """
    import os
    state = State.load()

    if via == "boto3":
        active = state.get_active()
        if active is None:
            console.print("[red]no active reservation in state.toml[/red]")
            raise typer.Exit(1)
        rc = region_config(active.region)

        # Resolve the IAM instance profile ARN — created by `eeg-ops iam create`.
        import boto3
        prof = boto3.client("iam").get_instance_profile(
            InstanceProfileName=rc.instance_profile_name
        )["InstanceProfile"]
        ip_arn = prof["Arn"]

        console.print(f"[cyan]launching[/cyan] {instance_type} into "
                      f"{active.reservation_id} ({rc.region}) ...")
        result = launcher_mod.launch_into_reservation(
            region_cfg=rc,
            reservation_id=active.reservation_id,
            instance_type=instance_type,
            key_name=key_name,
            security_group_id=security_group_id,
            instance_profile_arn=ip_arn,
            repo_url=repo_url,
            repo_branch=repo_branch,
            name_tag=cluster_name,
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            hf_token=os.environ.get("HF_TOKEN"),
        )
        launcher_mod.remember_launch(
            state, cluster_name=cluster_name,
            instance_id=result.instance_id, public_ip=result.public_ip,
        )
        console.print(f"[green]✓[/green] instance: [cyan]{result.instance_id}[/cyan]")
        console.print(f"[green]✓[/green] public IP: [cyan]{result.public_ip}[/cyan]")
        console.print()
        console.print("Connect:")
        console.print(f"  [bold]ssh -A -i ~/.ssh/{key_name}.pem ubuntu@{result.public_ip}[/bold]")
        console.print()
        console.print("Bootstrap is running on the box (cloud-init). Tail:")
        console.print(f"  [dim]ssh ubuntu@{result.public_ip} 'sudo tail -f /var/log/eeg-bootstrap.log'[/dim]")
        return

    if via == "skypilot":
        if state.active_reservation_id is None:
            console.print("[yellow]warn[/yellow] no active reservation in state.toml.")
        extra_envs = {k: os.environ[k] for k in forward_envs if os.environ.get(k)}
        rc_code = sky_mod.launch(
            cluster_name=cluster_name, yaml_path=yaml_path,
            retry_until_up=retry_until_up, extra_envs=extra_envs,
        )
        if rc_code == 0:
            state.cluster_name = cluster_name
            state.save()
        raise typer.Exit(rc_code)

    console.print(f"[red]unknown --via value: {via!r}[/red]")
    raise typer.Exit(2)


@cluster_app.command("status")
def cluster_status(
    name: str = typer.Option(None, "--name"),
    via: str = typer.Option("boto3", "--via",
                            help="'boto3' to query EC2 directly, 'skypilot' for sky status"),
):
    """Show the cluster's instance state, IPs, and ssh command."""
    state = State.load()
    if via == "skypilot":
        raise typer.Exit(sky_mod.status(cluster_name=name or state.cluster_name))

    # boto3 path
    active = state.get_active()
    if active is None:
        console.print("[yellow]no active reservation in state.toml[/yellow]")
        raise typer.Exit(1)
    cluster_tag = name or (state.cluster_name or "").split(":")[0] or "eeg-mumbai-2026w19"
    instances = launcher_mod.find_instances(region=active.region, cluster_tag=cluster_tag)
    if not instances:
        console.print(f"[yellow]no instances tagged cluster={cluster_tag} in {active.region}[/yellow]")
        raise typer.Exit(0)
    t = Table(title=f"Cluster {cluster_tag}")
    for col in ("instance_id", "state", "type", "public_ip", "az", "launch_time"):
        t.add_column(col, style="cyan" if col == "instance_id" else None)
    for i in instances:
        t.add_row(*[str(i.get(c, "")) for c in ("instance_id", "state", "type",
                                                 "public_ip", "az", "launch_time")])
    console.print(t)


@cluster_app.command("exec")
def cluster_exec(
    command: str = typer.Argument(..., help="Quoted shell command to run on the cluster."),
    name: str = typer.Option(None, "--name"),
    via: str = typer.Option("boto3", "--via",
                            help="'boto3' (ssh into the IP from state.toml) or 'skypilot' (sky exec)"),
    key_name: str = typer.Option("eeg-mumbai-2026w19", "--key-name"),
):
    """Run a shell command on the cluster."""
    state = State.load()
    if via == "skypilot":
        cn = name or state.cluster_name
        if not cn:
            console.print("[red]no cluster name in state.toml[/red]")
            raise typer.Exit(1)
        raise typer.Exit(sky_mod.exec_remote(cluster_name=cn, command=command))

    # boto3 path: ssh in directly
    active = state.get_active()
    if active is None or not state.cluster_name or ":" not in state.cluster_name:
        console.print("[red]boto3 path requires `eeg-ops cluster up --via boto3` first[/red]")
        raise typer.Exit(1)
    _name, _instance_id, ip = state.cluster_name.split(":", 2)
    if ip == "-":
        console.print("[red]no public IP recorded in state.toml[/red]")
        raise typer.Exit(1)
    import subprocess
    full = (
        f"ssh -A -i ~/.ssh/{key_name}.pem -o StrictHostKeyChecking=accept-new "
        f"ubuntu@{ip} {command!r}"
    )
    raise typer.Exit(subprocess.call(full, shell=True))


@cluster_app.command("down")
def cluster_down(
    name: str = typer.Option(None, "--name"),
    via: str = typer.Option("boto3", "--via",
                            help="'boto3' to ec2:TerminateInstances, 'skypilot' for sky down"),
    sync_runs: bool = typer.Option(True, "--sync-runs/--no-sync",
                                   help="rsync $EXP03_DATA_ROOT/runs/ → warehouse before terminate"),
    key_name: str = typer.Option("eeg-mumbai-2026w19", "--key-name"),
):
    """Terminate the cluster, optionally pushing local runs/ back to the warehouse first."""
    state = State.load()
    active = state.get_active()
    if active is None:
        console.print("[red]no active reservation in state.toml[/red]")
        raise typer.Exit(1)

    cluster_tag = name or (state.cluster_name or "").split(":")[0] or "eeg-mumbai-2026w19"

    if sync_runs:
        instances = launcher_mod.find_instances(region=active.region, cluster_tag=cluster_tag)
        running = [i for i in instances if i["state"] == "running"]
        if running and via == "boto3":
            ip = running[0]["public_ip"]
            console.print(f"[cyan]sync runs/ → warehouse on {ip}[/cyan]")
            ssh_cmd = (
                f"ssh -i ~/.ssh/{key_name}.pem -o StrictHostKeyChecking=accept-new "
                f"ubuntu@{ip} '"
                "rclone copy /opt/dlami/nvme/eeg/runs/ s3w:runs/exp03/ "
                "--transfers 32 --checkers 32 --quiet || true'"
            )
            import subprocess
            subprocess.call(ssh_cmd, shell=True)
        elif via == "skypilot":
            sky_cmd = (
                "rclone copy /opt/dlami/nvme/eeg/runs/ s3w:runs/exp03/ "
                "--transfers 32 --checkers 32 --quiet"
            )
            sky_mod.exec_remote(cluster_name=state.cluster_name or cluster_tag, command=sky_cmd)

    if via == "skypilot":
        rc = sky_mod.down(cluster_name=state.cluster_name or cluster_tag)
    else:
        n = launcher_mod.terminate_cluster(region=active.region, cluster_tag=cluster_tag)
        console.print(f"[green]✓[/green] terminated {n} instance(s)")
        rc = 0

    if rc == 0:
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
    # Plumb the role into ~/.sky/config.yaml so `sky launch` attaches it.
    sky_mod.set_remote_identity(arns["role_arn"])
    console.print("[green]✓[/green] ~/.sky/config.yaml updated: aws.remote_identity")


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
