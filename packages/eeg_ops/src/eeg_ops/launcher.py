"""Direct-boto3 cluster launcher.

Why this exists alongside SkyPilot
----------------------------------
SkyPilot is the better long-term abstraction (multi-cloud auto-failover,
managed jobs, recipes). But its capacity-reservation optimizer scans every
AWS region on each ``sky launch`` to verify reservation locations, which
takes 3-5 minutes on networks where any AWS regional endpoint is
unreachable (e.g. ``ec2.me-south-1.amazonaws.com`` from networks where
that region is geo-blocked or routed through slow paths).

For a literal one-region one-week reservation, a direct boto3 launch is
seconds, not minutes, and fails fast with explicit error messages. This
module is the deterministic path; SkyPilot remains available via
``eeg-ops cluster up --via skypilot`` if the network is fine.

The two paths share the *post-launch* surface: same SSH key, same AMI,
same bootstrap (the SkyPilot YAML's ``setup`` block is replicated in
``BOOTSTRAP_USER_DATA`` here as a cloud-init script).
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import (
    WAREHOUSE_BUCKET,
    WAREHOUSE_REGION,
    RegionConfig,
    State,
)


@dataclass
class LaunchResult:
    instance_id: str
    public_ip: str | None
    private_ip: str
    az: str
    region: str

    @property
    def ssh_target(self) -> str:
        return self.public_ip or self.private_ip


# ---------------------------------------------------------------------------
# user-data: runs once on first boot, before the user's first SSH.
# Mirrors the SkyPilot YAML's setup block exactly so both paths land on the
# same end state.
# ---------------------------------------------------------------------------


def _user_data(*,
               wandb_api_key: str | None,
               hf_token: str | None,
               cache_bucket: str,
               cache_region: str,
               warehouse_bucket: str,
               warehouse_region: str,
               repo_url: str,
               repo_branch: str,
               data_root: str = "/opt/dlami/nvme/eeg") -> str:
    env_lines: list[str] = []
    if wandb_api_key:
        env_lines.append(f'WANDB_API_KEY="{wandb_api_key}"')
    if hf_token:
        env_lines.append(f'HF_TOKEN="{hf_token}"')
    env_lines.extend([
        f'EXP03_DATA_ROOT="{data_root}"',
        f'EXP03_S3_BUCKET="{cache_bucket}"',
        f'EXP03_S3_REGION="{cache_region}"',
        f'S3_WAREHOUSE_BUCKET="{warehouse_bucket}"',
        f'S3_WAREHOUSE_REGION="{warehouse_region}"',
        f'REMOTE_REPO_URL="{repo_url}"',
        f'REMOTE_REPO_BRANCH="{repo_branch}"',
    ])
    env_block = "\n".join(f"export {line}" for line in env_lines)

    return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euxo pipefail
        exec > /var/log/eeg-bootstrap.log 2>&1

        # --- env block consumed by bootstrap ----------------------------
        mkdir -p /opt/eeg-config
        cat > /opt/eeg-config/env <<'INNER_EOF'
        {env_block}
        INNER_EOF
        chown -R ubuntu:ubuntu /opt/eeg-config

        # --- run the actual bootstrap as ubuntu -------------------------
        cat > /opt/eeg-config/bootstrap-stage1.sh <<'STAGE1_EOF'
        #!/usr/bin/env bash
        set -euxo pipefail
        source /opt/eeg-config/env
        export PATH="$HOME/.local/bin:$PATH"
        sudo mkdir -p "$EXP03_DATA_ROOT"
        sudo chown -R "$USER:$USER" "$EXP03_DATA_ROOT"
        mkdir -p "$EXP03_DATA_ROOT"/{{raw/hbn,raw/tuab,raw/tuev,derived,runs,models/hf_cache,scratch}}

        # uv
        if ! command -v uv >/dev/null 2>&1; then
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        fi

        # rclone
        if ! command -v rclone >/dev/null 2>&1; then
          curl -fsSL https://rclone.org/install.sh | sudo bash
        fi
        mkdir -p ~/.config/rclone
        cat > ~/.config/rclone/rclone.conf <<RCLONE_EOF
        [s3]
        type = s3
        provider = AWS
        env_auth = true
        region = $EXP03_S3_REGION

        [s3w]
        type = s3
        provider = AWS
        env_auth = true
        region = $S3_WAREHOUSE_REGION
        RCLONE_EOF

        # repo + venv (ssh key for github not yet set; first attempt is
        # https; fall back to deferred clone after user supplies token)
        cd ~
        if [[ ! -d eegModel ]]; then
          git clone "$REMOTE_REPO_URL" eegModel || \
            echo "[bootstrap] git clone failed; do it manually after SSH"
        fi
        if [[ -d eegModel ]]; then
          cd eegModel
          git fetch origin "$REMOTE_REPO_BRANCH" || true
          git checkout "$REMOTE_REPO_BRANCH" || true
          if [[ ! -d .venv ]]; then
            uv venv .venv --python 3.11
          fi
          source .venv/bin/activate
          uv pip install -e packages/eeg_common || true
          uv pip install -e packages/eeg_ops || true
          for exp in exp01_eeg_to_text exp02_eeg_ctc exp03_eeg_pretraining; do
            if [[ -f experiments/$exp/pyproject.toml ]]; then
              uv pip install -e "experiments/$exp[gpu]" || \
                uv pip install -e "experiments/$exp" || true
            fi
          done
          uv pip install -U "torch>=2.8" "s3torchconnector[dcp]>=1.4" \
                            mamba-ssm causal-conv1d wandb accelerate || true
        fi

        # logins
        if [[ -n "${{WANDB_API_KEY:-}}" ]]; then
          wandb login --relogin "$WANDB_API_KEY" || true
        fi
        if [[ -n "${{HF_TOKEN:-}}" ]]; then
          huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
        fi

        # sync derived shards from Mumbai mirror to NVMe
        for pl in hbn_minimal_500hz hbn_v2_clean_250hz tuab_v2_clean_250hz tuev_v2_clean_250hz; do
          if rclone lsd "s3:$EXP03_S3_BUCKET/derived/$pl" >/dev/null 2>&1; then
            rclone copy "s3:$EXP03_S3_BUCKET/derived/$pl" \
                        "$EXP03_DATA_ROOT/derived/$pl/" \
                        --transfers 64 --checkers 64 --s3-chunk-size 64M --quiet
          fi
        done

        nvidia-smi || true
        touch /opt/eeg-config/bootstrap-done
        STAGE1_EOF

        chmod +x /opt/eeg-config/bootstrap-stage1.sh
        sudo -u ubuntu bash /opt/eeg-config/bootstrap-stage1.sh
        echo "[user-data] complete"
    """)


# ---------------------------------------------------------------------------
# Direct boto3 launch
# ---------------------------------------------------------------------------


def launch_into_reservation(
    *,
    region_cfg: RegionConfig,
    reservation_id: str,
    instance_type: str,
    key_name: str,
    security_group_id: str,
    instance_profile_arn: str,
    repo_url: str,
    repo_branch: str = "main",
    name_tag: str = "eeg-mumbai",
    extra_tags: dict[str, str] | None = None,
    wandb_api_key: str | None = None,
    hf_token: str | None = None,
    wait_for_running: bool = True,
    wait_for_ssh: bool = True,
    ssh_timeout_s: int = 600,
) -> LaunchResult:
    """Launch one ``instance_type`` into an existing capacity reservation.

    This bypasses SkyPilot's global region scan entirely and uses the AWS
    EC2 ``RunInstances`` API directly. Returns a :class:`LaunchResult`
    populated with the instance ID and public IP, ready for SSH.
    """
    ec2 = boto3.client("ec2", region_name=region_cfg.region)

    user_data = _user_data(
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        cache_bucket=region_cfg.cache_bucket,
        cache_region=region_cfg.region,
        warehouse_bucket=WAREHOUSE_BUCKET,
        warehouse_region=WAREHOUSE_REGION,
        repo_url=repo_url,
        repo_branch=repo_branch,
    )

    tags = {"Name": name_tag, "cluster": name_tag, "billing": "research"}
    if extra_tags:
        tags.update(extra_tags)

    resp = ec2.run_instances(
        ImageId=region_cfg.dlami_pytorch_ami,
        InstanceType=instance_type,
        MinCount=1, MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        SubnetId=region_cfg.subnet_id,
        IamInstanceProfile={"Arn": instance_profile_arn},
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 200, "VolumeType": "gp3",
                "Iops": 3000, "Throughput": 250,
                "DeleteOnTermination": True,
            },
        }],
        CapacityReservationSpecification={
            "CapacityReservationTarget": {"CapacityReservationId": reservation_id}
        },
        TagSpecifications=[
            {"ResourceType": "instance",
             "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}
        ],
        UserData=user_data,
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
    )

    inst = resp["Instances"][0]
    instance_id = inst["InstanceId"]

    if wait_for_running:
        ec2.get_waiter("instance_running").wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 5, "MaxAttempts": 60},   # 5 min cap
        )

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    result = LaunchResult(
        instance_id=instance_id,
        public_ip=inst.get("PublicIpAddress"),
        private_ip=inst["PrivateIpAddress"],
        az=inst["Placement"]["AvailabilityZone"],
        region=region_cfg.region,
    )

    if wait_for_ssh and result.public_ip is not None:
        _wait_for_ssh(result.public_ip, timeout_s=ssh_timeout_s)

    return result


def _wait_for_ssh(host: str, *, port: int = 22, timeout_s: int = 600) -> None:
    """Poll TCP/22 until the box accepts a connection or we time out."""
    import socket
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"sshd on {host}:{port} did not come up within {timeout_s}s")


# ---------------------------------------------------------------------------
# Status / down — boto3 mirrors of `sky status` / `sky down`
# ---------------------------------------------------------------------------


def find_instances(*, region: str, cluster_tag: str) -> list[dict[str, Any]]:
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:cluster", "Values": [cluster_tag]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    out: list[dict[str, Any]] = []
    for r in resp.get("Reservations", []):
        for inst in r["Instances"]:
            out.append({
                "instance_id": inst["InstanceId"],
                "state": inst["State"]["Name"],
                "type": inst["InstanceType"],
                "public_ip": inst.get("PublicIpAddress"),
                "private_ip": inst.get("PrivateIpAddress"),
                "az": inst["Placement"]["AvailabilityZone"],
                "launch_time": inst.get("LaunchTime").isoformat()
                                if inst.get("LaunchTime") else None,
            })
    return out


def terminate_cluster(*, region: str, cluster_tag: str, dry_run: bool = False) -> int:
    """Terminate every pending/running/stopped instance tagged with the cluster tag."""
    ec2 = boto3.client("ec2", region_name=region)
    instances = find_instances(region=region, cluster_tag=cluster_tag)
    ids = [i["instance_id"] for i in instances if i["state"] != "terminated"]
    if not ids:
        return 0
    if dry_run:
        return 0
    try:
        ec2.terminate_instances(InstanceIds=ids)
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
            return 0
        raise
    return len(ids)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def remember_launch(state: State, *, cluster_name: str,
                    instance_id: str, public_ip: str | None) -> None:
    """Persist the launched instance ID + IP into state.toml.

    We piggyback on the existing State schema by stashing the IDs into the
    ``cluster_name`` field as ``"<name>:<instance_id>:<public_ip>"`` — a
    pragmatic shortcut so we don't widen the schema for a soon-to-be-
    superseded launcher (SkyPilot-via-network-fix is the long-term plan).
    """
    state.cluster_name = f"{cluster_name}:{instance_id}:{public_ip or '-'}"
    state.save()
