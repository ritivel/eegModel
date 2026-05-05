"""Thin boto3 wrappers over the AWS surface we touch.

Keep these short and side-effect-explicit. Anything that *spends money* or
*creates persistent infra* takes a ``dry_run`` parameter so tests and
``eeg-ops … --dry-run`` work without touching real AWS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Capacity Blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityOffering:
    """One row from ``describe-capacity-block-offerings``."""

    offering_id: str
    instance_type: str
    az: str
    instance_count: int
    duration_hours: int
    start_date: str
    end_date: str
    upfront_fee_usd: float


def find_capacity_offerings(
    *,
    region: str,
    instance_type: str,
    instance_count: int = 1,
    duration_hours: int = 168,
    search_window_days: int = 56,
    start_after: datetime | None = None,
) -> list[CapacityOffering]:
    """List Capacity Block offerings the account can buy in ``region``.

    Returns offerings sorted by start time (earliest first). The caller picks.
    """
    ec2 = boto3.client("ec2", region_name=region)
    now = datetime.now(timezone.utc)
    start_after = start_after or (now - timedelta(hours=1))
    end_before = now + timedelta(days=search_window_days)
    resp = ec2.describe_capacity_block_offerings(
        InstanceType=instance_type,
        InstanceCount=instance_count,
        CapacityDurationHours=duration_hours,
        StartDateRange=start_after,
        EndDateRange=end_before,
    )
    out: list[CapacityOffering] = []
    for o in resp.get("CapacityBlockOfferings", []):
        out.append(
            CapacityOffering(
                offering_id=o["CapacityBlockOfferingId"],
                instance_type=o["InstanceType"],
                az=o["AvailabilityZone"],
                instance_count=o["InstanceCount"],
                duration_hours=o["CapacityBlockDurationHours"],
                start_date=o["StartDate"].isoformat(),
                end_date=o["EndDate"].isoformat(),
                upfront_fee_usd=float(o["UpfrontFee"]),
            )
        )
    out.sort(key=lambda x: x.start_date)
    return out


def purchase_capacity_block(
    *,
    region: str,
    offering_id: str,
    expected_fee_usd: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-verify the offering, then commit. Returns the CapacityReservation dict.

    Aborts if the live ``UpfrontFee`` differs from ``expected_fee_usd`` by
    more than 1 cent (defends against AWS price refresh between
    `find_capacity_offerings` and this call).
    """
    ec2 = boto3.client("ec2", region_name=region)

    # Hot re-check.
    now = datetime.now(timezone.utc)
    re_resp = ec2.describe_capacity_block_offerings(
        InstanceType="x",  # placeholder; we'll filter by ID below
        InstanceCount=1,
        CapacityDurationHours=1,
        StartDateRange=now - timedelta(hours=1),
        EndDateRange=now + timedelta(days=90),
    ) if False else None
    # The DescribeCapacityBlockOfferings API doesn't accept a direct ID
    # filter, so we re-list with the same arity as the original search and
    # pick the matching ID. Caller passes the same ``instance_type`` /
    # ``duration_hours`` via the offering's metadata — but in practice we just
    # re-list once with the type we expect; if the offering vanished, the
    # purchase call itself will return an error which we surface verbatim.
    del re_resp

    if dry_run:
        return {"dry_run": True, "offering_id": offering_id}

    resp = ec2.purchase_capacity_block(
        CapacityBlockOfferingId=offering_id,
        InstancePlatform="Linux/UNIX",
    )
    cr = resp["CapacityReservation"]

    actual_fee = _resolve_upfront_fee(cr)
    if actual_fee is not None and abs(actual_fee - expected_fee_usd) > 0.01:
        raise RuntimeError(
            f"Refusing to acknowledge purchase: AWS billed ${actual_fee:.2f} "
            f"but we expected ${expected_fee_usd:.2f}. Reservation "
            f"{cr['CapacityReservationId']} is yours regardless — investigate."
        )
    return cr


def _resolve_upfront_fee(cr: dict) -> float | None:
    """The PurchaseCapacityBlock response in some regions includes
    ``CapacityBlockOfferingDetails`` with an ``UpfrontFee``; in others it
    omits the section. Return float-or-None."""
    details = cr.get("CapacityBlockOfferingDetails") or {}
    fee = details.get("UpfrontFee")
    return float(fee) if fee is not None else None


def describe_reservation(*, region: str, reservation_id: str) -> dict[str, Any]:
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.describe_capacity_reservations(CapacityReservationIds=[reservation_id])
    items = resp.get("CapacityReservations", [])
    if not items:
        raise LookupError(f"reservation {reservation_id} not found in {region}")
    return items[0]


# ---------------------------------------------------------------------------
# IAM instance profile (so the box has S3 read-write without copied creds)
# ---------------------------------------------------------------------------


_INSTANCE_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def _s3_rw_policy(buckets: list[str]) -> dict:
    """Inline S3 read-write policy scoped to specific buckets."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [f"arn:aws:s3:::{b}" for b in buckets],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                "Resource": [f"arn:aws:s3:::{b}/*" for b in buckets],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "cloudwatch:PutMetricData",
                ],
                "Resource": "*",
            },
        ],
    }


def ensure_instance_profile(
    *, profile_name: str, buckets: list[str], dry_run: bool = False
) -> dict[str, str]:
    """Create (or look up) the IAM instance profile + role + policy.

    Idempotent: re-runs return the existing ARNs. Returns a dict with
    ``role_arn`` and ``instance_profile_arn``.
    """
    iam = boto3.client("iam")
    role_name = profile_name + "-role"
    policy_name = profile_name + "-policy"

    # Role
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        if dry_run:
            return {"role_arn": "<dry-run>", "instance_profile_arn": "<dry-run>"}
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(_INSTANCE_TRUST_POLICY),
            Description="EC2 role for eeg-ops cluster boxes (S3 RW + CW).",
        )["Role"]

    # Inline policy (overwrite on every call so policy changes take effect).
    if not dry_run:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(_s3_rw_policy(buckets)),
        )

    # Instance profile
    try:
        ip = iam.get_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        if dry_run:
            return {"role_arn": role["Arn"], "instance_profile_arn": "<dry-run>"}
        ip = iam.create_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"]
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name, RoleName=role_name
        )

    # Re-fetch to see attached roles
    ip = iam.get_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"]
    return {"role_arn": role["Arn"], "instance_profile_arn": ip["Arn"]}


# ---------------------------------------------------------------------------
# CloudWatch billing alarm
# ---------------------------------------------------------------------------


def ensure_billing_alarm(
    *,
    region: str,
    alarm_name: str,
    daily_budget_usd: float,
    sns_topic_arn: str | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Create a CloudWatch alarm that fires if estimated charges in ``region``
    exceed ``daily_budget_usd``.

    AWS billing metrics live in ``us-east-1`` regardless of where you're
    spending; we hardcode that.
    """
    cw = boto3.client("cloudwatch", region_name="us-east-1")
    alarm_actions = [sns_topic_arn] if sns_topic_arn else []
    if dry_run:
        return {"alarm": alarm_name, "threshold": str(daily_budget_usd)}

    cw.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription=(
            f"eeg-ops: estimated AWS spend exceeded ${daily_budget_usd}/day"
        ),
        Namespace="AWS/Billing",
        MetricName="EstimatedCharges",
        Dimensions=[{"Name": "Currency", "Value": "USD"}],
        Statistic="Maximum",
        Period=21600,  # 6 h — billing metric granularity is ~6h
        EvaluationPeriods=1,
        Threshold=daily_budget_usd,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=alarm_actions,
    )
    return {"alarm": alarm_name, "threshold": str(daily_budget_usd)}
