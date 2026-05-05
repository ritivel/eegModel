"""Boto3 helper smoke tests, mocking the AWS surface where possible.

These deliberately don't talk to a live AWS account — boto3's `Stubber`
doesn't cover everything we use, so for the resource-creation paths we
just verify ``dry_run=True`` returns sane shape and doesn't crash.
"""

from __future__ import annotations

from eeg_ops.aws import _s3_rw_policy, ensure_billing_alarm, ensure_instance_profile


def test_s3_rw_policy_includes_bucket_and_object_arns():
    pol = _s3_rw_policy(["a", "b"])
    # Two of three Statement entries (S3 ones) carry list[Resource]; the
    # CloudWatch / logs entry uses a single "*" wildcard.
    arns: list[str] = []
    for stmt in pol["Statement"]:
        r = stmt["Resource"]
        arns.extend(r if isinstance(r, list) else [r])
    assert "arn:aws:s3:::a" in arns
    assert "arn:aws:s3:::a/*" in arns
    assert "arn:aws:s3:::b" in arns
    assert "arn:aws:s3:::b/*" in arns
    assert "*" in arns                  # CloudWatch / logs wildcard


def test_ensure_instance_profile_dry_run_returns_marker(monkeypatch):
    out = ensure_instance_profile(
        profile_name="test-profile", buckets=["x"], dry_run=True,
    )
    # On dry-run we either short-circuit before any boto call (if NoSuchEntity)
    # or return marker strings. Either way both keys are present.
    assert "role_arn" in out
    assert "instance_profile_arn" in out


def test_ensure_billing_alarm_dry_run_returns_threshold():
    out = ensure_billing_alarm(
        region="ap-south-1", alarm_name="t", daily_budget_usd=42.0, dry_run=True,
    )
    assert out["alarm"] == "t"
    assert out["threshold"] == "42.0"
