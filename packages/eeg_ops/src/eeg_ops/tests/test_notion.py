"""Notion module tests.

We never call the real Notion API in CI — instead we monkeypatch
:func:`eeg_ops.notion._http_request` and assert that the right payloads
are constructed and the right safety rails fire when ``NOTION_API_KEY``
is missing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eeg_ops import notion as N


@pytest.fixture
def fake_http(monkeypatch):
    """Replace _http_request with a recorder that pretends to succeed."""
    calls: list[tuple[str, str, dict | None]] = []

    def _fake(method: str, path: str, body: dict | None = None,
              *, timeout: float = 10.0) -> dict:
        calls.append((method, path, body))
        # Return a minimal fake response shape.
        return {"id": f"fake-{len(calls)}-id", "results": []}

    monkeypatch.setenv("NOTION_API_KEY", "secret_test_token")
    monkeypatch.setattr(N, "_http_request", _fake)
    return calls


def test_is_enabled_true_when_token_set(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "secret_x")
    assert N.is_enabled()


def test_is_enabled_false_without_token(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    assert not N.is_enabled()


def test_log_event_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    assert N.log_event(title="x", type="info") is None


def test_log_event_validates_type(fake_http):
    with pytest.raises(ValueError):
        N.log_event(title="x", type="not_a_real_type")


def test_log_event_payload_shape(fake_http):
    pid = N.log_event(
        title="hello",
        type="info",
        severity="success",
        source="eeg-ops",
        resource="my-resource",
        notes="some notes",
        timestamp=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
    assert pid == "fake-1-id"
    method, path, body = fake_http[0]
    assert method == "POST"
    assert path == "/pages"
    assert body["parent"]["database_id"] == N.HUB.events_db_id
    props = body["properties"]
    assert props["Title"]["title"][0]["text"]["content"] == "hello"
    assert props["Type"]["select"]["name"] == "info"
    assert props["Severity"]["select"]["name"] == "success"
    assert props["Source"]["select"]["name"] == "eeg-ops"
    assert props["Resource"]["rich_text"][0]["text"]["content"] == "my-resource"
    assert props["Timestamp"]["date"]["start"].startswith("2026-05-05T12:00")


def test_log_event_with_session_and_run_relations(fake_http):
    N.log_event(
        title="ckpt",
        type="ckpt_saved",
        session_id="ses-xxx",
        run_id="run-yyy",
    )
    _, _, body = fake_http[-1]
    assert body["properties"]["Session"]["relation"] == [{"id": "ses-xxx"}]
    assert body["properties"]["Run"]["relation"] == [{"id": "run-yyy"}]


def test_open_session_creates_then_logs_event(fake_http):
    sid = N.open_session(name="test-rental", type="gpu_rental",
                         resource="cr-x", cost_usd=42.0)
    assert sid == "fake-1-id"
    # Two HTTP calls: create the session row, then log the session_start event.
    assert len(fake_http) == 2
    create_path = fake_http[0][1]
    event_body = fake_http[1][2]
    assert create_path == "/pages"
    assert event_body["properties"]["Type"]["select"]["name"] == "session_start"
    assert event_body["properties"]["Session"]["relation"] == [{"id": "fake-1-id"}]


def test_close_session_patches_then_logs_event(fake_http):
    N.close_session(session_id="ses-zzz", outcome="all done", status="closed")
    method0, path0, body0 = fake_http[0]
    method1, path1, body1 = fake_http[1]
    assert method0 == "PATCH"
    assert path0 == "/pages/ses-zzz"
    assert body0["properties"]["Status"]["select"]["name"] == "closed"
    assert body1["properties"]["Type"]["select"]["name"] == "session_end"


def test_create_run_payload_has_relations_and_url(fake_http):
    rid = N.create_run(
        run_name="exp17-g2-seed0",
        paradigm="mar",
        region="ap-south-1",
        experiment_id="exp-xxx",
        s3_ckpt_uri="s3://bucket/prefix",
        wandb_url="https://wandb.ai/x/y/runs/z",
    )
    assert rid == "fake-1-id"
    create_body = fake_http[0][2]
    props = create_body["properties"]
    assert props["Run Name"]["title"][0]["text"]["content"] == "exp17-g2-seed0"
    assert props["Paradigm"]["select"]["name"] == "mar"
    assert props["Region"]["select"]["name"] == "ap-south-1"
    assert props["Experiment"]["relation"] == [{"id": "exp-xxx"}]
    assert props["S3 Checkpoint URI"]["url"] == "s3://bucket/prefix"
    assert props["WandB URL"]["url"] == "https://wandb.ai/x/y/runs/z"


def test_update_run_only_sets_provided_fields(fake_http):
    N.update_run(run_id="run-1", status="complete", steps_completed=17500,
                 final_loss=0.123)
    method, path, body = fake_http[0]
    assert method == "PATCH"
    assert path == "/pages/run-1"
    props = body["properties"]
    assert "Status" in props
    assert "Steps Completed" in props
    assert "Final Loss" in props
    # Fields we didn't set must be absent (no Region, no S3 URI overwrite).
    assert "Region" not in props
    assert "S3 Checkpoint URI" not in props


def test_log_finding_writes_finding_then_event(fake_http):
    fid = N.log_finding(
        title="MAR mul=4 unstable",
        summary="Loss diverges after step 800.",
        tags=["negative_result", "bug_or_artifact"],
        confidence="high",
        experiment_id="exp-17",
        run_id="run-7",
    )
    assert fid == "fake-1-id"
    finding_body = fake_http[0][2]
    event_body = fake_http[1][2]
    assert finding_body["parent"]["database_id"] == N.HUB.findings_db_id
    finding_props = finding_body["properties"]
    assert finding_props["Tags"]["multi_select"] == [
        {"name": "negative_result"}, {"name": "bug_or_artifact"}
    ]
    assert finding_props["Confidence"]["select"]["name"] == "high"
    assert finding_props["Experiment"]["relation"] == [{"id": "exp-17"}]
    assert event_body["properties"]["Type"]["select"]["name"] == "finding"


def test_property_builders_handle_none_values():
    # rich_text with None content
    assert N._rich_text(None) == {"rich_text": []}
    # select with None
    assert N._select(None) == {"select": None}
    # url with None
    assert N._url(None) == {"url": None}
    # number with None
    assert N._number(None) == {"number": None}


def test_safe_decorator_swallows_http_errors(monkeypatch):
    """When the API call raises NotionUnavailable, the helper returns None
    rather than propagating the exception."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_x")

    def _boom(method, path, body=None, **kw):
        raise N.NotionUnavailable("simulated API outage")

    monkeypatch.setattr(N, "_http_request", _boom)
    assert N.log_event(title="x", type="info") is None
    assert N.create_run(run_name="r", paradigm="mae") is None
