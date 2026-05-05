"""Notion ops-hub event logger.

Companion to the Notion databases that live under
"EEG Foundation Models — Operations Hub". This module is the single Python
surface that **writes** to those databases. Reads happen in the Notion app.

Auth backends
-------------
Two ways to talk to Notion, picked at runtime:

1. **Internal integration token** (preferred, future-proof). Set
   ``NOTION_API_KEY=secret_…`` in the environment. The module hits
   ``https://api.notion.com/v1/`` directly.

2. **Cursor-MCP / Composio backends** (transient, no code path here).
   When the operator works inside the Cursor agent, the AI itself can
   call the Notion MCP tools directly; this module's HTTP client is only
   used by code running on the GPU box / CI / cron, not the IDE.

Schema invariants
-----------------
The module knows the database IDs and required property names by name.
They were created by the chat session that built the hub; the IDs are
captured in :data:`HUB` below. To switch to a different workspace,
replace :data:`HUB` (or set ``EEG_OPS_NOTION_HUB_JSON`` to override at
runtime — the file format is the same dict).

We **never** create or edit databases here — only rows. The hub is a
human-stable artifact. If you want to add a column, do it in Notion's UI
or via ``notion-update-data-source`` and update :data:`HUB.events.properties`
afterwards.

Resilience
----------
All public functions are no-ops when ``NOTION_API_KEY`` is not set. They
also swallow HTTP errors and log to stderr — Notion being down should
never crash a training run.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import urllib.request
import urllib.error

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"


# ---------------------------------------------------------------------------
# Hub IDs — written by the chat session that built the workspace. Edit or
# override via $EEG_OPS_NOTION_HUB_JSON to repoint at a different workspace.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubIds:
    """Canonical IDs of the Operations Hub page and its 5 databases."""

    hub_page_id: str
    experiments_db_id: str
    sessions_db_id: str
    runs_db_id: str
    events_db_id: str
    findings_db_id: str

    def workspace_url(self, page_id: str) -> str:
        clean = page_id.replace("-", "")
        return f"https://www.notion.so/{clean}"


HUB = HubIds(
    hub_page_id="357939fb-cda4-813f-ab4b-c5fe0d84ea2c",
    experiments_db_id="c30098f0-e2ea-4d38-b678-0e16442c77bf",
    sessions_db_id="60c96c05-0c4f-4864-b6f9-21ea1556ba04",
    runs_db_id="e63827be-4630-4d40-8ddb-725f8583081c",
    events_db_id="5a05968e-ca13-4952-b4c7-df3e6062f89b",
    findings_db_id="4218525f-a4e9-437d-9bb4-3e7e9d59fb68",
)


def hub_ids() -> HubIds:
    """Resolve hub IDs (env override > package default)."""
    raw = os.environ.get("EEG_OPS_NOTION_HUB_JSON")
    if raw:
        d = json.loads(raw)
        return HubIds(**d)
    return HUB


# ---------------------------------------------------------------------------
# Allowed enum values — must match the Notion select option labels exactly.
# Defined here as constants so callers don't pass typos that silently
# get dropped on Notion's side.
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "cluster_up", "cluster_down",
    "capacity_buy", "capacity_status",
    "prewarm_start", "prewarm_done",
    "iam_create", "alarm_create",
    "run_started", "run_ended", "run_resumed", "run_crashed",
    "ckpt_saved", "ckpt_loaded",
    "eval_done",
    "session_start", "session_end",
    "error", "warning", "decision", "finding", "info",
)
SEVERITIES = ("info", "success", "warning", "error")
SOURCES = ("eeg-ops", "exp03/train", "exp03/cli", "agent", "manual")
SESSION_TYPES = ("chat", "gpu_rental", "training_run", "data_pipeline", "manual")
SESSION_STATUSES = ("open", "closed", "expired", "crashed")
RUN_STATUSES = ("queued", "running", "complete", "crashed", "killed", "resumed")


# ---------------------------------------------------------------------------
# Low-level Notion HTTP client (stdlib only — no requests dep).
# ---------------------------------------------------------------------------


class NotionUnavailable(RuntimeError):
    """Raised by the *strict* helpers when NOTION_API_KEY isn't set.

    The default helpers swallow this and warn rather than raise; use
    :func:`require_token` if you want a hard failure (e.g. in a CI job
    whose whole purpose is to log to Notion).
    """


def require_token() -> str:
    tok = os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
    if not tok:
        raise NotionUnavailable(
            "NOTION_API_KEY (or NOTION_TOKEN) is not set; "
            "create an internal integration at https://www.notion.so/profile/integrations "
            "and share the Operations Hub page with it."
        )
    return tok


def _http_request(method: str, path: str, body: dict | None = None,
                  *, timeout: float = 10.0) -> dict:
    """Tiny Notion REST client built on urllib (no extra deps)."""
    url = f"{NOTION_API_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {require_token()}")
    req.add_header("Notion-Version", NOTION_API_VERSION)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise NotionUnavailable(f"Notion API {e.code}: {detail}") from e


# ---------------------------------------------------------------------------
# Property builders — convert Python values to Notion property dicts.
# ---------------------------------------------------------------------------


def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text[:2000]}}]}


def _rich_text(text: str | None) -> dict:
    if not text:
        return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]}


def _select(name: str | None) -> dict:
    if name is None:
        return {"select": None}
    return {"select": {"name": name}}


def _multi_select(names: Iterable[str]) -> dict:
    return {"multi_select": [{"name": n} for n in names]}


def _date(start: datetime | str, end: datetime | str | None = None) -> dict:
    def _iso(v):
        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            return v.isoformat()
        return v
    payload: dict[str, Any] = {"start": _iso(start)}
    if end is not None:
        payload["end"] = _iso(end)
    return {"date": payload}


def _url(value: str | None) -> dict:
    return {"url": value or None}


def _number(value: float | int | None) -> dict:
    return {"number": float(value) if value is not None else None}


def _relation(page_ids: Iterable[str]) -> dict:
    return {"relation": [{"id": p} for p in page_ids if p]}


# ---------------------------------------------------------------------------
# High-level helpers — one function per operation that callers will do.
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """True iff a Notion token is set in the environment."""
    return bool(os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN"))


def _safe(fn):
    """Decorator: swallow only network/availability errors and warn.

    Validation errors (``ValueError`` from passing an unknown enum value,
    ``TypeError`` from a missing kwarg) propagate, since those are caller
    bugs that should fail loudly. Everything else (Notion outages, DNS
    failures, the env var simply not being set) returns ``None`` so a
    training run never crashes because Notion is unreachable.
    """
    def wrapper(*args, **kwargs):
        if not is_enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except (NotionUnavailable, OSError) as e:
            print(f"[notion] {fn.__name__} failed: {e}", file=sys.stderr)
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_safe
def log_event(
    *,
    title: str,
    type: str,
    severity: str = "info",
    source: str = "eeg-ops",
    resource: str | None = None,
    notes: str | None = None,
    timestamp: datetime | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> str | None:
    """Append a row to the Events database. Returns the new page ID, or
    None if Notion is disabled or the call failed."""
    if type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {type!r}; valid: {EVENT_TYPES}")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; valid: {SEVERITIES}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {SOURCES}")

    ts = timestamp or datetime.now(timezone.utc)
    props: dict[str, Any] = {
        "Title": _title(title),
        "Timestamp": _date(ts),
        "Type": _select(type),
        "Severity": _select(severity),
        "Source": _select(source),
        "Resource": _rich_text(resource),
        "Notes": _rich_text(notes),
    }
    if session_id:
        props["Session"] = _relation([session_id])
    if run_id:
        props["Run"] = _relation([run_id])

    resp = _http_request("POST", "/pages", {
        "parent": {"database_id": hub_ids().events_db_id},
        "properties": props,
    })
    return resp.get("id")


@_safe
def open_session(
    *,
    name: str,
    type: str,
    resource: str | None = None,
    source: str = "eeg-ops",
    started: datetime | None = None,
    cost_usd: float | None = None,
    outcome: str | None = None,
) -> str | None:
    """Create a new Session row with status=open. Returns the page ID."""
    if type not in SESSION_TYPES:
        raise ValueError(f"unknown session type {type!r}; valid: {SESSION_TYPES}")

    ts = started or datetime.now(timezone.utc)
    props: dict[str, Any] = {
        "Name": _title(name),
        "Type": _select(type),
        "Status": _select("open"),
        "Started": _date(ts),
        "Resource": _rich_text(resource),
        "Source": _select(source),
        "Outcome": _rich_text(outcome),
    }
    if cost_usd is not None:
        props["Cost USD"] = _number(cost_usd)

    resp = _http_request("POST", "/pages", {
        "parent": {"database_id": hub_ids().sessions_db_id},
        "properties": props,
    })
    sid = resp.get("id")
    log_event(
        title=f"Session opened: {name}",
        type="session_start",
        severity="info",
        source=source if source in SOURCES else "eeg-ops",  # type: ignore[arg-type]
        resource=resource,
        timestamp=ts,
        session_id=sid,
    )
    return sid


@_safe
def close_session(
    *,
    session_id: str,
    outcome: str | None = None,
    status: str = "closed",
    ended: datetime | None = None,
) -> None:
    """Close (or mark crashed/expired) a session and log a session_end event."""
    if status not in SESSION_STATUSES:
        raise ValueError(f"unknown session status {status!r}; valid: {SESSION_STATUSES}")
    ts = ended or datetime.now(timezone.utc)
    props: dict[str, Any] = {
        "Status": _select(status),
        "Ended": _date(ts),
    }
    if outcome:
        props["Outcome"] = _rich_text(outcome)
    _http_request("PATCH", f"/pages/{session_id}", {"properties": props})
    log_event(
        title=f"Session {status}: {session_id[:8]}",
        type="session_end",
        severity="success" if status == "closed" else "warning",
        source="eeg-ops",
        notes=outcome,
        timestamp=ts,
        session_id=session_id,
    )


@_safe
def create_run(
    *,
    run_name: str,
    paradigm: str,
    region: str = "ap-south-1",
    experiment_id: str | None = None,
    s3_ckpt_uri: str | None = None,
    wandb_url: str | None = None,
    started: datetime | None = None,
    notes: str | None = None,
) -> str | None:
    """Insert a Run row at run start (status=running). Returns page ID."""
    ts = started or datetime.now(timezone.utc)
    props: dict[str, Any] = {
        "Run Name": _title(run_name),
        "Status": _select("running"),
        "Paradigm": _select(paradigm if paradigm in {"mae", "ar", "mar", "jepa"} else "other"),
        "Region": _select(region if region in {"ap-south-1", "us-west-2", "us-east-1",
                                                "lambda", "gcp"} else "other"),
        "Started": _date(ts),
        "S3 Checkpoint URI": _url(s3_ckpt_uri),
        "WandB URL": _url(wandb_url),
        "Notes": _rich_text(notes),
    }
    if experiment_id:
        props["Experiment"] = _relation([experiment_id])

    resp = _http_request("POST", "/pages", {
        "parent": {"database_id": hub_ids().runs_db_id},
        "properties": props,
    })
    rid = resp.get("id")
    log_event(
        title=f"Run started: {run_name}",
        type="run_started",
        severity="info",
        source="exp03/train",
        resource=run_name,
        timestamp=ts,
        run_id=rid,
        notes=notes,
    )
    return rid


@_safe
def update_run(
    *,
    run_id: str,
    status: str | None = None,
    steps_completed: int | None = None,
    final_loss: float | None = None,
    s3_ckpt_uri: str | None = None,
    ended: datetime | None = None,
    notes: str | None = None,
) -> None:
    """Patch a Run row. Use at end of training (or to record a crash)."""
    props: dict[str, Any] = {}
    if status is not None:
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}; valid: {RUN_STATUSES}")
        props["Status"] = _select(status)
    if steps_completed is not None:
        props["Steps Completed"] = _number(steps_completed)
    if final_loss is not None:
        props["Final Loss"] = _number(final_loss)
    if s3_ckpt_uri is not None:
        props["S3 Checkpoint URI"] = _url(s3_ckpt_uri)
    if ended is not None:
        props["Ended"] = _date(ended)
    if notes is not None:
        props["Notes"] = _rich_text(notes)
    if not props:
        return
    _http_request("PATCH", f"/pages/{run_id}", {"properties": props})


@_safe
def log_finding(
    *,
    title: str,
    summary: str,
    tags: Iterable[str] = (),
    confidence: str = "medium",
    experiment_id: str | None = None,
    run_id: str | None = None,
    implication: str | None = None,
    action_items: str | None = None,
    date: datetime | None = None,
) -> str | None:
    """Insert a row in the Findings database."""
    ts = date or datetime.now(timezone.utc)
    props: dict[str, Any] = {
        "Title": _title(title),
        "Date": _date(ts),
        "Tags": _multi_select(tags),
        "Confidence": _select(confidence),
        "Summary": _rich_text(summary),
        "Implication": _rich_text(implication),
        "Action Items": _rich_text(action_items),
    }
    if experiment_id:
        props["Experiment"] = _relation([experiment_id])
    if run_id:
        props["Run"] = _relation([run_id])

    resp = _http_request("POST", "/pages", {
        "parent": {"database_id": hub_ids().findings_db_id},
        "properties": props,
    })
    fid = resp.get("id")
    log_event(
        title=f"Finding: {title}",
        type="finding",
        severity="info",
        source="manual",
        notes=summary,
        timestamp=ts,
    )
    return fid


# ---------------------------------------------------------------------------
# Convenience: read-only helpers used by the CLI and by training resume.
# ---------------------------------------------------------------------------


@dataclass
class ActiveSessions:
    """Snapshot of currently-open sessions, by type."""

    chat: list[str] = field(default_factory=list)
    gpu_rental: list[str] = field(default_factory=list)
    training_run: list[str] = field(default_factory=list)


def find_open_sessions() -> ActiveSessions:
    """Query the Sessions database for rows with Status=='open'."""
    if not is_enabled():
        return ActiveSessions()
    try:
        resp = _http_request("POST", f"/databases/{hub_ids().sessions_db_id}/query", {
            "filter": {"property": "Status", "select": {"equals": "open"}},
            "page_size": 50,
        })
    except (NotionUnavailable, OSError) as e:
        print(f"[notion] find_open_sessions failed: {e}", file=sys.stderr)
        return ActiveSessions()

    out = ActiveSessions()
    for row in resp.get("results", []):
        type_prop = row["properties"].get("Type", {}).get("select")
        t = type_prop.get("name") if type_prop else None
        getattr(out, t.replace("_", "_"), None)  # type: ignore[arg-type]
        if t == "chat":
            out.chat.append(row["id"])
        elif t == "gpu_rental":
            out.gpu_rental.append(row["id"])
        elif t == "training_run":
            out.training_run.append(row["id"])
    return out
