"""TUH-EEG ingestion: walk the local rsync'd tree, read EDF via MNE,
parse `.rec` event annotations, and derive the labels we need for the
§4.3 Protocol A.4 secondary eval (TUAB binary AUROC + TUEV 6-class
BAC + weighted F1).

Source: NEDC SFTP host ``nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/``
(credentialed). We do *not* talk to the SFTP server programmatically — the
canonical data movement is ``rsync`` (driven from the CLI subcommand
``eegfm tuh-rsync``), which mirrors a local copy under
``$EEG_DATA_ROOT/raw/{tuab,tuev}/``. This module then walks that local
copy.

Two corpora are supported:

* **TUAB** (Temple University Abnormal EEG, v3.0.1, 2026-01-20) — binary
  normal/abnormal classification. Layout::

        edf/{train,eval}/{normal,abnormal}/01_tcp_ar/<subj>/<sess>/<rec>.edf

  The label is read directly from the path (``normal`` → 0, ``abnormal``
  → 1). The split (``train`` vs ``eval``) is also in the path; we
  preserve it on each row so downstream code can train + eval on the
  *official* splits without LNSO-style subject randomisation.

* **TUEV** (Temple University Event Corpus, v2.0.1, 2026-01-20) —
  6-class event classification (SPSW / GPED / PLED / EYEM / ARTF /
  BCKG). Layout::

        edf/{train,eval}/<subj>/<rec>.edf
        edf/{train,eval}/<subj>/<rec>.rec   ← annotation file

  Each ``.rec`` file is a CSV-like ``channel,start_s,end_s,label_int``
  format with labels 1..6 (see ``TUEV_REC_LABEL_BY_INT``). For our iid-
  channel pretraining recipe (each ``(subject, channel, 4-sec window)``
  is one example) we collapse the per-channel annotations to a *per-
  window file-level* label by summing event durations across all
  channels within the window and taking argmax. This is appropriate for
  the random-init linear-probe floor; a future fine-tuned cell can
  switch to per-channel labelling if the TCP montage is applied
  upstream (see ``mini_experiments/03_backbone_ablation/`` for that
  question).

References:

    TUH EEG Corpus / NEDC:
        https://isip.piconepress.com/projects/nedc/html/tuh_eeg/
    TUEV README + 6-event taxonomy:
        ``data/tuh_eeg/tuh_eeg_events/v2.0.1/AAREADME.txt``
    TUAB v3.0 paper (Lopez et al. 2017 IEEE SPMB):
        https://www.isip.piconepress.com/publications/conference_proceedings/2017/ieee_spmb/abnormal/
    Original TUH EEG Corpus paper:
        Obeid & Picone 2016 Frontiers Neurosci, doi:10.3389/fnins.2016.00196
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Corpus codes the CLI accepts and the on-disk layout uses.
CORPORA = ("tuab", "tuev")
TUH_CORPUS_T = Literal["tuab", "tuev"]

# The version directories pinned at NEDC's bucket as of 2026-01-20. These are
# the canonical "latest" versions; bump when NEDC publishes new releases.
TUAB_VERSION = "v3.0.1"
TUEV_VERSION = "v2.0.1"
DEFAULT_VERSION_BY_CORPUS: dict[str, str] = {
    "tuab": TUAB_VERSION,
    "tuev": TUEV_VERSION,
}

# Remote subdirectories under ``data/tuh_eeg/`` on the NEDC SFTP host. The
# client's home is ``data/tuh_eeg/`` itself, hence the visible
# ``tuh_eeg_abnormal`` / ``tuh_eeg_events`` siblings (not ``data/...`` per
# the AAREADME's own `./edf/...` notation).
REMOTE_CORPUS_SUBDIR: dict[str, str] = {
    "tuab": "tuh_eeg/tuh_eeg_abnormal",
    "tuev": "tuh_eeg/tuh_eeg_events",
}

# SSH host alias added to ~/.ssh/config 2026-05-04 routing through
# ~/.ssh/id_ed25519_nedc. Used as the rsync source host.
NEDC_SSH_HOST_ALIAS = "nedc-tuh"

# TUAB binary classification: subdirectory name → label. Anything else
# (a path that doesn't include either token) → -1 missing.
TUAB_LABEL_BY_SUBDIR: dict[str, int] = {
    "normal": 0,
    "abnormal": 1,
}

# TUEV 6-event classification. Per the v2.0.1 AAREADME:
#
#     1: spsw  spike and slow wave
#     2: gped  generalised periodic epileptiform discharge
#     3: pled  periodic lateralised epileptiform discharge
#     4: eyem  eye movement
#     5: artf  artifact
#     6: bckg  background
#
# We map the 1-based REC integer → 0-based class index for downstream
# sklearn / numpy use. ``TUEV_LABEL_NAMES[i]`` is the canonical 4-letter
# code for class index ``i``.
TUEV_LABEL_NAMES: tuple[str, ...] = ("spsw", "gped", "pled", "eyem", "artf", "bckg")
TUEV_REC_LABEL_BY_INT: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
TUEV_BCKG_INDEX = 5  # default fallback label when a window has no overlap with any .rec row

# Filename → ``(subject, recording_num)`` for TUEV. Train-set filenames look
# like ``<subj>_00000001.edf`` (8 hex chars + 8 digits). Eval-set filenames
# look like ``<class>_<index>_a_.edf`` per the AAREADME. The same parser
# handles both — we don't need the recording_num downstream.
_TUEV_FILENAME_RE = re.compile(r"^(?P<stem>.+)\.edf$", re.IGNORECASE)

# TUAB v3.0.1 flattens the per-subject / per-session subtree: each .edf
# lives directly under ``<split>/<class>/<montage>/`` and the subject /
# session / recording numbers live in the filename:
#
#     <subj>_s<sess>_t<rec>.edf      e.g. aaaaaicc_s003_t000.edf
#
# Older TUAB releases used a deeper layout (`<subj>/<sess>/<recname>.edf`).
# The filename regex below supports both — :class:`Recording.subject_id`
# is parsed from the filename, not the path, so we don't depend on the
# parent-directory shape.
_TUAB_PATH_RE = re.compile(
    r"/(?P<split>train|eval)/(?P<class>normal|abnormal)/(?P<montage>[^/]+)/"
    r"(?:(?P<subj_dir>[^/]+)/(?P<sess_dir>[^/]+)/)?"
    r"(?P<recname>[^/]+\.edf)$",
    re.IGNORECASE,
)
_TUAB_FILENAME_RE = re.compile(
    r"^(?P<subj>[A-Za-z0-9]+)_s(?P<sess>\d+)_t(?P<rec>\d+)\.edf$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Recording dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """One TUH ``.edf`` recording (and, for TUEV, its sibling ``.rec``).

    Identifiers and per-corpus extras::

        corpus       "tuab" or "tuev"
        version      e.g. "v3.0.1" — the on-disk corpus version
        split        "train" or "eval" (the official NEDC split)
        subject_id   the corpus-specific subject token from the path:
                       TUAB → 8-char hash like "aaaaaaaa"
                       TUEV → 8-char hash (train) OR 3-digit index (eval)
        edf_path     absolute Path to the .edf
        rec_path     absolute Path to .rec (TUEV only; None for TUAB)
        montage      e.g. "01_tcp_ar" (TUAB always; TUEV not enforced).
                     None if the path doesn't expose it.
        tuab_label   0 normal / 1 abnormal / -1 missing (TUAB only)

    The ``.edf`` is the only mandatory file. ``.rec`` is required for TUEV
    label derivation; we surface it explicitly so the preprocess pass
    can warn loudly if it's missing rather than silently emitting -1.
    """

    corpus: TUH_CORPUS_T
    version: str
    split: str
    subject_id: str
    edf_path: Path
    rec_path: Path | None = None
    montage: str | None = None
    tuab_label: int = -1

    @property
    def basename(self) -> str:
        """Filename stem without ``.edf``."""
        return self.edf_path.stem


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------


def _walk_tuab(corpus_root: Path, version: str) -> list[Recording]:
    """Walk the local TUAB tree under ``corpus_root/edf/{train,eval}/...``.

    Supports both the v3.0.1 flat layout (``<split>/<class>/<montage>/<recname>.edf``)
    and the older deep layout (``<split>/<class>/<montage>/<subj>/<sess>/<recname>.edf``).
    Subject ID is parsed from the filename, which is uniform across versions:
    ``<subj>_s<sess>_t<rec>.edf``.
    """
    edf_root = corpus_root / "edf"
    if not edf_root.exists():
        raise FileNotFoundError(
            f"TUAB layout expected at {edf_root}; did rsync complete? "
            f"Try `eegfm tuh-rsync tuab` first."
        )

    out: list[Recording] = []
    for path in edf_root.rglob("*.edf"):
        m = _TUAB_PATH_RE.search(str(path))
        if not m:
            continue
        d = m.groupdict()
        klass = d["class"].lower()
        # Subject ID from filename (uniform across versions).
        fm = _TUAB_FILENAME_RE.match(path.name)
        if fm:
            subj = fm.group("subj")
        elif d.get("subj_dir"):
            subj = d["subj_dir"]
        else:
            # Last-resort fallback: use the filename stem.
            subj = path.stem
        out.append(Recording(
            corpus="tuab",
            version=version,
            split=d["split"].lower(),
            subject_id=subj,
            edf_path=path,
            rec_path=None,
            montage=d["montage"],
            tuab_label=TUAB_LABEL_BY_SUBDIR.get(klass, -1),
        ))
    return out


def _walk_tuev(corpus_root: Path, version: str) -> list[Recording]:
    """Walk the local TUEV tree under ``corpus_root/edf/{train,eval}/<subj>/...``.

    TUEV layout is flatter than TUAB: each subject directory contains the
    ``.edf`` recording files directly (no per-session/per-montage subtree).
    The sibling ``.rec`` is the authoritative event annotation source.
    """
    edf_root = corpus_root / "edf"
    if not edf_root.exists():
        raise FileNotFoundError(
            f"TUEV layout expected at {edf_root}; did rsync complete? "
            f"Try `eegfm tuh-rsync tuev` first."
        )

    out: list[Recording] = []
    for split in ("train", "eval"):
        split_root = edf_root / split
        if not split_root.exists():
            continue
        for subj_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for edf_path in sorted(subj_dir.rglob("*.edf")):
                rec_path = edf_path.with_suffix(".rec")
                out.append(Recording(
                    corpus="tuev",
                    version=version,
                    split=split,
                    subject_id=subj_dir.name,
                    edf_path=edf_path,
                    rec_path=rec_path if rec_path.exists() else None,
                ))
    return out


def walk_corpus(corpus_root: Path, corpus: TUH_CORPUS_T,
                version: str | None = None) -> list[Recording]:
    """Enumerate all recordings under one corpus root on local NVMe."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus: {corpus!r}; valid: {CORPORA}")
    version = version or DEFAULT_VERSION_BY_CORPUS[corpus]
    if corpus == "tuab":
        return _walk_tuab(corpus_root, version)
    return _walk_tuev(corpus_root, version)


# ---------------------------------------------------------------------------
# .rec annotation parsing  (TUEV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecEvent:
    """One row of a TUEV ``.rec`` file: a labelled (channel, time-interval)."""
    channel_idx: int       # TCP-montage channel 0..21 (per AAREADME)
    start_s: float
    end_s: float
    label: int             # 0..5  (already mapped from the 1..6 REC integer)


def parse_rec(rec_path: Path) -> list[RecEvent]:
    """Parse a TUEV ``.rec`` file into a list of RecEvent.

    Format per the AAREADME: each line is
        ``channel,start_s,end_s,label_int``
    with ``label_int`` ∈ {1..6}. Whitespace tolerant; blank lines skipped.

    Raises ValueError on a malformed line — the file format is well-
    specified and any deviation is a flag for downstream-corruption
    investigation, not a silent skip.
    """
    rec_path = Path(rec_path)
    if not rec_path.exists():
        raise FileNotFoundError(rec_path)
    out: list[RecEvent] = []
    with rec_path.open("r", encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"{rec_path.name}:{ln}: expected 4 fields, got {len(parts)}: {line!r}")
            try:
                ch = int(parts[0])
                t0 = float(parts[1])
                t1 = float(parts[2])
                lab_int = int(parts[3])
            except ValueError as e:
                raise ValueError(f"{rec_path.name}:{ln}: bad numeric field: {line!r}") from e
            lab = TUEV_REC_LABEL_BY_INT.get(lab_int)
            if lab is None:
                raise ValueError(f"{rec_path.name}:{ln}: bad label {lab_int} (expected 1..6)")
            out.append(RecEvent(channel_idx=ch, start_s=t0, end_s=t1, label=lab))
    return out


def tuev_window_label(events: list[RecEvent], window_start_s: float,
                      window_seconds: float = 4.0) -> int:
    """Per-window TUEV label by argmax-overlap-duration.

    For each window ``[window_start_s, window_start_s + window_seconds)``,
    sum the overlapping seconds *across all channels and all events* per
    label class, and return ``argmax``. If no events overlap (rare but
    possible — early/late windows on quiet recordings), return BCKG (5)
    as the conservative default.

    Channel-collapsing is intentional. For our iid-channel pretraining
    recipe, each row in the parquet shard is one ``(subject, channel,
    window)`` triple sharing the same window-level label. A future
    per-channel TCP-montage cell can re-derive a finer-grained label
    from the same parsed events.
    """
    win_end = window_start_s + window_seconds
    bins = np.zeros(len(TUEV_LABEL_NAMES), dtype=np.float64)
    for ev in events:
        # Overlap with the half-open window
        ov = min(ev.end_s, win_end) - max(ev.start_s, window_start_s)
        if ov > 0.0:
            bins[ev.label] += ov
    if bins.sum() == 0.0:
        return TUEV_BCKG_INDEX
    return int(bins.argmax())


# ---------------------------------------------------------------------------
# .edf loader
# ---------------------------------------------------------------------------


# Pattern for "EEG channels" we keep from the raw EDF. TUH EDFs include
# extras like ECG, EKG, IBI, BURST, EMG, PHOTIC, PULSE, RESP, SUUR, TRIG
# that are not scalp EEG and shouldn't go through the iid expansion.
_EEG_CH_RE = re.compile(r"^EEG\s+([A-Za-z0-9]+)-(REF|LE)$", re.IGNORECASE)


def is_eeg_channel(ch_name: str) -> bool:
    """True if ``ch_name`` matches the standard ``EEG <site>-<REF|LE>`` form."""
    return bool(_EEG_CH_RE.match(ch_name))


def load_recording(edf_path: Path, *, eeg_only: bool = True):
    """Read a TUH EDF via MNE → ``(eeg, sr, channel_names)``.

    Returns:
        eeg (float32):  shape (n_channels, n_samples), in **volts**
                        (MNE's native unit; the per-channel z-score
                        downstream makes this scale-invariant).
        sr (float):     sampling rate in Hz.
        channel_names:  list[str], length n_channels.

    By default keeps only channels matching ``EEG <site>-<REF|LE>`` (drops
    ECG, EMG, PHOTIC, etc.). Pass ``eeg_only=False`` to keep all channels
    as-is.

    Raises FileNotFoundError if the .edf is missing.
    """
    import mne

    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"missing .edf file: {edf_path}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    if eeg_only:
        keep = [c for c in raw.ch_names if is_eeg_channel(c)]
        if not keep:
            raise ValueError(f"no EEG channels found in {edf_path.name}; "
                             f"channels were: {raw.ch_names}")
        raw = raw.pick(keep)

    eeg = raw.get_data().astype("float32")
    sr = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    return eeg, sr, channel_names


# ---------------------------------------------------------------------------
# rsync driver — pulls the data from NEDC SFTP into local NVMe
# ---------------------------------------------------------------------------


def rsync_remote_path(corpus: TUH_CORPUS_T, version: str | None = None) -> str:
    """Build the ``nedc-tuh:...`` rsync source URL for a corpus version."""
    version = version or DEFAULT_VERSION_BY_CORPUS[corpus]
    sub = REMOTE_CORPUS_SUBDIR[corpus]
    return f"{NEDC_SSH_HOST_ALIAS}:data/{sub}/{version}/"


def rsync_corpus(corpus: TUH_CORPUS_T, local_root: Path,
                 *, version: str | None = None,
                 dry_run: bool = False,
                 extra_rsync_args: Iterable[str] = (),
                 ) -> int:
    """Run ``rsync`` to mirror a NEDC corpus version into ``local_root``.

    Defaults to a streaming ``rsync -avL`` (preserve attrs, follow
    symlinks, verbose). Returns the rsync process's exit code.

    ``local_root`` is the per-corpus root, e.g. ``$EEG_DATA_ROOT/raw/tuab/``.
    The contents under it after a successful sync mirror the remote
    ``data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/`` (i.e. the ``edf/`` and
    ``AAREADME.txt`` siblings live directly under ``local_root``).
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus: {corpus!r}; valid: {CORPORA}")
    local_root = Path(local_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    src = rsync_remote_path(corpus, version)
    dst = str(local_root) + "/"
    cmd = ["rsync", "-avL", "--info=progress2", src, dst, *extra_rsync_args]
    if dry_run:
        cmd.insert(1, "-n")
    print("[tuh] " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Iteration helpers (used by the CLI)
# ---------------------------------------------------------------------------


def iter_recordings(corpus_root: Path, corpus: TUH_CORPUS_T,
                    *, splits: tuple[str, ...] = ("train", "eval"),
                    version: str | None = None,
                    max_per_split: int | None = None,
                    ) -> list[Recording]:
    """Walk a corpus root and return recordings, optionally capped per split."""
    recs = walk_corpus(corpus_root, corpus, version=version)
    if splits != ("train", "eval"):
        recs = [r for r in recs if r.split in splits]
    if max_per_split is None:
        return recs
    capped: list[Recording] = []
    counts: dict[str, int] = {}
    for r in recs:
        n = counts.get(r.split, 0)
        if n >= max_per_split:
            continue
        counts[r.split] = n + 1
        capped.append(r)
    return capped
