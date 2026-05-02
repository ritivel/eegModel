"""HBN-EEG ingestion: catalog the public AWS bucket, download recordings,
read EEGLAB ``.set/.fdt`` via MNE, parse BIDS-iEEG sidecar metadata, and
derive the labels we need for the §4.3 eval suite (HBN ADHD-binary AUROC,
HBN 6-task classification BAC + WF1).

Source: ``s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/`` —
NIH-funded public mirror, no credentials needed for read; we use anonymous
boto3 (UNSIGNED) to avoid charging the user's AWS account for the
requester-pays-style listing.

Per-release layout (verified against the 2026-05 bucket state):

    s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/
        participants.tsv                                ← per-release
        dataset_description.json
        code/
        sub-NDARABCD/
            eeg/
                sub-NDARABCD_task-RestingState_channels.tsv
                sub-NDARABCD_task-RestingState_coordsystem.json
                sub-NDARABCD_task-RestingState_eeg.set     ← raw EEG (EEGLAB header)
                sub-NDARABCD_task-RestingState_eeg.fdt     ← raw EEG (EEGLAB binary)
                sub-NDARABCD_task-RestingState_eeg.json    ← BIDS sidecar
                sub-NDARABCD_task-RestingState_electrodes.tsv
                sub-NDARABCD_task-RestingState_events.tsv
                sub-NDARABCD_task-seqLearning8target_eeg.set
                sub-NDARABCD_task-symbolSearch_eeg.set
                sub-NDARABCD_task-surroundSupp_run-1_eeg.set
                sub-NDARABCD_task-surroundSupp_run-2_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-1_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-2_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-3_eeg.set
                sub-NDARABCD_task-DespicableMe_eeg.set        ← Video category
                sub-NDARABCD_task-DiaryOfAWimpyKid_eeg.set    ← Video category
                sub-NDARABCD_task-FunwithFractals_eeg.set     ← Video category
                sub-NDARABCD_task-ThePresent_eeg.set          ← Video category

10 releases known: NC, R1..R9. Total ~2,639 subjects across all releases
(NC=447, R1=136, R2=150, R3=184, R4=324, R5=330, R6=135, R7=381,
R8=257, R9=295) per a 2026-05-02 listing.

References:
    HBN-EEG dataset:  Shirazi et al. 2024 bioRxiv,
        https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2
    AWS public mirror layout (``s3://fcp-indi/.../HBN/BIDS_EEG/...``):
        https://fcp-indi.s3.amazonaws.com/index.html
    NeurIPS 2025 EEG Foundation Challenge (uses HBN):
        https://eeg2025.github.io/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HBN's six cognitive tasks → label index per `mini_experiments.md` §4.3
# Protocol A.2 (HBN 6-task classification). Task names as they appear in BIDS
# filenames are case-sensitive and have HBN-specific quirks: seqLearning8target
# has a digit inside the name; surroundSupp + contrastChangeDetection ship
# with multiple `_run-<N>` runs we collapse to one label; the "Video" category
# splits across four distinct movie-clip task names.
TASK_LABEL: dict[str, int] = {
    "RestingState":             0,
    "seqLearning8target":       1,
    "symbolSearch":             2,
    "surroundSupp":             3,
    "contrastChangeDetection":  4,
    # Video category — collapse the four movie-clip task names → label 5
    "DespicableMe":             5,
    "DiaryOfAWimpyKid":         5,
    "FunwithFractals":          5,
    "ThePresent":               5,
}

# Reverse map: label → canonical task name (we pick a representative for
# label 5 for display purposes).
TASK_NAME_BY_LABEL: dict[int, str] = {
    0: "RestingState",
    1: "seqLearning8target",
    2: "symbolSearch",
    3: "surroundSupp",
    4: "contrastChangeDetection",
    5: "Video",
}

# Filename-parse regex. BIDS filenames look like:
#     sub-NDARxxx_task-<task>[_run-<N>]_<datatype>.<ext>
# datatype ∈ {eeg, channels, electrodes, coordsystem, events}.
# We capture the task name (alphanumeric, may contain digits) up to the
# optional ``_run-<N>`` and the trailing ``_<datatype>``.
_TASK_FILENAME_RE = re.compile(
    r"task-([A-Za-z][A-Za-z0-9]*?)(?:_run-\d+)?_(?:eeg|channels|electrodes|coordsystem|events)"
)

# HBN site code: provided in participants.tsv via the "ehq_total_completed"-
# adjacent "release_number" / "site" columns (release-dependent). We resolve
# site lazily from the participants.tsv frame at preprocess time.
HBN_SITES: tuple[str, ...] = ("RU", "CBIC", "CUNY", "SI")

# 128-channel EGI HydroCel naming convention; HBN's channels.tsv uses these
# literal names. Order matters — channel_idx in our parquet schema is the
# position in this list.
HYDROCEL_128_NAMES: tuple[str, ...] = tuple(f"E{i}" for i in range(1, 129))

# FCP-INDI HBN-EEG bucket + prefix (BIDS-formatted release tree).
HBN_S3_BUCKET = "fcp-indi"
HBN_S3_PREFIX = "data/Projects/HBN/BIDS_EEG"

# HBN release codes (the prefix we type when calling the CLI). The actual S3
# directory name is ``cmi_bids_<release>`` — see ``release_s3_prefix``.
KNOWN_RELEASES: tuple[str, ...] = (
    "NC", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
)


def release_s3_prefix(release: str) -> str:
    """Map user-facing release code (e.g. 'R1') → S3 prefix segment ('cmi_bids_R1')."""
    return f"cmi_bids_{release}"


def parse_task_from_filename(filename: str) -> tuple[str, int] | None:
    """Parse a BIDS-iEEG filename → (canonical_task_name, task_label) or None.

    Examples:
        ``sub-NDARxxx_task-RestingState_eeg.set``         → ("RestingState", 0)
        ``sub-NDARxxx_task-seqLearning8target_eeg.set``   → ("seqLearning8target", 1)
        ``sub-NDARxxx_task-surroundSupp_run-1_eeg.set``   → ("surroundSupp", 3)
        ``sub-NDARxxx_task-DespicableMe_eeg.set``         → ("DespicableMe", 5)
        ``sub-NDARxxx_task-Unknown_eeg.set``              → None  (not in TASK_LABEL)
    """
    m = _TASK_FILENAME_RE.search(filename)
    if not m:
        return None
    task = m.group(1)
    label = TASK_LABEL.get(task)
    if label is None:
        return None
    return task, label


# ---------------------------------------------------------------------------
# Subject + recording dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """One ``.set/.fdt`` recording for one subject.

    Identifiers in S3 + on local disk (note: the local layout mirrors the
    BIDS S3 layout including the ``cmi_bids_<release>/`` prefix, so a flat
    ``rclone`` between local and S3 is a no-op rename):

        s3_set:  s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/sub-<id>/eeg/<basename>.set
        s3_fdt:  ...<basename>.fdt
        local:   <storage.raw_hbn>/cmi_bids_<release>/sub-<id>/eeg/<basename>.set
    """

    subject_id: str           # "NDARABCD1234"
    release: str              # "NC", "R1", ...
    task: str                 # canonical task name from TASK_LABEL keys
    task_label: int           # 0..5 (label index for §4.3 Protocol A.2)
    basename: str             # e.g. "sub-NDARABCD1234_task-RestingState_eeg"
    s3_set_key: str           # full S3 key of the .set
    s3_fdt_key: str           # full S3 key of the .fdt

    @property
    def s3_set_uri(self) -> str:
        return f"s3://{HBN_S3_BUCKET}/{self.s3_set_key}"

    @property
    def s3_fdt_uri(self) -> str:
        return f"s3://{HBN_S3_BUCKET}/{self.s3_fdt_key}"

    def _local_eeg_dir(self, raw_root: Path) -> Path:
        return raw_root / release_s3_prefix(self.release) / f"sub-{self.subject_id}" / "eeg"

    def local_set_path(self, raw_root: Path) -> Path:
        return self._local_eeg_dir(raw_root) / f"{self.basename}.set"

    def local_fdt_path(self, raw_root: Path) -> Path:
        return self._local_eeg_dir(raw_root) / f"{self.basename}.fdt"


# ---------------------------------------------------------------------------
# S3 anonymous client + listing
# ---------------------------------------------------------------------------


def _anonymous_s3_client():
    """boto3 client that doesn't sign requests (FCP-INDI bucket is public)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_releases() -> list[str]:
    """List all HBN-EEG release codes available under ``BIDS_EEG/cmi_bids_*``.

    Returns user-facing release codes (e.g. ``["NC", "R1", "R2", ...]``),
    stripping the ``cmi_bids_`` prefix the bucket uses for the directory name.
    """
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    releases: set[str] = set()
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=f"{HBN_S3_PREFIX}/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            tail = cp["Prefix"].rstrip("/").split("/")[-1]
            if tail.startswith("cmi_bids_"):
                releases.add(tail[len("cmi_bids_"):])
    return sorted(releases)


def list_subjects(release: str, *, max_subjects: int | None = None) -> list[str]:
    """List subject IDs (without ``sub-`` prefix) under one release."""
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/"
    subjects: list[str] = []
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            tail = cp["Prefix"].rstrip("/").split("/")[-1]
            if tail.startswith("sub-"):
                subjects.append(tail[len("sub-"):])
                if max_subjects is not None and len(subjects) >= max_subjects:
                    return subjects
    return subjects


def list_subject_recordings(subject_id: str, release: str) -> list[Recording]:
    """List all recordings for one subject in one release."""
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/sub-{subject_id}/eeg/"

    recordings: list[Recording] = []
    seen_basenames: set[str] = set()
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".set"):
                continue
            basename = Path(key).stem
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)

            parsed = parse_task_from_filename(basename)
            if parsed is None:
                continue
            task, task_label = parsed

            fdt_key = key[:-4] + ".fdt"  # parallel filename; .set / .fdt come as a pair
            recordings.append(Recording(
                subject_id=subject_id,
                release=release,
                task=task,
                task_label=task_label,
                basename=basename,
                s3_set_key=key,
                s3_fdt_key=fdt_key,
            ))
    return recordings


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_recording(rec: Recording, raw_root: Path,
                       *, overwrite: bool = False) -> tuple[Path, Path]:
    """Download a single ``.set + .fdt`` pair to local NVMe.

    Returns (local_set_path, local_fdt_path). Skips files that already exist
    unless ``overwrite=True``.
    """
    s3 = _anonymous_s3_client()
    set_path = rec.local_set_path(raw_root)
    fdt_path = rec.local_fdt_path(raw_root)
    set_path.parent.mkdir(parents=True, exist_ok=True)

    for s3_key, local_path in [
        (rec.s3_set_key, set_path),
        (rec.s3_fdt_key, fdt_path),
    ]:
        if local_path.exists() and not overwrite:
            continue
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        s3.download_file(HBN_S3_BUCKET, s3_key, str(tmp_path))
        tmp_path.replace(local_path)

    return set_path, fdt_path


def download_subject_sidecars(subject_id: str, release: str, raw_root: Path,
                              *, overwrite: bool = False) -> list[Path]:
    """Pull the BIDS sidecar files (channels.tsv, eeg.json, electrodes.tsv,
    coordsystem.json, events.tsv).

    These are small (~kB each) and needed for channel-name parsing + sample
    rate verification. We pull them all per-subject in one shot; the .set/.fdt
    pairs are pulled separately per-recording.
    """
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/sub-{subject_id}/eeg/"
    out: list[Path] = []
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith((".set", ".fdt")):
                continue
            local = raw_root / release_s3_prefix(release) / f"sub-{subject_id}" / "eeg" / Path(key).name
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and not overwrite:
                out.append(local)
                continue
            s3.download_file(HBN_S3_BUCKET, key, str(local))
            out.append(local)
    return out


def download_release_metadata(release: str, raw_root: Path,
                              *, overwrite: bool = False) -> dict[str, Path]:
    """Pull release-level files: participants.tsv + dataset_description.json."""
    s3 = _anonymous_s3_client()
    out: dict[str, Path] = {}
    for fname in ("participants.tsv", "dataset_description.json"):
        key = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/{fname}"
        local = raw_root / release_s3_prefix(release) / fname
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and not overwrite:
            out[fname] = local
            continue
        try:
            s3.download_file(HBN_S3_BUCKET, key, str(local))
            out[fname] = local
        except Exception as e:  # noqa: BLE001 — best-effort; release may lack one of these
            out.setdefault(fname, None)
            print(f"[hbn] release {release}: missing {fname} ({e})")
    return out


# ---------------------------------------------------------------------------
# .set/.fdt loader (MNE)
# ---------------------------------------------------------------------------


def load_recording(set_path: Path):
    """Read a ``.set/.fdt`` pair via MNE → return (eeg, sr, channel_names).

    Returns:
        eeg: float32 ndarray of shape (n_channels, n_samples), in **volts**
             (MNE's native unit; we keep volts here and let the per-channel
             z-score normalise downstream).
        sr:  float, sampling rate in Hz.
        channel_names: list[str], length n_channels.

    Raises FileNotFoundError if the .set or its sibling .fdt is missing.
    """
    import mne

    set_path = Path(set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"missing .set file: {set_path}")
    fdt_path = set_path.with_suffix(".fdt")
    if not fdt_path.exists():
        # MNE will fail with a less helpful error if the .fdt is missing.
        raise FileNotFoundError(f"missing .fdt sidecar: {fdt_path}")

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    eeg = raw.get_data().astype("float32")  # (n_channels, n_samples), volts
    sr = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    return eeg, sr, channel_names


# ---------------------------------------------------------------------------
# participants.tsv parsing — derive HBN ADHD-binary label per §4.3 A.1
# ---------------------------------------------------------------------------


# Match any DSM-V diagnosis column that mentions ADHD / Attention-Deficit.
# HBN uses several free-text labels: "ADHD-Combined Type", "ADHD-Inattentive
# Type", "ADHD, Combined Presentation", etc. The regex is intentionally loose.
ADHD_DX_PATTERN = re.compile(r"\b(?:ADHD|attention[\s\-_]*deficit)\b", re.IGNORECASE)


def load_participants(participants_tsv: Path):
    """Read participants.tsv into a pandas DataFrame.

    Adds derived columns:
        adhd:  1 if any Dx_* column matches ADHD_DX_PATTERN,
               0 if none of them match and ≥1 is non-empty,
              -1 (missing) if all Dx_* columns are empty/NaN.
        site:  RU / CBIC / CUNY / SI if present in the row, else "" .
        age:   float; NaN if missing.
        sex:   "M" / "F" / "" .

    The column names HBN uses vary slightly across releases ("p_factor",
    "EHQ_Total", etc.) — we keep all of them and only standardise the
    columns we need for the §4.3 eval suite.
    """
    import pandas as pd

    df = pd.read_csv(participants_tsv, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    # Normalise the participant_id column → bare subject ID without "sub-"
    if "participant_id" in df.columns:
        df["subject_id"] = df["participant_id"].str.replace(r"^sub-", "", regex=True)
    elif "subject_id" not in df.columns:
        raise KeyError(f"participants.tsv missing participant_id / subject_id: cols={list(df.columns)}")

    # ADHD label
    dx_cols = [c for c in df.columns if c.lower().startswith(("dx_", "diagnosis_"))]
    if dx_cols:
        adhd = []
        for _, row in df[dx_cols].iterrows():
            non_empty = [v for v in row.values if isinstance(v, str) and v.strip()]
            if not non_empty:
                adhd.append(-1)
            elif any(ADHD_DX_PATTERN.search(v) for v in non_empty):
                adhd.append(1)
            else:
                adhd.append(0)
        df["adhd"] = adhd
    else:
        df["adhd"] = -1

    # Site (best-effort: HBN uses "site" or "Site" or per-release tags)
    site_col = next((c for c in df.columns if c.lower() == "site"), None)
    df["site"] = df[site_col].fillna("") if site_col else ""

    # Age + sex
    if "age" in df.columns:
        df["age"] = df["age"].astype("float", errors="ignore")
    else:
        df["age"] = float("nan")
    if "sex" not in df.columns:
        df["sex"] = ""
    df["sex"] = df["sex"].fillna("").str.upper().replace({"MALE": "M", "FEMALE": "F"})

    return df


def metadata_for_subject(subject_id: str, participants_df) -> dict:
    """Look up demographics + ADHD label for one subject.

    Returns a dict with keys (adhd, site, age, sex) compatible with the
    parquet base_metadata field. Missing values are filled with sentinels:
        adhd = -1, site = "", age = NaN, sex = "".
    """
    rows = participants_df[participants_df["subject_id"] == subject_id]
    if rows.empty:
        return {"adhd": -1, "site": "", "age": float("nan"), "sex": ""}
    r = rows.iloc[0]
    return {
        "adhd": int(r.get("adhd", -1)),
        "site": str(r.get("site", "")),
        "age": float(r.get("age", float("nan"))) if r.get("age", "") not in ("", None) else float("nan"),
        "sex": str(r.get("sex", "")),
    }


# ---------------------------------------------------------------------------
# Iteration helpers (used by the CLI)
# ---------------------------------------------------------------------------


def iter_releases_and_subjects(releases: Iterable[str] | None = None,
                               max_subjects_per_release: int | None = None,
                               ) -> list[tuple[str, str]]:
    """Cross-product (release, subject_id) iteration helper.

    If ``releases`` is None, lists all releases. ``max_subjects_per_release``
    caps each release independently — useful for Tier-1 sanity processing.
    """
    rels = list(releases) if releases else list_releases()
    out: list[tuple[str, str]] = []
    for release in rels:
        subs = list_subjects(release, max_subjects=max_subjects_per_release)
        for sub in subs:
            out.append((release, sub))
    return out
