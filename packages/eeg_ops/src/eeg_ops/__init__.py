"""eeg-ops — cluster-lifecycle CLI for the eegModel project.

Public re-exports are intentionally minimal: callers should reach into the
typed submodules (:mod:`eeg_ops.config`, :mod:`eeg_ops.aws`, etc.) when used
as a library, not import from the top-level package. The CLI in
:mod:`eeg_ops.cli` is the supported user surface.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
