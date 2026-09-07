"""Provenance record for a preprocessed log.

The record makes it impossible to confuse OTHER-PROJECT sample data with future
UGRP data: every output carries an explicit ``source_project`` and
``dataset_role``.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from system_id.preprocessing.schema import (
    AIRFRAME_PARAM_KEYS,
    MC_PARAM_KEYS,
    MPC_PARAM_KEYS,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
)
from system_id.preprocessing.ulog_loader import LoadedULog

# Canonical labels. Use these for the four OTHER-PROJECT sample logs.
SOURCE_PROJECT_SAMPLE = "OTHER_PROJECT_SAMPLE"
SOURCE_PROJECT_UGRP = "UGRP"
DATASET_ROLE_VALIDATION = "PIPELINE_VALIDATION_ONLY"
DATASET_ROLE_SI = "SYSTEM_IDENTIFICATION"


def _git_state(repo_dir: Optional[str]) -> Dict[str, object]:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD") or None
        branch = run("rev-parse", "--abbrev-ref", "HEAD") or None
        status = run("status", "--porcelain")
        dirty = bool(status)
        changed = [ln[3:] for ln in status.splitlines()][:50]
        return {"commit": commit, "branch": branch, "dirty": dirty, "changed_files": changed}
    except Exception as exc:  # pragma: no cover - git optional
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def _tool_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("pyulog", "numpy", "scipy", "pandas"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "unavailable"
    return out


@dataclass
class Provenance:
    original_path: str
    sha256: str
    source_project: str
    dataset_role: str
    px4: Dict[str, str]
    hardware: str
    airframe: Dict[str, object]
    mpc_params: Dict[str, object]
    mc_params: Dict[str, object]
    tool_versions: Dict[str, str]
    git: Dict[str, object]
    processed_utc: str
    schema_version: str
    preprocessing_version: str
    log_window_s: float
    n_dropouts: int
    dropout_s: float
    loader_warnings: list = field(default_factory=list)
    missing_optional_topics: list = field(default_factory=list)
    multi_instance_topics: list = field(default_factory=list)
    notes: str = (
        "Controller parameters below belong to the LOGGED vehicle. For "
        "source_project=OTHER_PROJECT_SAMPLE they are NOT UGRP values and must "
        "never be copied into a UGRP config."
    )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _subset(params: Dict[str, float], keys) -> Dict[str, object]:
    return {k: params[k] for k in keys if k in params}


def build_provenance(
    loaded: LoadedULog,
    source_project: str,
    dataset_role: str,
    repo_dir: Optional[str] = None,
) -> Provenance:
    hw = loaded.px4.get("ver_hw", "UNKNOWN")
    if loaded.px4.get("ver_hw_subtype"):
        hw = f"{hw}/{loaded.px4['ver_hw_subtype']}"
    return Provenance(
        original_path=loaded.path,
        sha256=loaded.sha256,
        source_project=source_project,
        dataset_role=dataset_role,
        px4=dict(loaded.px4),
        hardware=hw,
        airframe=_subset(loaded.params, AIRFRAME_PARAM_KEYS),
        mpc_params=_subset(loaded.params, MPC_PARAM_KEYS),
        mc_params=_subset(loaded.params, MC_PARAM_KEYS),
        tool_versions=_tool_versions(),
        git=_git_state(repo_dir),
        processed_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        preprocessing_version=PREPROCESSING_VERSION,
        log_window_s=round((loaded.last_us - loaded.start_us) / 1e6, 3),
        n_dropouts=loaded.n_dropouts,
        dropout_s=round(loaded.dropout_us / 1e6, 4),
        loader_warnings=list(loaded.warnings),
        missing_optional_topics=list(loaded.missing_optional),
        multi_instance_topics=list(loaded.multi_instance_topics),
    )
