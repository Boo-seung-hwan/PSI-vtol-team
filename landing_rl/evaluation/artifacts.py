"""Locations and identity of the PPO baseline artifacts (RL Baseline v0.1).

The trained model and its VecNormalize statistics are NOT stored in git. This
module is the single place that says which files form the pair of record and
where they are looked for, so that no script or test depends on the shell's
current working directory or on a machine-specific absolute path.

Lookup order for the artifact directory:

    1. the ``UGRP_RL_ARTIFACT_DIR`` environment variable, if set
    2. ``<repo root>/mujoco_rl/runs``  (git-ignored; the historical location)

Explicit ``--model`` / ``--vecnorm`` arguments on the evaluation scripts always
win over both. Outputs are never written to the artifact directory by default
(see ``default_output_dir``), so pointing ``UGRP_RL_ARTIFACT_DIR`` at a
directory of archived results cannot overwrite them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR_ENV = "UGRP_RL_ARTIFACT_DIR"
REQUIRE_ARTIFACTS_ENV = "UGRP_RL_REQUIRE_ARTIFACTS"

# Pair of record: the model and the VecNormalize statistics it was trained
# with. They are only meaningful together.
BASELINE_MODEL_NAME = "ppo_landing_residual_v4_stage2_contact_final.zip"
BASELINE_VECNORM_NAME = "vecnormalize_v4_stage2_contact.pkl"


def default_artifact_dir() -> Path:
    override = os.environ.get(ARTIFACT_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "mujoco_rl" / "runs"


def default_model_path() -> Path:
    return default_artifact_dir() / BASELINE_MODEL_NAME


def default_vecnorm_path() -> Path:
    return default_artifact_dir() / BASELINE_VECNORM_NAME


def default_output_dir() -> Path:
    """Where evaluation scripts write result files by default. Deliberately
    independent of ``UGRP_RL_ARTIFACT_DIR`` (inputs and outputs never share a
    default location)."""
    return REPO_ROOT / "mujoco_rl" / "runs" / "paper_eval"


def artifacts_required() -> bool:
    """True when missing artifacts must fail loudly instead of skipping tests."""
    return os.environ.get(REQUIRE_ARTIFACTS_ENV, "").strip().lower() in {"1", "true", "yes"}


def missing_artifact_message(
    model: Optional[Path] = None, vecnorm: Optional[Path] = None
) -> Optional[str]:
    """Human-readable explanation if the PPO pair is not usable, else None."""
    model = Path(model) if model is not None else default_model_path()
    vecnorm = Path(vecnorm) if vecnorm is not None else default_vecnorm_path()

    missing = []
    if not model.is_file():
        missing.append(f"model     : {model}")
    if not vecnorm.is_file():
        missing.append(f"vecnorm   : {vecnorm}")
    if not missing:
        return None

    return (
        "baseline PPO artifact not available:\n  "
        + "\n  ".join(missing)
        + f"\nProvide them with --model/--vecnorm, or set {ARTIFACT_DIR_ENV}=<dir containing "
        f"{BASELINE_MODEL_NAME} and {BASELINE_VECNORM_NAME}>. See MODEL_ARTIFACTS.md. "
        "PID-only evaluation (--policy pid) does not need them."
    )
