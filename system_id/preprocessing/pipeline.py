"""End-to-end preprocessing orchestrator.

    raw .ulg
      -> load_ulog
      -> build_all (velocity / attitude / rate / translation datasets)
      -> split_segments per loop
      -> build_provenance
      -> per_log_report (data-quality gate)
      -> save_outputs  (.npz per loop + JSON sidecars)

No system-identification fitting. Derived sample data is written under
``system_id/derived/sample_other_project/`` (a gitignored path); raw ULogs are
read-only and never moved or deleted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from system_id.preprocessing.dataset import LoopDataset, build_all
from system_id.preprocessing.masks import BLOCK_NAMES
from system_id.preprocessing.provenance import (
    DATASET_ROLE_VALIDATION,
    SOURCE_PROJECT_SAMPLE,
    build_provenance,
)
from system_id.preprocessing.report import BLOCK_LOOP, per_log_report, render_text
from system_id.preprocessing.schema import ALL_LOOPS, LOOP_NOMINAL_HZ
from system_id.preprocessing.segments import Segment, split_segments
from system_id.preprocessing.ulog_loader import load_ulog

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_OUT_ROOT = os.path.join(REPO_DIR, "system_id", "derived", "sample_other_project")
DEFAULT_SAMPLE_LOGS = tuple(
    os.path.join(REPO_DIR, "landing_rl", "flight_log_ulg", name)
    for name in ("03_49_26.ulg", "05_34_35.ulg", "06_56_24.ulg", "08_53_49.ulg")
)


@dataclass
class PreprocessResult:
    log_path: str
    stem: str
    out_dir: Optional[str]
    datasets: Dict[str, LoopDataset]
    block_segments: Dict[str, List[Segment]]
    provenance: Dict[str, object]
    report: Dict[str, object]


def _segments_for(loaded_path: str, datasets: Dict[str, LoopDataset]) -> Dict[str, List[Segment]]:
    """Contiguous valid segments per SI BLOCK (not per loop): each block uses
    its own block mask, so ground-effect / near-level / pairing rejections that
    apply to one block do not shrink another."""
    out: Dict[str, List[Segment]] = {}
    stem = os.path.basename(loaded_path)
    for block in BLOCK_NAMES:
        loop = BLOCK_LOOP[block]
        ds = datasets[loop]
        out[block] = split_segments(
            source_log=stem,
            loop=block,
            t_s=ds.columns["t"],
            valid_mask=ds.block_masks[block],
            first_reason=ds.mask.first_reason,
            nominal_dt_s=1.0 / LOOP_NOMINAL_HZ[loop],
        )
    return out


def preprocess_log(
    ulg_path: str,
    out_root: Optional[str] = DEFAULT_OUT_ROOT,
    source_project: str = SOURCE_PROJECT_SAMPLE,
    dataset_role: str = DATASET_ROLE_VALIDATION,
    write: bool = True,
) -> PreprocessResult:
    loaded = load_ulog(ulg_path)
    datasets = build_all(loaded)
    block_segments = _segments_for(ulg_path, datasets)
    prov = build_provenance(loaded, source_project, dataset_role, repo_dir=REPO_DIR).to_dict()
    report = per_log_report(loaded, datasets, block_segments, source_project, dataset_role)

    stem = os.path.splitext(os.path.basename(ulg_path))[0]
    out_dir = None
    if write and out_root:
        out_dir = os.path.join(out_root, stem)
        _save_outputs(out_dir, datasets, block_segments, prov, report)

    return PreprocessResult(
        log_path=ulg_path, stem=stem, out_dir=out_dir,
        datasets=datasets, block_segments=block_segments, provenance=prov, report=report,
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _save_outputs(out_dir, datasets, block_segments, prov, report):
    os.makedirs(out_dir, exist_ok=True)
    columns_meta = {}
    for loop, ds in datasets.items():
        npz_path = os.path.join(out_dir, f"{loop}.npz")
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in ds.columns.items()}
        arrays["_flight_base"] = ds.mask.flight_base.astype(np.int8)
        arrays["_ground_effect"] = ds.mask.ground_effect.astype(np.int8)
        for reason, m in ds.mask.reason_masks.items():
            arrays[f"_reject_{reason}"] = m.astype(np.int8)
        for block, m in ds.block_masks.items():
            arrays[f"_block_{block}"] = np.asarray(m, dtype=np.int8)
        for name, m in ds.aux.items():
            arrays[f"_aux_{name}"] = np.asarray(m, dtype=np.int8)
        np.savez_compressed(npz_path, **arrays)
        columns_meta[loop] = {
            "npz": os.path.basename(npz_path),
            "master_topic": ds.timebase.master_topic,
            "stamp_kind": ds.timebase.stamp_kind,
            "effective_hz": ds.timebase.effective_hz,
            "n": ds.n,
            "columns": ds.column_meta,
            "flight_base_array": "_flight_base",
            "ground_effect_array": "_ground_effect",
            "reject_arrays": [f"_reject_{r}" for r in ds.mask.reason_masks],
            "block_mask_arrays": {b: f"_block_{b}" for b in ds.block_masks},
            "aux_mask_arrays": {a: f"_aux_{a}" for a in ds.aux},
            "setpoint_pairing": ds.pairing or None,
        }
    with open(os.path.join(out_dir, "columns.json"), "w") as fh:
        json.dump(columns_meta, fh, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "segments.json"), "w") as fh:
        json.dump({k: [s.to_dict() for s in v] for k, v in block_segments.items()},
                  fh, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(render_text(report) + "\n")


def run_batch(
    logs=DEFAULT_SAMPLE_LOGS,
    out_root: Optional[str] = DEFAULT_OUT_ROOT,
    source_project: str = SOURCE_PROJECT_SAMPLE,
    dataset_role: str = DATASET_ROLE_VALIDATION,
    write: bool = True,
) -> List[PreprocessResult]:
    results = []
    for p in logs:
        if not os.path.exists(p):
            print(f"[skip] {p} (not found)")
            continue
        res = preprocess_log(p, out_root, source_project, dataset_role, write=write)
        print(render_text(res.report))
        if res.out_dir:
            print(f"  -> wrote {res.out_dir}")
        results.append(res)
    return results


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="SI ULog preprocessing (no fitting).")
    ap.add_argument("logs", nargs="*", help="ULog paths (default: the 4 sample logs)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--source-project", default=SOURCE_PROJECT_SAMPLE)
    ap.add_argument("--dataset-role", default=DATASET_ROLE_VALIDATION)
    ap.add_argument("--no-write", action="store_true")
    ns = ap.parse_args(argv)
    logs = ns.logs or DEFAULT_SAMPLE_LOGS
    run_batch(logs, None if ns.no_write else ns.out_root,
              ns.source_project, ns.dataset_role, write=not ns.no_write)


if __name__ == "__main__":  # pragma: no cover
    main()
