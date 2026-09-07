"""Contiguous SI-segment extraction.

A "segment" is a maximal run of samples that are ALL valid on the loop's own
timebase. Disconnected valid intervals are never concatenated as if
continuous. Short runs are discarded (they cannot support a transient fit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# A run shorter than this cannot show a meaningful command->response transient.
MIN_SEGMENT_S = 1.5
# Runs may be split if an internal sample gap exceeds this (clock jump / logger
# stall); default = 3x the loop period is applied by the caller.
MAX_INTERNAL_GAP_FACTOR = 3.0


@dataclass
class Segment:
    source_log: str
    loop: str
    index: int
    start_idx: int
    end_idx: int              # inclusive
    start_s: float
    end_s: float
    duration_s: float
    n: int
    valid_fraction: float     # by construction 1.0; kept for interface symmetry
    rejection_summary: Dict[str, float] = field(default_factory=dict)  # reasons in the PRECEDING gap

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_log": self.source_log,
            "loop": self.loop,
            "index": self.index,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration_s": round(self.duration_s, 4),
            "n": self.n,
            "valid_fraction": round(self.valid_fraction, 4),
            "preceding_gap_rejections": {k: round(v, 4) for k, v in self.rejection_summary.items()},
        }


def _runs(mask: np.ndarray):
    """Yield (start, end_inclusive) index pairs for maximal True runs."""
    if mask.size == 0:
        return
    m = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    for s, e in zip(starts, ends):
        yield int(s), int(e)


def split_segments(
    source_log: str,
    loop: str,
    t_s: np.ndarray,
    valid_mask: np.ndarray,
    first_reason: np.ndarray,
    nominal_dt_s: float,
    min_duration_s: float = MIN_SEGMENT_S,
) -> List[Segment]:
    """Return the list of usable :class:`Segment` objects."""
    t_s = np.asarray(t_s, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    max_gap = MAX_INTERNAL_GAP_FACTOR * nominal_dt_s if nominal_dt_s > 0 else np.inf

    segs: List[Segment] = []
    idx = 0
    for s, e in _runs(valid_mask):
        # split the run further at internal time gaps
        sub_start = s
        for k in range(s + 1, e + 1):
            if t_s[k] - t_s[k - 1] > max_gap:
                _maybe_add(segs, source_log, loop, t_s, first_reason,
                           sub_start, k - 1, min_duration_s, len(segs))
                sub_start = k
        _maybe_add(segs, source_log, loop, t_s, first_reason,
                   sub_start, e, min_duration_s, len(segs))
    for i, seg in enumerate(segs):
        seg.index = i
    return segs


def _maybe_add(segs, source_log, loop, t_s, first_reason, s, e, min_dur, idx):
    if e < s:
        return
    dur = float(t_s[e] - t_s[s])
    if dur < min_dur:
        return
    # rejection reasons in the gap immediately before this segment
    gap_lo = 0 if not segs else segs[-1].end_idx + 1
    gap_reasons: Dict[str, float] = {}
    if s > gap_lo:
        window = first_reason[gap_lo:s]
        window = window[window != ""]
        if window.size:
            uniq, cnt = np.unique(window.astype(str), return_counts=True)
            tot = float(cnt.sum())
            gap_reasons = {u: float(c) / tot for u, c in zip(uniq, cnt)}
    segs.append(Segment(
        source_log=source_log,
        loop=loop,
        index=idx,
        start_idx=int(s),
        end_idx=int(e),
        start_s=float(t_s[s]),
        end_s=float(t_s[e]),
        duration_s=dur,
        n=int(e - s + 1),
        valid_fraction=1.0,
        rejection_summary=gap_reasons,
    ))


def segments_table(segments: List[Segment]) -> Dict[str, object]:
    if not segments:
        return {"n_segments": 0, "total_valid_s": 0.0, "longest_valid_s": 0.0, "segments": []}
    durs = [s.duration_s for s in segments]
    return {
        "n_segments": len(segments),
        "total_valid_s": round(float(sum(durs)), 3),
        "longest_valid_s": round(float(max(durs)), 3),
        "segments": [s.to_dict() for s in segments],
    }
