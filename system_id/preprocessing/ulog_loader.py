"""Thin, validated wrapper around :mod:`pyulog`.

Responsibilities:
  * compute the file SHA256 (provenance identity),
  * open the log and translate any parse failure into :class:`ULogLoadError`,
  * expose PX4 version / hardware / airframe / parameters,
  * expose one :class:`TopicData` per topic (multi-instance 0 by default),
  * assert that every REQUIRED topic is present; record which OPTIONAL topics
    are missing (never fatal, e.g. ``esc_status``).

No resampling, masking, or frame maths happens here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from system_id.preprocessing.schema import (
    ALL_TOPICS,
    OPTIONAL_TOPICS,
    REQUIRED_TOPICS,
)


class ULogLoadError(Exception):
    """Raised for a missing file, an unreadable/corrupt ULog, or a missing
    required topic."""


@dataclass
class TopicData:
    """One logged uORB topic instance."""

    name: str
    multi_id: int
    fields: Tuple[str, ...]
    data: Dict[str, np.ndarray]
    n: int
    has_timestamp_sample: bool

    def get(self, field_name: str) -> np.ndarray:
        if field_name not in self.data:
            raise KeyError(
                f"{self.name}[{self.multi_id}] has no field {field_name!r}; "
                f"available: {list(self.fields)}"
            )
        return self.data[field_name]

    def has(self, field_name: str) -> bool:
        return field_name in self.data

    def stamp_us(self, prefer_sample: bool) -> np.ndarray:
        """Return the chosen timestamp array (float64 microseconds)."""
        if prefer_sample and self.has_timestamp_sample:
            return np.asarray(self.data["timestamp_sample"], dtype=np.float64)
        return np.asarray(self.data["timestamp"], dtype=np.float64)


@dataclass
class LoadedULog:
    """Everything the downstream stages need from one ``.ulg`` file."""

    path: str
    sha256: str
    px4: Dict[str, str]                       # ver_sw, ver_sw_release, ver_sw_branch, ver_hw, ...
    params: Dict[str, float]
    topics: Dict[str, TopicData]             # name -> instance-0 TopicData
    start_us: float
    last_us: float
    dropout_us: float
    n_dropouts: int
    present_topics: Tuple[str, ...]
    missing_required: Tuple[str, ...]
    missing_optional: Tuple[str, ...]
    multi_instance_topics: Tuple[str, ...]   # topics where instances >0 existed and were dropped
    warnings: List[str] = field(default_factory=list)

    def topic(self, name: str) -> TopicData:
        if name not in self.topics:
            raise ULogLoadError(f"topic {name!r} not available in {self.path}")
        return self.topics[name]

    def has_topic(self, name: str) -> bool:
        return name in self.topics


def sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(_chunk), b""):
                h.update(block)
    except OSError as exc:  # pragma: no cover - trivial
        raise ULogLoadError(f"cannot read {path}: {exc}") from exc
    return h.hexdigest()


def _extract_px4_info(ulog) -> Dict[str, str]:
    info = getattr(ulog, "msg_info_dict", {}) or {}
    keys = (
        "ver_sw", "ver_sw_release", "ver_sw_branch",
        "ver_hw", "ver_hw_subtype", "sys_name", "sys_os_name",
    )
    out = {}
    for k in keys:
        v = info.get(k)
        if v is not None:
            out[k] = str(v)
    return out


def _topic_from_dataset(ds) -> TopicData:
    data = {name: np.asarray(arr) for name, arr in ds.data.items()}
    fields = tuple(sorted(data.keys()))
    n = len(data["timestamp"]) if "timestamp" in data else (
        len(next(iter(data.values()))) if data else 0
    )
    return TopicData(
        name=ds.name,
        multi_id=int(getattr(ds, "multi_id", 0)),
        fields=fields,
        data=data,
        n=int(n),
        has_timestamp_sample="timestamp_sample" in data,
    )


def load_ulog(
    path: str,
    required_topics: Sequence[str] = REQUIRED_TOPICS,
    optional_topics: Sequence[str] = OPTIONAL_TOPICS,
    wanted_topics: Optional[Sequence[str]] = None,
) -> LoadedULog:
    """Parse ``path`` and return a validated :class:`LoadedULog`.

    Raises :class:`ULogLoadError` for a missing/corrupt file or a missing
    required topic. Missing optional topics are recorded, never fatal.
    """
    try:
        from pyulog import ULog
    except Exception as exc:  # pragma: no cover
        raise ULogLoadError(f"pyulog import failed: {exc}") from exc

    digest = sha256_file(path)

    want = set(wanted_topics) if wanted_topics is not None else set(ALL_TOPICS)
    want |= set(required_topics)

    try:
        ulog = ULog(path, message_name_filter_list=sorted(want))
    except Exception as exc:
        raise ULogLoadError(f"failed to parse ULog {path}: {exc}") from exc

    topics: Dict[str, TopicData] = {}
    multi_seen: Dict[str, int] = {}
    for ds in ulog.data_list:
        mid = int(getattr(ds, "multi_id", 0))
        multi_seen[ds.name] = max(multi_seen.get(ds.name, 0), mid)
        if mid != 0:
            continue
        if ds.name in topics:  # pragma: no cover - pyulog yields one inst-0
            continue
        try:
            topics[ds.name] = _topic_from_dataset(ds)
        except Exception as exc:  # pragma: no cover
            raise ULogLoadError(
                f"failed to extract topic {ds.name} from {path}: {exc}"
            ) from exc

    missing_required = tuple(t for t in required_topics if t not in topics)
    if missing_required:
        raise ULogLoadError(
            f"{path}: required topic(s) missing: {', '.join(missing_required)}"
        )
    missing_optional = tuple(t for t in optional_topics if t not in topics)
    multi_instance = tuple(sorted(k for k, v in multi_seen.items() if v > 0))

    stamps = []
    for td in topics.values():
        if "timestamp" in td.data and td.n:
            ts = np.asarray(td.data["timestamp"], dtype=np.float64)
            stamps.append((ts[0], ts[-1]))
    start_us = float(min(s for s, _ in stamps)) if stamps else 0.0
    last_us = float(max(e for _, e in stamps)) if stamps else 0.0

    dropout_us = float(sum(d.duration for d in getattr(ulog, "dropouts", []) or []) * 1000.0)
    n_dropouts = len(getattr(ulog, "dropouts", []) or [])

    warnings: List[str] = []
    if multi_instance:
        warnings.append(
            "multi-instance topics reduced to instance 0: " + ", ".join(multi_instance)
        )
    if missing_optional:
        warnings.append("optional topics absent: " + ", ".join(missing_optional))

    return LoadedULog(
        path=path,
        sha256=digest,
        px4=_extract_px4_info(ulog),
        params=dict(getattr(ulog, "initial_parameters", {}) or {}),
        topics=topics,
        start_us=start_us,
        last_us=last_us,
        dropout_us=dropout_us,
        n_dropouts=n_dropouts,
        present_topics=tuple(sorted(topics.keys())),
        missing_required=(),
        missing_optional=missing_optional,
        multi_instance_topics=multi_instance,
        warnings=warnings,
    )
