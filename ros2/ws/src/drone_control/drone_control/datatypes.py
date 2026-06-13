from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import math
import numpy as np


class SegmentType(Enum):
    MULTICOPTER = auto()        #회전익 모드 이동/호버링
    TRANSITION_TO_FW = auto()       #회전익 → 고정익 천이
    FIXEDWING_DUBINS = auto()       #고정익 Dubins 경로
    TRANSITION_TO_MC = auto()       #고정익 → 회전익 천이
    RESCUE = auto()                 #REP 조난자 구조
    LAND = auto()                   #착륙 구간


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    acceptance_radius_m: float = 4.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_msl_m: Optional[float] = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def pose4(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw], dtype=float)


@dataclass
class PathPoint:
    x: float
    y: float
    z: float
    yaw: float
    event_id: int
    target_speed_mps: float
    segment_type: SegmentType
    acceptance_radius_m: float = 4.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
