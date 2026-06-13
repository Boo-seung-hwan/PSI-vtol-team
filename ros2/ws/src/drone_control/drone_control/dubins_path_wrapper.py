from typing import List
import copy
import math
import numpy as np

from .datatypes import PathPoint, SegmentType, Waypoint, wrap_pi
from .DubinsAirplaneFunctions_cleaned import (
    DubinsAirplanePath,
    ExtractDubinsAirplanePath,
    MinTurnRadius_DubinsAirplane,
)


def _compute_yaws(path_xyz: np.ndarray) -> np.ndarray:
    yaws = []
    for i in range(len(path_xyz)):
        if i < len(path_xyz) - 1:
            delta = path_xyz[i + 1, :2] - path_xyz[i, :2]
        else:
            delta = path_xyz[i, :2] - path_xyz[i - 1, :2]
        yaws.append(wrap_pi(math.atan2(delta[1], delta[0])))
    return np.asarray(yaws, dtype=float)


def _downsample_by_distance(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points) <= 2 or spacing_m <= 0:
        return points
    keep = [0]
    last = points[0]
    for i in range(1, len(points) - 1):
        if np.linalg.norm(points[i] - last) >= spacing_m:
            keep.append(i)
            last = points[i]
    keep.append(len(points) - 1)
    return points[keep]


def generate_fixedwing_dubins_path(
    start: Waypoint,
    goal: Waypoint,
    event_id: int,
    cruise_speed_mps: float,
    max_bank_deg: float,
    max_flight_path_angle_deg: float,
    output_spacing_m: float = 1.0,
) -> List[PathPoint]:
    """Generate a fixed-wing Dubins-Airplane path between two waypoints.

    Output PathPoint coordinates are PX4 local NED, ready to be fed into
    TrajectorySetpoint.position = [x, y, z].
    """
    init_conf = start.pose4.copy()
    final_conf = goal.pose4.copy()
    r_min = MinTurnRadius_DubinsAirplane(
        cruise_speed_mps,
        math.radians(max_bank_deg),
    )
    gamma_max = math.radians(max_flight_path_angle_deg)

    # The legacy Dubins file uses a global dictionary internally. Deep-copying
    # the result prevents later calls from mutating an already generated route.
    solution = copy.deepcopy(DubinsAirplanePath(init_conf, final_conf, r_min, gamma_max))
    path_3xn = ExtractDubinsAirplanePath(solution)
    xyz = np.asarray(path_3xn.T, dtype=float)
    xyz = _downsample_by_distance(xyz, output_spacing_m)
    yaws = _compute_yaws(xyz)

    return [
        PathPoint(
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]),
            yaw=float(yaws[i]),
            event_id=event_id,
            target_speed_mps=cruise_speed_mps,
            segment_type=SegmentType.FIXEDWING_DUBINS,
            acceptance_radius_m=goal.acceptance_radius_m,
        )
        for i, p in enumerate(xyz)
    ]
