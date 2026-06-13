import math
from typing import Iterable, List

import numpy as np

from drone_control.datatypes import Waypoint, PathPoint
from drone_control.dubins_path_wrapper import generate_fixedwing_dubins_path
from drone_control.datatypes import SegmentType, PathPoint

def _yaw_from_to(start: np.ndarray, goal: np.ndarray) -> float:
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    delta = goal[:2] - start[:2]

    if np.linalg.norm(delta) < 1e-6:
        return 0.0

    return math.atan2(delta[1], delta[0])


def _make_waypoint(
    name: str,
    position: np.ndarray,
    yaw: float,
    acceptance_radius_m: float = 4.0,
) -> Waypoint:
    position = np.asarray(position, dtype=float).reshape(3)

    return Waypoint(
        name=name,
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        yaw=float(yaw),
        acceptance_radius_m=float(acceptance_radius_m),
    )


def pathpoints_to_ndarray(points: Iterable[PathPoint]) -> np.ndarray:
    points = list(points)

    if not points:
        raise ValueError("PathPoint list is empty.")

    return np.array(
        [[p.x, p.y, p.z] for p in points],
        dtype=float,
    )


def _append_path_without_duplicate(
    total: List[PathPoint],
    segment: List[PathPoint],
) -> None:
    if not segment:
        return

    if not total:
        total.extend(segment)
        return

    last = np.array([total[-1].x, total[-1].y, total[-1].z], dtype=float)
    first = np.array([segment[0].x, segment[0].y, segment[0].z], dtype=float)

    if np.linalg.norm(last - first) < 1e-6:
        total.extend(segment[1:])
    else:
        total.extend(segment)


def generate_fw_generated_path(
    wp2: np.ndarray,
    wp3: np.ndarray,
    wp4: np.ndarray,
    wp5: np.ndarray,
    cruise_speed_mps: float = 18.0,
    max_bank_deg: float = 25.0,
    max_flight_path_angle_deg: float = 8.0,
    output_spacing_m: float = 2.0,
) -> np.ndarray:
    """
    WP2 -> WP3 -> WP4 -> WP5 고정익 Dubins 경로 생성 함수.

    반환값:
        np.ndarray, shape = (N, 3)
        각 행은 [x_ned, y_ned, z_ned]
    """

    wp2 = np.asarray(wp2, dtype=float).reshape(3)
    wp3 = np.asarray(wp3, dtype=float).reshape(3)
    wp4 = np.asarray(wp4, dtype=float).reshape(3)
    wp5 = np.asarray(wp5, dtype=float).reshape(3)

    yaw2 = _yaw_from_to(wp2, wp3)
    yaw3 = _yaw_from_to(wp3, wp4)
    yaw4 = _yaw_from_to(wp4, wp5)
    yaw5 = yaw4

    w2 = _make_waypoint("wp2", wp2, yaw2)
    w3 = _make_waypoint("wp3", wp3, yaw3)
    w4 = _make_waypoint("wp4", wp4, yaw4)
    w5 = _make_waypoint("wp5", wp5, yaw5)

    route: List[PathPoint] = []

    seg_23 = _safe_generate_fixedwing_path(
        start=w2,
        goal=w3,
        event_id=4,
        cruise_speed_mps=cruise_speed_mps,
        max_bank_deg=max_bank_deg,
        max_flight_path_angle_deg=max_flight_path_angle_deg,
        output_spacing_m=output_spacing_m,
    )

    seg_34 = _safe_generate_fixedwing_path(
        start=w3,
        goal=w4,
        event_id=5,
        cruise_speed_mps=cruise_speed_mps,
        max_bank_deg=max_bank_deg,
        max_flight_path_angle_deg=max_flight_path_angle_deg,
        output_spacing_m=output_spacing_m,
    )

    seg_45 = _safe_generate_fixedwing_path(
        start=w4,
        goal=w5,
        event_id=6,
        cruise_speed_mps=cruise_speed_mps,
        max_bank_deg=max_bank_deg,
        max_flight_path_angle_deg=max_flight_path_angle_deg,
        output_spacing_m=output_spacing_m,
    )

    _append_path_without_duplicate(route, seg_23)
    _append_path_without_duplicate(route, seg_34)
    _append_path_without_duplicate(route, seg_45)

    return pathpoints_to_ndarray(route)


def _make_line_path_points(
    start: Waypoint,
    goal: Waypoint,
    event_id: int,
    target_speed_mps: float,
    spacing_m: float,
) -> List[PathPoint]:
    start_pos = start.position
    goal_pos = goal.position

    vec = goal_pos - start_pos
    dist = float(np.linalg.norm(vec))

    if dist < 1e-6:
        return [
            PathPoint(
                x=goal.x,
                y=goal.y,
                z=goal.z,
                yaw=goal.yaw,
                event_id=event_id,
                target_speed_mps=target_speed_mps,
                segment_type=SegmentType.FIXEDWING_DUBINS,
                acceptance_radius_m=goal.acceptance_radius_m,
            )
        ]

    n = max(2, int(np.ceil(dist / max(spacing_m, 0.1))) + 1)
    yaw = _yaw_from_to(start_pos, goal_pos)

    points = []
    for i in range(n):
        alpha = i / (n - 1)
        p = (1.0 - alpha) * start_pos + alpha * goal_pos

        points.append(
            PathPoint(
                x=float(p[0]),
                y=float(p[1]),
                z=float(p[2]),
                yaw=float(yaw),
                event_id=event_id,
                target_speed_mps=target_speed_mps,
                segment_type=SegmentType.FIXEDWING_DUBINS,
                acceptance_radius_m=goal.acceptance_radius_m,
            )
        )

    return points


def _safe_generate_fixedwing_path(
    start: Waypoint,
    goal: Waypoint,
    event_id: int,
    cruise_speed_mps: float,
    max_bank_deg: float,
    max_flight_path_angle_deg: float,
    output_spacing_m: float,
) -> List[PathPoint]:
    try:
        return generate_fixedwing_dubins_path(
            start=start,
            goal=goal,
            event_id=event_id,
            cruise_speed_mps=cruise_speed_mps,
            max_bank_deg=max_bank_deg,
            max_flight_path_angle_deg=max_flight_path_angle_deg,
            output_spacing_m=output_spacing_m,
        )
    except ValueError as e:
        print(
            f"[fw_path_generator] Dubins failed from {start.name} to {goal.name}: {e}. "
            f"Fallback to straight line path."
        )

        return _make_line_path_points(
            start=start,
            goal=goal,
            event_id=event_id,
            target_speed_mps=cruise_speed_mps,
            spacing_m=output_spacing_m,
        )