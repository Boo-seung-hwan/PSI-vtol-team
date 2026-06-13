import math  # atan2, sin, cos, NaN 처리 등 수학 함수 사용
import time  # mission state가 시작된 시간과 경과 시간을 계산하기 위해 사용
from dataclasses import dataclass  # LookaheadResult처럼 여러 결과값을 묶는 클래스 생성용
from enum import Enum, auto  # mission state를 이름으로 관리하기 위해 사용
from typing import Optional  # current_pos처럼 값이 없을 수도 있는 변수 타입 표시

import numpy as np  # NED 좌표, 벡터, 거리 계산에 사용

import rclpy  # ROS2 Python 라이브러리
from rclpy.node import Node  # ROS2 노드를 만들기 위한 기본 클래스
from rclpy.qos import QoSProfile  # ROS2 topic QoS 설정용
from rclpy.qos import QoSReliabilityPolicy  # QoS reliability 설정용
from rclpy.qos import QoSDurabilityPolicy  # QoS durability 설정용
from rclpy.qos import QoSHistoryPolicy  # QoS history 설정용

from px4_msgs.msg import OffboardControlMode  # PX4 Offboard 제어 모드 heartbeat 메시지
from px4_msgs.msg import TrajectorySetpoint  # PX4에 position/yaw setpoint를 보내는 메시지
from px4_msgs.msg import VehicleCommand  # PX4 mode change, arm, VTOL transition 명령 메시지
from px4_msgs.msg import VehicleLocalPosition  # PX4 현재 local position, velocity 수신 메시지
from px4_msgs.msg import VehicleStatus  # PX4 현재 nav_state, arming_state 수신 메시지
from std_msgs.msg import String

from drone_control.fw_path_generator import generate_fw_generated_path
# ============================================================
# 공통 함수
# ============================================================

def distance_xy(a: np.ndarray, b: np.ndarray) -> float:
    # 두 점 a, b의 N, E 성분만 사용해서 수평거리 계산
    return float(np.linalg.norm(a[:2] - b[:2]))


def wrap_pi(angle: float) -> float:
    # 임의의 각도 angle을 -pi ~ pi 범위로 정리
    return math.atan2(math.sin(angle), math.cos(angle))


def make_straight_path(
    start: np.ndarray,
    end: np.ndarray,
    spacing_m: float = 2.0,
) -> np.ndarray:
    # start에서 end까지 직선 경로를 여러 개의 점으로 나누어 생성하는 함수

    start = np.asarray(start, dtype=float)  # start를 float형 numpy 배열로 변환
    end = np.asarray(end, dtype=float)  # end를 float형 numpy 배열로 변환

    segment = end - start  # start에서 end로 향하는 벡터 계산
    length = float(np.linalg.norm(segment))  # start와 end 사이의 3차원 거리 계산

    if length < 1e-6:  # start와 end가 거의 같은 위치라면
        return np.array([start.copy()], dtype=float)  # 경로점 하나만 반환

    num_steps = max(1, int(np.ceil(length / spacing_m)))  # spacing_m 간격이 되도록 나눌 개수 계산

    path = []  # 생성된 경로점을 저장할 리스트

    for i in range(num_steps + 1):  # 시작점부터 끝점까지 포함해야 하므로 +1
        ratio = i / num_steps  # 현재 점이 전체 구간에서 몇 % 위치인지 계산
        point = start + ratio * segment  # start와 end 사이의 보간점 계산
        path.append(point)  # 계산한 점을 path 리스트에 추가

    return np.array(path, dtype=float)  # 최종 경로를 numpy 배열로 반환


# ============================================================
# Adaptive Lookahead Follower
# ============================================================

@dataclass
class LookaheadResult:
    # AdaptiveLookaheadFollower.update()가 반환하는 결과를 묶는 클래스

    target: np.ndarray  # PX4에 보낼 목표점 [N, E, D]
    finished: bool  # 경로를 끝까지 따라갔는지 여부
    yaw: float  # 경로 진행 방향 yaw [rad]
    progress_s: float  # 경로 시작점부터 현재까지 진행한 거리 [m]
    cross_track_error: float  # 현재 위치가 경로에서 수평으로 얼마나 벗어났는지 [m]
    lookahead_distance: float  # 이번 주기에 실제로 사용한 lookahead 거리 [m]


class AdaptiveLookaheadFollower:

    def __init__(
        self,
        path: np.ndarray,
        base_lookahead: float = 20.0,
        min_lookahead: float = 8.0,
        max_lookahead: float = 45.0,
        speed_gain: float = 1.2,
        turn_slowdown_gain: float = 0.5,
        finish_radius: float = 15.0,
        finish_altitude_radius: float = 3.0,
    ):

        self.path = np.asarray(path, dtype=float)  # 입력 path를 float형 numpy 배열로 변환

        if self.path.ndim != 2 or self.path.shape[1] != 3:
            # path가 2차원 배열이고, 각 점이 [N, E, D] 형태인지 확인
            raise ValueError("path must have shape (N, 3)")  # 형식이 틀리면 에러 발생

        if len(self.path) < 2:
            # 경로점이 최소 2개는 있어야 segment를 만들 수 있음
            raise ValueError("path must contain at least 2 points")  # 경로점 부족 에러 발생

        self.path = self._remove_duplicate_points(self.path)  # 중복 경로점 제거

        self.base_lookahead = base_lookahead  # 기본 lookahead 거리 저장
        self.min_lookahead = min_lookahead  # 최소 lookahead 거리 저장
        self.max_lookahead = max_lookahead  # 최대 lookahead 거리 저장
        self.speed_gain = speed_gain  # 속도가 커질 때 lookahead를 얼마나 늘릴지 저장
        self.turn_slowdown_gain = turn_slowdown_gain  # 코너에서 lookahead를 얼마나 줄일지 저장
        self.finish_radius = finish_radius  # 마지막 점 수평 도착 판정 반경 저장
        self.finish_altitude_radius = finish_altitude_radius  # 마지막 점 고도 도착 판정 반경 저장

        self.segment_vectors = self.path[1:] - self.path[:-1]  # 각 경로점 사이의 벡터 계산
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)  # 각 segment 길이 계산

        self.cumulative_s = np.zeros(len(self.path))  # 각 path point까지의 누적거리 배열 생성
        self.cumulative_s[1:] = np.cumsum(self.segment_lengths)  # segment 길이 누적합 저장

        self.total_length = float(self.cumulative_s[-1])  # 전체 경로 길이 저장
        self.progress_s = 0.0  # 현재까지 진행한 거리 초기화
        self.last_segment_index = 0  # 마지막으로 가까웠던 segment index 초기화

    def _remove_duplicate_points(self, path: np.ndarray) -> np.ndarray:
        # 서로 거의 같은 점이 연속으로 들어온 경우 제거하는 함수

        clean = [path[0]]  # 첫 번째 점은 무조건 유지

        for p in path[1:]:  # 두 번째 점부터 끝까지 검사
            if np.linalg.norm(p - clean[-1]) > 1e-6:
                # 바로 이전에 저장된 점과 충분히 다르면
                clean.append(p)  # 중복이 아니므로 추가

        return np.array(clean, dtype=float)  # 중복 제거된 경로 반환

    def reset(self):
        # 경로 추종 상태를 초기화하는 함수

        self.progress_s = 0.0  # 진행거리를 0으로 초기화
        self.last_segment_index = 0  # segment index도 0으로 초기화

    def _segment_index_from_s(self, s: float) -> int:
        # 경로 진행거리 s가 어느 segment에 속하는지 찾는 함수

        idx = int(np.searchsorted(self.cumulative_s, s, side="right") - 1)
        # cumulative_s 배열에서 s가 들어갈 위치를 찾고, 그 앞 segment index를 계산

        return int(np.clip(idx, 0, len(self.segment_lengths) - 1))
        # index가 0보다 작거나 마지막 segment를 넘지 않도록 제한해서 반환

    def _turn_factor_at_s(self, s: float) -> float:
        # 현재 위치 근처의 코너 정도를 보고 lookahead를 줄일 계수를 계산하는 함수

        idx = self._segment_index_from_s(s)  # 현재 s가 포함된 segment index 계산

        if idx >= len(self.segment_vectors) - 1:
            # 현재 segment가 마지막 segment면 다음 segment가 없음
            return 1.0  # 코너 판단 불가하므로 factor 1 반환

        v1 = self.segment_vectors[idx][:2]  # 현재 segment의 수평 방향 벡터 [N, E]
        v2 = self.segment_vectors[idx + 1][:2]  # 다음 segment의 수평 방향 벡터 [N, E]

        n1 = np.linalg.norm(v1)  # 현재 segment의 수평 길이
        n2 = np.linalg.norm(v2)  # 다음 segment의 수평 길이

        if n1 < 1e-6 or n2 < 1e-6:
            # 둘 중 하나의 길이가 거의 0이면 방향 계산이 불안정함
            return 1.0  # 코너 판단하지 않고 factor 1 반환

        yaw1 = math.atan2(v1[1], v1[0])  # 현재 segment의 yaw 계산
        yaw2 = math.atan2(v2[1], v2[0])  # 다음 segment의 yaw 계산

        turn_angle = abs(wrap_pi(yaw2 - yaw1))  # 두 segment 방향 차이를 -pi~pi로 정리 후 절댓값 계산

        turn_strength = turn_angle / math.pi  # 회전 강도를 0~1 정도로 정규화

        factor = 1.0 - self.turn_slowdown_gain * turn_strength
        # 코너가 급할수록 factor가 작아짐

        return float(np.clip(factor, 0.45, 1.0))
        # factor가 너무 작아지지 않도록 0.45~1.0 사이로 제한

    def _adaptive_lookahead(self, speed_xy: float, closest_s: float) -> float:
        # 현재 수평 속도와 코너 정도를 반영해서 lookahead 거리 계산

        lookahead = self.base_lookahead + self.speed_gain * speed_xy
        # 기본 lookahead에 현재 수평 속도에 비례한 값을 더함

        turn_factor = self._turn_factor_at_s(closest_s)
        # 현재 위치 근처가 코너인지 확인해서 감소 계수 계산

        lookahead *= turn_factor
        # 코너가 급하면 lookahead를 줄임

        lookahead = float(np.clip(
            lookahead,
            self.min_lookahead,
            self.max_lookahead,
        ))
        # lookahead가 최소/최대 범위를 넘지 않도록 제한

        return lookahead  # 최종 lookahead 거리 반환

    def find_closest_s(self, current_pos: np.ndarray) -> tuple[float, float]:
        # 현재 위치와 가장 가까운 경로상 진행거리 s를 찾는 함수

        current_pos = np.asarray(current_pos, dtype=float)  # 현재 위치를 numpy 배열로 변환

        best_distance = float("inf")  # 가장 작은 수평 거리 초기값을 무한대로 설정
        best_s = self.progress_s  # 가장 가까운 진행거리 초기값을 현재 progress_s로 설정

        for i in range(len(self.segment_vectors)):
            # 모든 segment를 순서대로 검사

            start_xy = self.path[i][:2]  # 현재 segment 시작점의 수평 좌표 [N, E]
            seg_xy = self.segment_vectors[i][:2]  # 현재 segment의 수평 방향 벡터 [N, E]
            current_xy = current_pos[:2]  # 현재 기체 위치의 수평 좌표 [N, E]

            seg_len_xy = np.linalg.norm(seg_xy)  # 현재 segment의 수평 길이 계산

            if seg_len_xy < 1e-6:
                # 수평 길이가 거의 0이면 투영 계산 불가
                continue  # 이 segment는 건너뜀

            t = np.dot(current_xy - start_xy, seg_xy) / (seg_len_xy ** 2)
            # 현재 위치를 segment 위로 투영했을 때 segment의 몇 % 지점인지 계산

            t = np.clip(t, 0.0, 1.0)
            # t가 0보다 작거나 1보다 크면 segment 바깥이므로 0~1로 제한

            closest_xy = start_xy + t * seg_xy
            # segment 위에서 current_xy와 가장 가까운 점 계산

            distance_xy = float(np.linalg.norm(current_xy - closest_xy))
            # 현재 위치와 closest_xy 사이의 수평 거리 계산

            if distance_xy < best_distance:
                # 이번 segment에서의 거리가 지금까지 최소 거리보다 작다면

                best_distance = distance_xy  # 최소 거리 갱신
                best_s = float(self.cumulative_s[i] + t * self.segment_lengths[i])
                # 경로 시작점부터 closest point까지의 진행거리 계산

                self.last_segment_index = i  # 현재 가장 가까운 segment index 저장

        best_s = max(best_s, self.progress_s)
        # progress_s가 뒤로 줄어들지 않도록 제한
        # target이 갑자기 뒤쪽으로 튀는 것을 방지

        return best_s, best_distance
        # 가장 가까운 경로 진행거리와 경로 이탈거리 반환

    def interpolate_at_s(self, s: float) -> np.ndarray:
        # 경로 진행거리 s에 해당하는 실제 좌표 [N, E, D]를 계산하는 함수

        s = float(np.clip(s, 0.0, self.total_length))
        # s가 경로 시작보다 작거나 끝보다 크지 않도록 제한

        for i in range(len(self.segment_lengths)):
            # 모든 segment를 순서대로 검사

            s_start = self.cumulative_s[i]  # 현재 segment 시작점의 누적거리
            s_end = self.cumulative_s[i + 1]  # 현재 segment 끝점의 누적거리

            if s_start <= s <= s_end:
                # 입력 s가 현재 segment 안에 있으면

                seg_len = self.segment_lengths[i]  # 현재 segment 길이 가져오기

                if seg_len < 1e-6:
                    # segment 길이가 거의 0이면
                    return self.path[i].copy()  # 시작점을 그대로 반환

                ratio = (s - s_start) / seg_len
                # 현재 segment 안에서 s가 몇 % 위치인지 계산

                return self.path[i] + ratio * self.segment_vectors[i]
                # 해당 비율 위치의 실제 [N, E, D] 좌표 반환

        return self.path[-1].copy()
        # 혹시 segment를 못 찾으면 마지막 경로점 반환

    def get_yaw_at_s(self, s: float) -> float:
        # 경로 진행거리 s 지점에서의 진행 방향 yaw 계산

        s = float(np.clip(s, 0.0, self.total_length))
        # s를 경로 범위 안으로 제한

        idx = self._segment_index_from_s(s)
        # s가 포함된 segment index 계산

        direction = self.segment_vectors[idx]
        # 해당 segment의 방향 벡터 가져오기

        return math.atan2(direction[1], direction[0])
        # NED 기준 yaw = atan2(East 방향 성분, North 방향 성분)

    def update(
        self,
        current_pos: np.ndarray,
        current_vel: Optional[np.ndarray] = None,
    ) -> LookaheadResult:
        # 매 제어 주기마다 호출해서 현재 위치 기준 lookahead target 계산

        current_pos = np.asarray(current_pos, dtype=float)  # 현재 위치를 numpy 배열로 변환

        if current_vel is None:
            # 현재 속도 정보를 못 받았다면
            speed_xy = 0.0  # 속도를 0으로 간주
        else:
            # 현재 속도 정보가 있다면
            current_vel = np.asarray(current_vel, dtype=float)  # 속도를 numpy 배열로 변환
            speed_xy = float(np.linalg.norm(current_vel[:2]))  # 수평 속도 크기 계산

        closest_s, cross_track_error = self.find_closest_s(current_pos)
        # 현재 위치와 가장 가까운 경로 진행거리 closest_s와 경로 이탈거리 계산

        self.progress_s = closest_s
        # 현재 진행거리 업데이트

        lookahead_distance = self._adaptive_lookahead(
            speed_xy=speed_xy,
            closest_s=closest_s,
        )
        # 속도와 코너를 고려해서 이번 주기의 lookahead 거리 계산

        target_s = closest_s + lookahead_distance
        # 현재 위치보다 lookahead 거리만큼 앞쪽의 경로 진행거리 계산

        target_s = min(target_s, self.total_length)
        # target_s가 경로 끝을 넘지 않도록 제한

        target = self.interpolate_at_s(target_s)
        # target_s에 해당하는 실제 [N, E, D] 좌표 계산

        yaw = self.get_yaw_at_s(target_s)
        # target_s 지점에서의 진행 방향 yaw 계산

        final_point = self.path[-1]
        # 경로 마지막 점 가져오기

        final_xy_error = float(np.linalg.norm(current_pos[:2] - final_point[:2]))
        # 현재 위치와 마지막 점 사이의 수평 거리 계산

        final_z_error = abs(float(current_pos[2] - final_point[2]))
        # 현재 위치와 마지막 점 사이의 고도 차이 계산

        finished = (
            final_xy_error < self.finish_radius
            and final_z_error < self.finish_altitude_radius
        ) or self.progress_s >= self.total_length - 1e-3
        # 마지막 점 근처에 도착했거나, 진행거리가 전체 경로 길이에 거의 도달하면 finished=True

        return LookaheadResult(
            target=target,
            finished=finished,
            yaw=yaw,
            progress_s=self.progress_s,
            cross_track_error=cross_track_error,
            lookahead_distance=lookahead_distance,
        )
        # 계산 결과를 LookaheadResult 형태로 반환


# ============================================================
# Mission State
# ============================================================

class MissionState(Enum):
    # 전체 임무 상태를 이름으로 관리하기 위한 Enum

    WAIT_OFFBOARD = auto()
    # Offboard 진입 전에 setpoint stream을 보내는 준비 상태

    TAKEOFF_MC_AT_WP1 = auto()
    # WP1에서 회전익 모드로 상승하는 상태

    TRANSITION_TO_FW_TO_WP2 = auto()
    # WP2 방향으로 가면서 회전익에서 고정익으로 전환하는 상태

    FOLLOW_FW_GENERATED_PATH = auto()
    # 고정익 상태에서 WP2→WP5 generated_path를 추종하는 상태

    TRANSITION_TO_MC_WP5_TO_REP = auto()
    # WP5에서 REP 방향으로 가면서 고정익에서 회전익으로 전환하는 상태

    FOLLOW_MC_PATH_TO_REP = auto()
    # 회전익 전환 완료 후 REP까지 이동하는 상태

    TRANSITION_TO_FW_REP_TO_WP5_RETURN = auto()
    FOLLOW_FW_GENERATED_PATH_RETURN = auto()
    TRANSITION_TO_MC_WP2_TO_WP1_RETURN = auto()
    FOLLOW_MC_PATH_TO_WP1 = auto()

    MISSION_DONE = auto()
    # 임무 완료 후 현재 위치 유지 상태


# ============================================================
# VTOL Mission Manager
# ============================================================

class VTOLMissionManager(Node):
    # 전체 미션을 관리하는 ROS2 노드 클래스

    def __init__(self):
        # 노드가 생성될 때 실행되는 초기화 함수

        super().__init__("generator_tracking")
        # ROS2 노드 이름을 generator_tracking로 설정

        # ----------------------------
        # Mission parameters
        # ----------------------------
        self.cruise_altitude = 30.0
        # 고정익 순항 고도 [m]

        self.rep_altitude = 5.0
        # REP 접근 고도 [m]

        self.takeoff_accept_xy = 1.5
        # WP1 상승 완료 판정용 수평 허용 오차 [m]

        self.takeoff_accept_z = 1.0
        # WP1 상승 완료 판정용 고도 허용 오차 [m]

        self.fw_transition_wait_time = 7.0
        # 회전익→고정익 전환을 기다리는 시간 [s]

        self.mc_transition_wait_time = 6.0
        # 고정익→회전익 전환을 기다리는 시간 [s]

        # ----------------------------
        # State variables
        # ----------------------------
        self.state = MissionState.WAIT_OFFBOARD
        # 초기 mission state를 WAIT_OFFBOARD로 설정

        self.mission_state = "INACTIVE"

        self.state_start_time = time.monotonic()
        # 현재 state가 시작된 시간 저장

        self.transition_command_sent = False
        # VTOL transition 명령을 한 번만 보내기 위한 flag

        self.offboard_counter = 0
        # Offboard 진입 전 setpoint를 몇 번 보냈는지 세는 counter

        self.current_pos: Optional[np.ndarray] = None
        # 현재 기체 위치 [N, E, D], 아직 못 받았으면 None

        self.current_vel: Optional[np.ndarray] = None
        # 현재 기체 속도 [vN, vE, vD], 아직 못 받았으면 None

        self.current_yaw: float = 0.0
        # 현재 yaw 저장, 기본값 0

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        # PX4 navigation state 저장용 변수

        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        # PX4 arm 상태 저장용 변수

        # ----------------------------
        # QoS
        # ----------------------------
        self.qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # PX4로 publish할 때 사용하는 QoS 설정

        self.qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # PX4에서 subscribe할 때 사용하는 QoS 설정

        # ----------------------------
        # Publishers
        # ----------------------------
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            self.qos_pub,
        )
        # PX4로 OffboardControlMode를 보내는 publisher 생성

        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/generator_tracking/trajectory_setpoint",
            self.qos_pub,
        )
        # PX4로 직접 보내는 것이 아니라 setpoint_mux로 보낼 generator setpoint publisher

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            self.qos_pub,
        )
        # PX4로 VehicleCommand를 보내는 publisher 생성


        self.status_pub = self.create_publisher(
            String,
            "/generator_tracking/status",
            10,
        )
        # MissionManager에게 generator_tracking 진행 상태를 알려주는 publisher

        # ----------------------------
        # Subscribers
        # ----------------------------
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_position_callback,
            self.qos_sub,
        )
        # PX4 local position을 받는 subscriber 생성

        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            self.qos_sub,
        )
        # PX4 vehicle status를 받는 subscriber 생성

        self.mission_state_sub = self.create_subscription(
            String,
            "/mission/state",
            self.mission_state_callback,
            10,
        )
        # 최상위 MissionManager가 보내는 현재 임무 상태 구독

        # ----------------------------
        # Waypoint, path, follower setup
        # ----------------------------
        self.load_waypoints_and_paths()
        # waypoint, generated_path, follower 초기화


        self.timer = self.create_timer(0.05, self.timer_callback)
        # 0.05초마다 timer_callback 실행, 즉 20Hz 제어 루프

    # ========================================================
    # Waypoints and paths
    # ========================================================

    def load_waypoints_and_paths(self):
        # waypoint와 경로, follower를 설정하는 함수

        self.wp1 = np.array([0.0, 8.0, -self.cruise_altitude], dtype=float)
        # WP1 좌표, N=0, E=0, 고도 30m

        self.wp2 = np.array([50.0, 0.0, -self.cruise_altitude], dtype=float)
        # WP2 좌표, 고정익 경로 시작점

        self.wp3 = np.array([100.0, 40.0, -self.cruise_altitude], dtype=float)
        # WP3 좌표, QGC waypoint 예시

        self.wp4 = np.array([150.0, 40.0, -self.cruise_altitude], dtype=float)
        # WP4 좌표, QGC waypoint 예시

        self.wp5 = np.array([100.0, 0.0, -self.cruise_altitude], dtype=float)
        # WP5 좌표, 고정익 generated_path 끝점

        self.rep = np.array([10.0, 30.0, -self.rep_altitude], dtype=float)
        # REP 좌표, 회전익 전환 후 접근할 지점

        self.fw_generated_path = generate_fw_generated_path(
            wp2=self.wp2,
            wp3=self.wp3,
            wp4=self.wp4,
            wp5=self.wp5,
            cruise_speed_mps=12.0,
            max_bank_deg=35.0,
            max_flight_path_angle_deg=8.0,
            output_spacing_m=2.0,
        )

        if distance_xy(self.fw_generated_path[0], self.wp2) > 1.0:
            # generated_path 시작점이 WP2와 1m 이상 다르면
            self.fw_generated_path = np.vstack([self.wp2, self.fw_generated_path])
            # WP2를 경로 맨 앞에 붙임

        if distance_xy(self.fw_generated_path[-1], self.wp5) > 1.0:
            # generated_path 끝점이 WP5와 1m 이상 다르면
            self.fw_generated_path = np.vstack([self.fw_generated_path, self.wp5])
            # WP5를 경로 맨 뒤에 붙임

        self.fw_follower = AdaptiveLookaheadFollower(
            path=self.fw_generated_path,
            base_lookahead=20.0,
            min_lookahead=12.0,
            max_lookahead=45.0,
            speed_gain=1.2,
            turn_slowdown_gain=0.5,
            finish_radius=20.0,
            finish_altitude_radius=3.0,
        )
        # 고정익용 AdaptiveLookaheadFollower 생성


        rep_target = self.rep.copy()
        # REP 좌표 복사

        rep_target[2] = -self.rep_altitude
        # REP 목표 고도를 rep_altitude로 설정

        wp5_mc_start = self.wp5.copy()
        # WP5 좌표 복사

        wp5_mc_start[2] = -self.rep_altitude
        # 회전익 REP 이동 시작 고도를 REP 고도와 맞춤

        self.mc_path_to_rep = make_straight_path(
            start=wp5_mc_start,
            end=rep_target,
            spacing_m=2.0,
        )
        # WP5에서 REP까지 회전익용 직선 경로 생성

        self.mc_follower_to_rep = AdaptiveLookaheadFollower(
            path=self.mc_path_to_rep,
            base_lookahead=4.0,
            min_lookahead=2.0,
            max_lookahead=8.0,
            speed_gain=0.4,
            turn_slowdown_gain=0.3,
            finish_radius=1.5,
            finish_altitude_radius=0.8,
        )
        # 회전익 REP 이동용 AdaptiveLookaheadFollower 생성

        self.fw_return_path = np.flip(self.fw_generated_path, axis=0).copy()

        self.fw_return_follower = AdaptiveLookaheadFollower(
            path=self.fw_return_path,
            base_lookahead=20.0,
            min_lookahead=12.0,
            max_lookahead=45.0,
            speed_gain=1.2,
            turn_slowdown_gain=0.5,
            finish_radius=20.0,
            finish_altitude_radius=3.0,
        )

        # 회전익 복귀 경로: WP2 → WP1
        wp2_mc_start = self.wp2.copy()
        wp2_mc_start[2] = -self.rep_altitude

        wp1_mc_target = self.wp1.copy()
        wp1_mc_target[2] = -self.rep_altitude

        self.mc_path_to_wp1 = make_straight_path(
            start=wp2_mc_start,
            end=wp1_mc_target,
            spacing_m=2.0,
        )

        self.mc_follower_to_wp1 = AdaptiveLookaheadFollower(
            path=self.mc_path_to_wp1,
            base_lookahead=4.0,
            min_lookahead=2.0,
            max_lookahead=8.0,
            speed_gain=0.4,
            turn_slowdown_gain=0.3,
            finish_radius=1.5,
            finish_altitude_radius=0.8,
        )


    # ========================================================
    # PX4 callbacks
    # ========================================================

    def local_position_callback(self, msg: VehicleLocalPosition):
        # PX4 local position topic을 받을 때마다 실행되는 callback

        self.current_pos = np.array([msg.x, msg.y, msg.z], dtype=float)
        # 현재 위치 [N, E, D] 저장

        self.current_vel = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
        # 현재 속도 [vN, vE, vD] 저장

        try:
            self.current_yaw = float(msg.heading)
            # msg에 heading 필드가 있으면 현재 yaw로 저장
        except AttributeError:
            self.current_yaw = 0.0
            # heading 필드가 없으면 yaw를 0으로 둠

    def vehicle_status_callback(self, msg: VehicleStatus):
        # PX4 vehicle_status topic을 받을 때마다 실행되는 callback

        self.nav_state = msg.nav_state
        # 현재 PX4 navigation state 저장

        self.arming_state = msg.arming_state
        # 현재 PX4 arming state 저장


    def mission_state_callback(self, msg: String):
    # 최상위 MissionManager가 보낸 mission state를 저장

        new_state = msg.data.strip()

        if new_state != self.mission_state:
            self.get_logger().info(
                f"Mission state changed: {self.mission_state} -> {new_state}"
            )

            self.mission_state = new_state

            if new_state == "GENERATOR_OUTBOUND":
                # 출발 경로 추종이 시작되면 내부 상태 초기화
                self.state = MissionState.WAIT_OFFBOARD
                self.offboard_counter = 0
                self.transition_command_sent = False
                self.fw_follower.reset()
                self.mc_follower_to_rep.reset()

            elif new_state == "GENERATOR_RETURN":
                # 구조 완료 후 복귀 경로 추종 시작
                self.state = MissionState.TRANSITION_TO_FW_REP_TO_WP5_RETURN
                self.transition_command_sent = False
                self.fw_return_follower.reset()

    # ========================================================
    # PX4 publish functions
    # ========================================================

    def now_us(self) -> int:
        # 현재 ROS 시간을 PX4 timestamp 형식인 microsecond로 변환하는 함수

        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_offboard_mode_position(self):
        # PX4에 position setpoint를 사용할 것이라고 알려주는 OffboardControlMode publish 함수

        msg = OffboardControlMode()
        # OffboardControlMode 메시지 생성

        msg.timestamp = self.now_us()
        # timestamp 입력

        msg.position = True
        # position 제어 활성화

        msg.velocity = False
        # velocity 제어 비활성화

        msg.acceleration = False
        # acceleration 제어 비활성화

        msg.attitude = False
        # attitude 직접 제어 비활성화

        msg.body_rate = False
        # body rate 직접 제어 비활성화

        msg.thrust_and_torque = False
        # thrust/torque 직접 제어 비활성화

        msg.direct_actuator = False
        # actuator 직접 제어 비활성화

        self.offboard_control_mode_pub.publish(msg)
        # PX4로 OffboardControlMode publish

    def publish_position_setpoint(self, target: np.ndarray, yaw: float = math.nan):
        # PX4에 position setpoint를 보내는 함수

        msg = TrajectorySetpoint()
        # TrajectorySetpoint 메시지 생성

        msg.timestamp = self.now_us()
        # timestamp 입력

        msg.position = [
            float(target[0]),
            float(target[1]),
            float(target[2]),
        ]
        # 목표 위치 [N, E, D] 입력

        msg.velocity = [math.nan, math.nan, math.nan]
        # velocity setpoint는 사용하지 않으므로 NaN 입력

        msg.acceleration = [math.nan, math.nan, math.nan]
        # acceleration setpoint는 사용하지 않으므로 NaN 입력

        msg.yaw = float(yaw) if math.isfinite(yaw) else math.nan
        # yaw가 유효하면 입력하고, 아니면 NaN 입력

        msg.yawspeed = math.nan
        # yaw rate는 사용하지 않음

        self.trajectory_setpoint_pub.publish(msg)
        # PX4로 TrajectorySetpoint publish

    def publish_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ):
        # PX4에 VehicleCommand를 보내는 공통 함수

        msg = VehicleCommand()
        # VehicleCommand 메시지 생성

        msg.timestamp = self.now_us()
        # timestamp 입력

        msg.command = command
        # command 종류 입력

        msg.param1 = float(param1)
        # command parameter 1 입력

        msg.param2 = float(param2)
        # command parameter 2 입력

        msg.target_system = 1
        # target system ID 설정

        msg.target_component = 1
        # target component ID 설정

        msg.source_system = 1
        # source system ID 설정

        msg.source_component = 1
        # source component ID 설정

        msg.from_external = True
        # 외부 companion computer에서 보낸 명령임을 표시

        self.vehicle_command_pub.publish(msg)
        # PX4로 VehicleCommand publish

    def publish_status(self, text: str):
        # generator_tracking 상태를 MissionManager에게 publish

        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def arm(self):
        # PX4 arm 명령 함수

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
            0.0,
        )
        # param1=1.0이면 arm

    def set_offboard_mode(self):
        # PX4를 Offboard mode로 바꾸는 함수

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )
        # param1=1, param2=6 조합으로 Offboard mode 요청

    def transition_to_fw(self):
        # 회전익에서 고정익으로 전환하는 명령 함수

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_VTOL_TRANSITION,
            4.0,
            0.0,
        )
        # param1=4.0은 고정익 상태로 전환 의미

    def transition_to_mc(self):
        # 고정익에서 회전익으로 전환하는 명령 함수

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_VTOL_TRANSITION,
            3.0,
            0.0,
        )
        # param1=3.0은 회전익 상태로 전환 의미

    # ========================================================
    # State helper
    # ========================================================

    def change_state(self, new_state: MissionState):
        # mission state를 바꾸는 함수

        if self.state == new_state:
            # 이미 같은 상태라면
            return
            # 아무것도 하지 않음

        self.get_logger().warn(
            f"STATE CHANGE: {self.state.name} -> {new_state.name}"
        )
        # 상태 전환 로그 출력

        self.state = new_state
        # 현재 상태 변경

        self.state_start_time = time.monotonic()
        # 새 상태가 시작된 시간 저장

        self.transition_command_sent = False
        # 새 상태에 들어왔으므로 transition 명령 flag 초기화

        if new_state == MissionState.FOLLOW_FW_GENERATED_PATH:
            # 고정익 generated_path 추종 상태로 들어가면
            self.fw_follower.reset()
            # 고정익 follower 초기화

        if new_state == MissionState.FOLLOW_MC_PATH_TO_REP:
            # 회전익 REP 추종 상태로 들어가면
            self.mc_follower_to_rep.reset()
            # 회전익 follower 초기화

    # ========================================================
    # Main timer
    # ========================================================

    def timer_callback(self):
        # 20Hz로 반복 실행되는 메인 제어 루프

        if self.mission_state not in ["GENERATOR_OUTBOUND", "GENERATOR_RETURN"]:
            return


        if self.mission_state == "GENERATOR_OUTBOUND":
            self.publish_status("GENERATOR_OUTBOUND_ACTIVE")

        elif self.mission_state == "GENERATOR_RETURN":
            self.publish_status("GENERATOR_RETURN_ACTIVE")

        if self.current_pos is None:
            return

        if self.current_pos is None:
            # 아직 PX4 위치 정보를 받지 못했다면
            return
            # 제어하지 않음

        if self.state == MissionState.WAIT_OFFBOARD:
            # 현재 상태가 Offboard 준비 상태라면
            self.handle_wait_offboard()
            # Offboard 준비 함수 실행

        elif self.state == MissionState.TAKEOFF_MC_AT_WP1:
            # 현재 상태가 WP1 회전익 상승 상태라면
            self.handle_takeoff_mc_at_wp1()
            # 이륙 상태 함수 실행

        elif self.state == MissionState.TRANSITION_TO_FW_TO_WP2:
            # 현재 상태가 WP2 방향 고정익 전환 상태라면
            self.handle_transition_to_fw_to_wp2()
            # 회전익→고정익 전환 함수 실행

        elif self.state == MissionState.FOLLOW_FW_GENERATED_PATH:
            # 현재 상태가 고정익 generated_path 추종 상태라면
            self.handle_follow_fw_generated_path()
            # 고정익 경로 추종 함수 실행

        elif self.state == MissionState.TRANSITION_TO_MC_WP5_TO_REP:
            # 현재 상태가 REP 방향 회전익 전환 상태라면
            self.handle_transition_to_mc_wp5_to_rep()
            # 고정익→회전익 전환 함수 실행

        elif self.state == MissionState.FOLLOW_MC_PATH_TO_REP:
            # 현재 상태가 회전익 REP 경로 추종 상태라면
            self.handle_follow_mc_path_to_rep()
            # 회전익 경로 추종 함수 실행

        elif self.state == MissionState.MISSION_DONE:
            # 현재 상태가 임무 완료 상태라면
            self.handle_mission_done()
            # 현재 위치 유지 함수 실행

        elif self.state == MissionState.TRANSITION_TO_FW_REP_TO_WP5_RETURN:
            self.handle_transition_to_fw_rep_to_wp5_return()

        elif self.state == MissionState.FOLLOW_FW_GENERATED_PATH_RETURN:
            self.handle_follow_fw_generated_path_return()

        elif self.state == MissionState.TRANSITION_TO_MC_WP2_TO_WP1_RETURN:
            self.handle_transition_to_mc_wp2_to_wp1_return()

        elif self.state == MissionState.FOLLOW_MC_PATH_TO_WP1:
            self.handle_follow_mc_path_to_wp1()

    # ========================================================
    # State handlers
    # ========================================================

    def handle_wait_offboard(self):
        # Offboard 진입 전에 setpoint stream을 먼저 보내는 상태

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        hold = self.current_pos.copy()
        # 현재 위치를 hold target으로 설정

        self.publish_position_setpoint(hold)
        # 현재 위치 유지 setpoint publish

        self.offboard_counter += 1
        # setpoint를 보낸 횟수 증가

        if self.offboard_counter == 20:
            # 약 1초 정도 setpoint를 보낸 뒤
            self.set_offboard_mode()
            # Offboard mode 요청

            self.arm()
            # arm 요청

        if self.offboard_counter > 30:
            # setpoint stream을 충분히 보냈다면
            self.change_state(MissionState.TAKEOFF_MC_AT_WP1)
            # WP1 상승 상태로 전환

    def handle_takeoff_mc_at_wp1(self):
        # WP1에서 회전익 모드로 상승하는 상태

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        target = self.wp1.copy()
        # WP1 좌표 복사

        target[2] = -self.cruise_altitude
        # 목표 고도를 cruise_altitude로 설정

        self.publish_position_setpoint(target)
        # WP1 상공 목표 위치 publish

        xy_error = distance_xy(self.current_pos, target)
        # 현재 위치와 target 사이 수평 오차 계산

        z_error = abs(self.current_pos[2] - target[2])
        # 현재 위치와 target 사이 고도 오차 계산

        if xy_error < self.takeoff_accept_xy and z_error < self.takeoff_accept_z:
            # 수평 오차와 고도 오차가 허용 범위 안이면
            self.change_state(MissionState.TRANSITION_TO_FW_TO_WP2)
            # WP2 방향 회전익→고정익 전환 상태로 이동

    def handle_transition_to_fw_to_wp2(self):
        # WP2 방향으로 이동하면서 회전익에서 고정익으로 전환하는 상태

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        target = self.wp2.copy()
        # WP2를 전환 중 목표점으로 설정

        target[2] = -self.cruise_altitude
        # 목표 고도는 고정익 순항 고도로 설정

        self.publish_position_setpoint(target)
        # WP2 방향 position setpoint publish

        if not self.transition_command_sent:
            # 아직 transition 명령을 안 보냈다면
            self.get_logger().warn("Command: VTOL transition to fixed-wing")
            # 로그 출력

            self.transition_to_fw()
            # 회전익→고정익 전환 명령 전송

            self.transition_command_sent = True
            # transition 명령을 보냈다고 표시

        elapsed = time.monotonic() - self.state_start_time
        # 이 상태에 들어온 뒤 지난 시간 계산

        if elapsed > self.fw_transition_wait_time:
            # 지정한 전환 대기 시간이 지나면
            self.change_state(MissionState.FOLLOW_FW_GENERATED_PATH)
            # 고정익 경로 추종 상태로 전환

    def handle_follow_fw_generated_path(self):
        # 고정익 상태에서 WP2→WP5 generated_path를 따라가는 상태

        if self.current_vel is None:
            # 아직 현재 속도 정보를 받지 못했다면
            return
            # 제어하지 않음

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        result = self.fw_follower.update(
            current_pos=self.current_pos,
            current_vel=self.current_vel,
        )
        # 현재 위치와 속도를 넣어서 고정익 lookahead target 계산

        self.publish_position_setpoint(
            target=result.target,
            yaw=result.yaw,
        )
        # 계산된 lookahead target과 yaw를 PX4로 publish

        if result.finished:
            # generated_path 끝에 도달했다면
            self.change_state(MissionState.TRANSITION_TO_MC_WP5_TO_REP)
            # WP5→REP 방향 고정익→회전익 전환 상태로 이동

    def handle_transition_to_mc_wp5_to_rep(self):
        # WP5에서 REP 방향으로 이동하면서 고정익에서 회전익으로 전환하는 상태

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        target = self.rep.copy()
        # REP를 전환 중 목표점으로 설정

        target[2] = -self.rep_altitude
        # 목표 고도는 REP 고도로 설정

        self.publish_position_setpoint(target)
        # REP 방향 position setpoint publish

        if not self.transition_command_sent:
            # 아직 transition 명령을 보내지 않았다면
            self.get_logger().warn(
                "Command: VTOL transition to multicopter while heading to REP"
            )
            # 로그 출력

            self.transition_to_mc()
            # 고정익→회전익 전환 명령 전송

            self.transition_command_sent = True
            # transition 명령을 보냈다고 표시

        elapsed = time.monotonic() - self.state_start_time
        # 이 상태에 들어온 뒤 지난 시간 계산

        if elapsed > self.mc_transition_wait_time:
            # 지정한 회전익 전환 시간이 지나면
            self.change_state(MissionState.FOLLOW_MC_PATH_TO_REP)
            # 회전익 REP 경로 추종 상태로 이동

    def handle_follow_mc_path_to_rep(self):
        # 회전익 전환 완료 후 REP까지 경로를 따라가는 상태

        if self.current_vel is None:
            # 아직 현재 속도 정보를 받지 못했다면
            return
            # 제어하지 않음

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        result = self.mc_follower_to_rep.update(
            current_pos=self.current_pos,
            current_vel=self.current_vel,
        )
        # 현재 위치와 속도를 넣어서 회전익용 lookahead target 계산

        self.publish_position_setpoint(
            target=result.target,
            yaw=result.yaw,
        )
        # 계산된 회전익용 lookahead target publish

        if result.finished:
            # REP 경로 끝에 도달했다면
            self.publish_status("GENERATOR_OUTBOUND_DONE")
            return

    def handle_transition_to_fw_rep_to_wp5_return(self):
        # 구조 완료 후 REP에서 WP5 방향으로 이동하면서 고정익 전환

        self.publish_offboard_mode_position()

        target = self.wp5.copy()
        target[2] = -self.cruise_altitude

        self.publish_position_setpoint(target)

        if not self.transition_command_sent:
            self.get_logger().warn("Return: VTOL transition to fixed-wing while heading to WP5")
            self.transition_to_fw()
            self.transition_command_sent = True

        elapsed = time.monotonic() - self.state_start_time

        if elapsed > self.fw_transition_wait_time:
            self.change_state(MissionState.FOLLOW_FW_GENERATED_PATH_RETURN)
        

    def handle_follow_fw_generated_path_return(self):
        # 고정익 복귀 경로 WP5 → WP2 추종

        if self.current_vel is None:
            return

        self.publish_offboard_mode_position()

        result = self.fw_return_follower.update(
            current_pos=self.current_pos,
            current_vel=self.current_vel,
        )

        self.publish_position_setpoint(
            target=result.target,
            yaw=result.yaw,
        )

        if result.finished:
            self.change_state(MissionState.TRANSITION_TO_MC_WP2_TO_WP1_RETURN)

    def handle_transition_to_mc_wp2_to_wp1_return(self):
        # WP2에서 WP1 방향으로 이동하면서 회전익 전환

        self.publish_offboard_mode_position()

        target = self.wp1.copy()
        target[2] = -self.rep_altitude

        self.publish_position_setpoint(target)

        if not self.transition_command_sent:
            self.get_logger().warn("Return: VTOL transition to multicopter while heading to WP1")
            self.transition_to_mc()
            self.transition_command_sent = True

        elapsed = time.monotonic() - self.state_start_time

        if elapsed > self.mc_transition_wait_time:
            self.change_state(MissionState.FOLLOW_MC_PATH_TO_WP1)

    def handle_follow_mc_path_to_wp1(self):
        # 회전익 복귀 경로 WP2 → WP1 추종

        if self.current_vel is None:
            return

        self.publish_offboard_mode_position()

        result = self.mc_follower_to_wp1.update(
            current_pos=self.current_pos,
            current_vel=self.current_vel,
        )

        self.publish_position_setpoint(
            target=result.target,
            yaw=result.yaw,
        )

        if result.finished:
            self.get_logger().info("Arrived at WP1. Notify supervision.")

            self.publish_status("ARRIVED_WP1")
            # supervision이 이걸 받고 VERTIPORT_TRACKING으로 넘어감

            return

    def handle_mission_done(self):
        # 임무 완료 후 현재 위치를 유지하는 상태

        self.publish_offboard_mode_position()
        # position offboard mode heartbeat 전송

        hold = self.current_pos.copy()
        # 현재 위치를 hold target으로 설정

        self.publish_position_setpoint(hold)
        # 현재 위치 유지 setpoint publish


# ============================================================
# main
# ============================================================

def main(args=None):
    # ROS2 노드를 실행하는 main 함수

    rclpy.init(args=args)
    # ROS2 초기화

    node = VTOLMissionManager()
    # VTOLMissionManager 노드 생성

    try:
        rclpy.spin(node)
        # 노드를 계속 실행하면서 callback 처리
    except KeyboardInterrupt:
        pass
        # Ctrl+C로 종료하면 조용히 빠져나감

    node.destroy_node()
    # 노드 객체 제거

    rclpy.shutdown()
    # ROS2 종료


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 main 실행

    main()
    # main 함수 호출