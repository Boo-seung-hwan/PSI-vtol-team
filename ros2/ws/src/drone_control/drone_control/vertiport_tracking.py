import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleLocalPosition

@dataclass
class AxisPIDConfig:  # 한 축 PID 설정값
    kp: float = 0.35  # P gain, 현재 위치 오차에 비례한 속도 명령 생성, 클수록 목표를 향해 더 빠르게 움직임. v = kp*error
    kd: float = 0.12  # D gain, 현재 속도 기반 감쇠, 브레이크 역활
    output_limit: float = 0.8  # 해당 축 최대 속도 명령 제한
    derivative_tau: float = 0.20  # D항 low-pass filter 시간상수
    deadband: float = 0.05  # 작은 오차 무시 범위


@dataclass
class ControllerConfig:  # 전체 제어기 설정값
    x: AxisPIDConfig = field(default_factory=lambda: AxisPIDConfig(
        kp=0.35, kd=0.16, 
        output_limit=0.8, derivative_tau=0.20, deadband=0.05
    ))  # NED x축 제어 설정

    y: AxisPIDConfig = field(default_factory=lambda: AxisPIDConfig(
        kp=0.35, kd=0.16, 
        output_limit=0.8, derivative_tau=0.20, deadband=0.05
    ))  # NED y축 제어 설정

    z: AxisPIDConfig = field(default_factory=lambda: AxisPIDConfig(
        kp=0.22,  kd=0.10, 
        output_limit=0.35, derivative_tau=0.25, deadband=0.05
    ))  # NED z축 제어 설정

    yaw: AxisPIDConfig = field(default_factory=lambda: AxisPIDConfig(
        kp=0.60, kd=0.04, 
        output_limit=0.35, derivative_tau=0.20, deadband=0.03
    ))  # yaw 제어 설정

    control_period: float = 0.05  # 제어 주기, 0.05초 = 20Hz
    max_accel_xy: float = 0.60  # x/y 속도 명령 변화율 제한 [m/s^2], 1초에 0.03m/s씩 바뀔 수 있음.
    max_accel_z: float = 0.30  # z 속도 명령 변화율 제한 [m/s^2]

    filter_alpha: float = 0.35  # target NED exponential moving average 필터 계수, 새 좌표35% -> 너무 값이 튀지 않게 함.
    outlier_gate_m: float = 1.50  # 이전 filtered target에서 이 거리 이상 튀면 outlier로 판단
    max_consecutive_outliers: int = 5  # outlier가 연속으로 이 횟수 이상이면 target 재초기화 허용 -> 실제 타켓이 바뀐 것일수도 있음. 
    max_target_distance_m: float = 15.0


    short_loss_s: float = 0.70  # valid target이 끊긴 지 이 시간 이하이면 짧은 손실로 처리 -> 한두 프레임 놓치는건 괜찮음.
    long_loss_s: float = 3.00  # valid target이 이 시간 이상 끊기면 failsafe 요청 상태

    align_xy_threshold_m: float = 0.15  # x-y 평면 정렬 완료 기준 [m] 목표와의 거리가 이 보다 작으면 수평 정렬 완료
    align_z_threshold_m: float = 0.25  # z축 정렬 완료 기준 [m]
    align_yaw_threshold_rad: float = math.radians(8.0)  # yaw 정렬 완료 기준 [rad]
    stable_time_s: float = 1.00  # 정렬 상태가 연속으로 유지되어야 하는 시간 [s]


    rescue_ascent_speed_ned: float = 0.30  # 구조 완료 후 상승 속도 [m/s]
    rescue_return_z_tolerance_m: float = 0.20  # 원래 높이에 도달했다고 판단하는 z 오차 [m]

    rescue_hover_height_m: float = 2.0  # 조난자 위에서 유지할 hover 높이 [m]
    rescue_hover_z_tolerance_m: float = 0.20  # hover 높이 오차 허용 범위 [m]
    landing_xy_threshold_m: float = 0.18  # 착륙 하강 중 유지해야 하는 xy 오차 기준 [m]
    landing_descent_speed_max_ned: float = 0.25  # 목표 지점에서 멀 때 최대 하강 속도 [m/s]
    landing_descent_speed_min_ned: float = 0.05  # 목표 지점에 가까울 때 최소 하강 속도 [m/s]
    landing_slow_zone_m: float = 1.50  # 목표 z로부터 이 거리 안에 들어오면 감속 시작 [m]
    landing_finish_z_error_m: float = 0.15  # 목표 z에 이 정도 가까워지면 착륙 목표 도달로 판단 [m]
    active_mission_states: tuple[str, str] = ("RESCUE_APPROACH", "VERTIPORT_LANDING")  # 이 상태에서만 정밀제어 활성화

def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class AxisPID:  # 한 축의 위치 오차를 속도 명령으로 바꾸는 PID 제어기
    def __init__(self, cfg: AxisPIDConfig, name: str):  # 설정값과 이름을 받아 초기화
        self.cfg = cfg  # 이 축의 PID 설정 저장
        self.name = name  # 디버깅용 이름 저장
        self.prev_error: Optional[float] = None  # 이전 오차 초기화
        self.filtered_derivative = 0.0  # 필터링된 D항 초기화

    def reset(self):  # PID 내부 상태 초기화 -> target을 놓쳤거나 dt가 갑자기 튈때, mission state가 바뀌었을때 리셋
        self.prev_error = None  # 이전 오차 기록 삭제
        self.filtered_derivative = 0.0  # D항 필터 리셋

    def update(self, error: float, dt: float, measurement_rate: Optional[float] = None) -> float:  # PID 출력 계산
        if dt <= 1e-6:  # dt가 비정상적으로 작으면 -> D항 계산에 문제가 생길 수 있음
            return 0.0  # 안전하게 0 반환 

        error = 0.0 if abs(error) < self.cfg.deadband else float(error)  # deadband 이내 오차, 즉 작은 오차는 그냥 0으로 처리 -> 불필요한 흔들림 줄여줌.

        if measurement_rate is not None:  # 현재 속도를 알고 있으면
            derivative_raw = -float(measurement_rate)  # error_dot = -current_velocity로 D항 계산
        elif self.prev_error is None:  # 이전 오차가 없으면
            derivative_raw = 0.0  # 첫 계산에서 D항은 0
        else:  # 이전 오차가 있으면
            derivative_raw = (error - self.prev_error) / dt  # 오차 변화율 계산

        alpha = dt / (self.cfg.derivative_tau + dt)  # D항 필터 계수 계산, tau작음-> D항이 빠르게 반응하지만 노이즈에 민감(카메라 좌표가 흔들리는 것 보정)
        self.filtered_derivative = alpha * derivative_raw + (1.0 - alpha) * self.filtered_derivative  # D항 low-pass filtering -> 비전 좌표가 튀면 D항도 크게 될 수 있는데, 그 튐을 줄여주는 필터

        p_term = self.cfg.kp * error  # P항 계산 -> 속도 명령
        d_term = self.cfg.kd * self.filtered_derivative  # D항 계산 -> 오차 변화율을 보고 명령을 줄이거나 보정

        u = p_term + d_term
        u = float(np.clip(u, -self.cfg.output_limit, self.cfg.output_limit))

        self.prev_error = error
        return u

class SlewRateLimiter:  # 속도 명령 변화율 제한
    def __init__(self, max_delta_per_sec: np.ndarray):  # 축별 최대 변화율을 받아 초기화
        self.max_delta_per_sec = np.array(max_delta_per_sec, dtype=float)  # 최대 변화율
        self.last: Optional[np.ndarray] = None  # 이전 출력 저장 변수

    def reset(self):  # 내부 상태 초기화
        self.last = None  # 이전 출력 기록 삭제

    def update(self, command: np.ndarray, dt: float) -> np.ndarray:  # 변화율 제한 적용
        command = np.array(command, dtype=float)  # 속도 명령을 numpy 배열로 변환

        if self.last is None:  # 이전 출력이 없으면
            self.last = command.copy()  # 현재 명령을 이전 출력으로 저장
            return command  # 첫 출력은 그대로 반환

        max_delta = self.max_delta_per_sec * max(dt, 1e-6)  # 이번 주기에 허용되는 최대 변화량
        delta = np.clip(command - self.last, -max_delta, max_delta)  # 명령 변화량 제한
        self.last = self.last + delta  # 제한된 변화량만 반영
        return self.last.copy()  # 제한된 명령 반환

class CascadePositionToVelocityController:  # 위치 오차를 속도 명령으로 바꾸는 cascade outer-loop 제어기
    def __init__(self, cfg: ControllerConfig):  # 설정값을 받아 초기화
        self.cfg = cfg  # 설정값 저장
        self.pid_x = AxisPID(cfg.x, "x")  # x축 PD 생성
        self.pid_y = AxisPID(cfg.y, "y")  # y축 PD 생성
        self.pid_z = AxisPID(cfg.z, "z")  # z축 PD 생성
        self.pid_yaw = AxisPID(cfg.yaw, "yaw")  # yaw PD 생성
        self.limiter = SlewRateLimiter(np.array([cfg.max_accel_xy, cfg.max_accel_xy, cfg.max_accel_z]))  # 속도 변화율 제한기 생성

    def reset(self):  # 제어기 전체 리셋
        self.pid_x.reset()  # x축 PID 리셋
        self.pid_y.reset()  # y축 PID 리셋
        self.pid_z.reset()  # z축 PID 리셋
        self.pid_yaw.reset()  # yaw PID 리셋
        self.limiter.reset()  # slew limiter 리셋

    def compute(self, pos_error: np.ndarray, current_vel: np.ndarray, yaw_error: float, dt: float) -> tuple[np.ndarray, float]:  # 속도 명령 계산
        vx = self.pid_x.update(pos_error[0], dt, measurement_rate=current_vel[0])  # x축 속도 명령 계산
        vy = self.pid_y.update(pos_error[1], dt, measurement_rate=current_vel[1])  # y축 속도 명령 계산
        vz = self.pid_z.update(pos_error[2], dt, measurement_rate=current_vel[2])  # z축 속도 명령 계산
        yaw_rate = self.pid_yaw.update(yaw_error, dt, measurement_rate=None)  # yaw rate 명령 계산
        vel = np.array([vx, vy, vz], dtype=float)
        vel = self.limit_xy_speed(vel, max_xy=0.6)
        vel = self.limiter.update(vel, dt)
        return vel, yaw_rate  # 최종 속도 명령과 yaw rate 반환

    def limit_xy_speed(self, vel: np.ndarray, max_xy: float) -> np.ndarray:
        xy_norm = float(np.linalg.norm(vel[:2]))
        if xy_norm > max_xy and xy_norm > 1e-6:
            vel[:2] *= max_xy / xy_norm
        return vel


class TargetFilter:  # NED target filtering 및 outlier rejection 담당 클래스 (순간적으로 흔들리거나 튀는 것을 막아주는 클래스)
    def __init__(self, cfg: ControllerConfig):  # 설정값을 받아 초기화
        self.cfg = cfg  # 설정값 저장
        self.filtered: Optional[np.ndarray] = None  # 필터링된 target 저장
        self.last_raw: Optional[np.ndarray] = None  # 마지막 raw target 저장
        self.consecutive_outliers = 0  # 연속 outlier 횟수

    def reset(self):  # 필터 초기화
        self.filtered = None  # filtered target 삭제
        self.last_raw = None  # raw target 삭제
        self.consecutive_outliers = 0  # outlier count 초기화

    def update(self, raw_target: np.ndarray) -> tuple[Optional[np.ndarray], bool]:  # raw target을 받아 filtered target과 outlier 여부 반환
        raw_target = np.array(raw_target, dtype=float)  # raw target을 numpy 배열로 변환

        if not np.all(np.isfinite(raw_target)):  # NaN이나 inf가 있으면 이상한 값이 있는지 확인
            return self.filtered, True  # 현재 filtered 유지, outlier로 처리

        if self.filtered is None:  # 첫 target이면
            self.filtered = raw_target.copy()  # 그대로 초기화
            self.last_raw = raw_target.copy()  # raw 저장
            self.consecutive_outliers = 0  # outlier count 초기화
            return self.filtered.copy(), False  # 정상 target 반환

        jump = float(np.linalg.norm(raw_target - self.filtered))  # raw target과 filtered target 사이 거리 계산

        if jump > self.cfg.outlier_gate_m:  # 갑자기 너무 멀리 튀었으면
            self.consecutive_outliers += 1  # outlier count 증가

            if self.consecutive_outliers < self.cfg.max_consecutive_outliers:  # outlier가 아직 연속으로 충분히 많이 발생하지 않았다면, 일시적 outlier로 판단하고 무시
                return self.filtered.copy(), True  # 이전 filtered target 유지

            self.filtered = raw_target.copy()  # outlier가 계속되면 실제 target 변경으로 보고 재초기화
            self.last_raw = raw_target.copy()  # raw 저장
            self.consecutive_outliers = 0  # outlier count 초기화
            return self.filtered.copy(), False  # 새 target 정상 처리, raw target을 새 기준으로 받아들임.

        self.consecutive_outliers = 0  # 정상 target이면 outlier count 초기화
        self.filtered = self.cfg.filter_alpha * raw_target + (1.0 - self.cfg.filter_alpha) * self.filtered  # EMA 필터 적용
        self.last_raw = raw_target.copy()  # raw target 저장
        return self.filtered.copy(), False  # filtered target 반환

class PX4PrecisionMissionNode(Node):  # PX4 Offboard가 이미 유지되는 상황에서 정밀제어만 담당하는 노드
    def __init__(self):  # 노드 초기화
        super().__init__("vertiport_tracking")  # ROS2 노드 이름 설정

        self.cfg = ControllerConfig()  # 제어 설정 생성
        self.controller = CascadePositionToVelocityController(self.cfg)  # cascade PID 제어기 생성
        self.target_filter = TargetFilter(self.cfg)  # target filter 생성

        self.current_pos = np.zeros(3)  # PX4 현재 NED 위치 저장
        self.current_vel = np.zeros(3)  # PX4 현재 NED 속도 저장
        self.current_yaw = 0.0  # PX4 현재 heading 저장
        self.local_ready = False  # 현재 위치 수신 여부

        self.raw_target_ned = np.zeros(3)  # raw target NED 절대좌표 저장
        self.filtered_target_ned: Optional[np.ndarray] = None  # filtered target NED 저장
        self.target_yaw = math.nan  # target yaw 저장
        self.target_valid_msg = False  # 메시지에서 valid flag가 1인지 저장
        self.last_msg_stamp = 0.0  # 마지막 target 메시지 수신 시간, 메세지 들어왔는지 확인
        self.last_valid_stamp = 0.0  # 마지막 valid target 수신 시간

        self.mission_state = "INACTIVE"  # 현재 mission state 저장
        self.internal_phase = "IDLE"  # 정밀제어 내부 단계 저장
        self.stable_count = 0  # 연속 정렬 성공 count
        self.stable_required_count = max(1, int(math.ceil(self.cfg.stable_time_s / self.cfg.control_period)))  # stable 상태로 인정하기 위해 필요한 count를 계산

        self.rescue_done = False  # 구조 완료 신호 수신 여부
        self.rescue_return_z: Optional[float] = None  # 구조 후 돌아갈 원래 높이 NED z

        self.last_time = self.get_clock().now().nanoseconds / 1e9  # dt 계산용 마지막 시간 저장

        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )  # PX4 out topic 구독에 맞는 QoS 설정


        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.local_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_position_callback,
            qos_sub
        )  # PX4 현재 위치/속도/yaw subscriber

        self.target_sub = self.create_subscription(
            Float32MultiArray,
            "/target/ned",
            self.target_callback,
            10
        )  # NED 절대좌표 target subscriber

        self.mission_sub = self.create_subscription(
            String,
            "/mission/state",
            self.mission_callback,
            10
        )  # mission state subscriber


        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            "/vertiport_tracking/trajectory_setpoint",
            qos_pub
        )  # PX4 velocity setpoint publisher

        self.status_pub = self.create_publisher(
            String,
            "/vertiport_tracking/status",
            10
        )  # precision controller 상태 publisher

        self.timer = self.create_timer(
            self.cfg.control_period,
            self.control_loop
        )  # 0.05초마다 control loop 생성

        self.rescue_status_sub = self.create_subscription(
            String,
            "/mission/rescue_status",
            self.rescue_status_callback,
            10
        )


        self.get_logger().info("PX4 precision mission controller started.")  # 시작 로그
        self.get_logger().info("This node assumes OFFBOARD is already maintained by the mission manager.")  # Offboard 유지 가정 로그
        self.get_logger().info("/target/ned format: [x_ned, y_ned, z_ned, yaw_rad, valid].")  # target format 로그

    def now_us(self) -> int:  # PX4 timestamp용 현재 시간 반환
        return int(self.get_clock().now().nanoseconds / 1000)  # nanoseconds를 microseconds로 변환(PX4로 보내는 메시지에 현재 시간을 찍어주는 역할)

    def publish_debug_status(
        self,
        label: str,
        pos_error: Optional[np.ndarray] = None,
        yaw_error: float = math.nan,
        vel_cmd: Optional[np.ndarray] = None,
        ):
        loss_s = self.target_loss_duration()

        if pos_error is not None:
            xy_error = float(np.linalg.norm(pos_error[:2]))
            z_error = float(abs(pos_error[2]))
        else:
            xy_error = math.nan
            z_error = math.nan

        if vel_cmd is None:
            vel_cmd = np.array([math.nan, math.nan, math.nan], dtype=float)

        yaw_deg = math.degrees(yaw_error) if math.isfinite(yaw_error) else math.nan

        if self.filtered_target_ned is not None:
            target_text = (
                f"target=({self.filtered_target_ned[0]:.2f},"
                f"{self.filtered_target_ned[1]:.2f},"
                f"{self.filtered_target_ned[2]:.2f})"
            )
        else:
            target_text = "target=None"

        text = (
            f"{label} | "
            f"state={self.mission_state}, "
            f"phase={self.internal_phase}, "
            f"xy={xy_error:.3f}, "
            f"z={z_error:.3f}, "
            f"yaw_deg={yaw_deg:.1f}, "
            f"stable={self.stable_count}/{self.stable_required_count}, "
            f"valid={self.target_valid_msg}, "
            f"loss={loss_s:.2f}, "
            f"pos=({self.current_pos[0]:.2f},"
            f"{self.current_pos[1]:.2f},"
            f"{self.current_pos[2]:.2f}), "
            f"vel=({vel_cmd[0]:.2f},"
            f"{vel_cmd[1]:.2f},"
            f"{vel_cmd[2]:.2f}), "
            f"{target_text}"
        )
        
        msg =String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_status(self, text: str):
        # MissionManager가 읽을 짧은 상태 메시지 publish

        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def local_position_callback(self, msg: VehicleLocalPosition):  # PX4 현재 위치 수신 callback
        self.current_pos = np.array([msg.x, msg.y, msg.z], dtype=float)  # 현재 NED 위치 저장
        self.current_vel = np.array([msg.vx, msg.vy, msg.vz], dtype=float)  # 현재 NED 속도 저장
        self.current_yaw = float(msg.heading)  # 현재 yaw 저장
        self.local_ready = True  # 위치 수신 완료 표시

    def mission_callback(self, msg: String):  # mission state 수신 callback
        new_state = msg.data.strip()  # 문자열 앞뒤 공백 제거

        if new_state != self.mission_state:  # mission state가 기존 state와 다를떼 실행
            self.get_logger().info(f"Mission state changed: {self.mission_state} -> {new_state}")  # 상태 변경 로그
            self.mission_state = new_state  # 새 mission state 저장
            self.controller.reset()  # 이전 상태 PID 메모리 제거
            self.target_filter.reset()  # 이전 target 필터 제거
            self.stable_count = 0  # stable counter 초기화
            self.internal_phase = "ALIGN" if new_state in self.cfg.active_mission_states else "IDLE"  # 활성 상태면 ALIGN부터 시작
            
            self.rescue_done = False

            if new_state == "RESCUE_APPROACH":
                self.rescue_return_z = self.current_pos[2]
            else:
                self.rescue_return_z = None

    def target_callback(self, msg: Float32MultiArray):  # NED target 수신 callback
        data = list(msg.data)  # 배열 데이터를 list로 변환
        self.last_msg_stamp = time.monotonic()  # 마지막 메시지 수신 시간 저장

        if len(data) < 3:  # x, y, z가 없으면
            self.target_valid_msg = False  # invalid 처리
            return  # callback 종료

        valid = bool(data[4] > 0.5) if len(data) >= 5 else True  # valid flag가 있으면 사용, 없으면 True로 가정
        self.target_valid_msg = valid  # 메시지 valid 상태 저장

        if not valid:  # valid가 0이면
            return  # target filter 업데이트하지 않음

        raw = np.array([data[0], data[1], data[2]], dtype=float)  # NED 절대좌표 target 저장
        yaw = float(data[3]) if len(data) >= 4 and math.isfinite(float(data[3])) else math.nan  # yaw가 있으면 저장

        if self.local_ready:
            target_distance = float(np.linalg.norm(raw - self.current_pos))

            if target_distance > self.cfg.max_target_distance_m:
                self.publish_debug_status("TARGET_REJECTED_TOO_FAR")
                return

        filtered, outlier = self.target_filter.update(raw)  # target filter 및 outlier rejection 적용

        if outlier:  # outlier로 판단되면
            self.publish_debug_status(
                "TARGET_OUTLIER_REJECTED")  # outlier 상태 publish
            return  # outlier는 valid target으로 갱신하지 않음

        self.raw_target_ned = raw.copy()  # raw target 저장
        self.filtered_target_ned = filtered.copy() if filtered is not None else None  # filtered target 저장
        self.target_yaw = yaw  # target yaw 저장
        self.last_valid_stamp = time.monotonic()  # 마지막 valid target 시간 갱신

    def rescue_status_callback(self, msg: String):
        status = msg.data.strip()

        if status == "RESCUE_DONE":
            self.rescue_done = True
            self.publish_debug_status("RESCUE_DONE_SIGNAL_RECEIVED")

    def is_active_state(self) -> bool:  # 현재 mission state가 이 노드의 활성 대상인지 판단
        return self.mission_state in self.cfg.active_mission_states  # RESCUE_APPROACH 또는 VERTIPORT_LANDING인지 반환

    def publish_offboard_control_mode(self):  # PX4 OffboardControlMode publish   -> 위치 명령이 아니라 속도 명령을 사용할 것임
        msg = OffboardControlMode()  # 메시지 생성
        msg.timestamp = self.now_us()  # timestamp 설정
        msg.position = False  # position setpoint는 사용하지 않음
        msg.velocity = True  # velocity setpoint 사용
        msg.acceleration = False  # acceleration setpoint 미사용
        msg.attitude = False  # attitude setpoint 미사용
        msg.body_rate = False  # body rate setpoint 미사용   

        if hasattr(msg, "thrust_and_torque"):  # px4_msgs 버전별 필드 확인
            msg.thrust_and_torque = False  # thrust/torque 직접 제어 미사용

        if hasattr(msg, "direct_actuator"):  # px4_msgs 버전별 필드 확인
            msg.direct_actuator = False  # actuator 직접 제어 미사용

        self.offboard_pub.publish(msg)  # OffboardControlMode publish

    def publish_velocity_setpoint(self, vel: np.ndarray, yaw_sp: float, yaw_rate: float):  # PX4 velocity setpoint publish
        msg = TrajectorySetpoint()  # TrajectorySetpoint 메시지 생성
        msg.timestamp = self.now_us()  # timestamp 설정
        msg.position = [math.nan, math.nan, math.nan]  # velocity 제어이므로 position은 NaN
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]  # NED velocity setpoint 설정
        msg.acceleration = [math.nan, math.nan, math.nan]  # acceleration 미사용
        msg.yaw = float(yaw_sp) if math.isfinite(yaw_sp) else math.nan  # yaw setpoint 설정
        msg.yawspeed = float(yaw_rate)  # yaw rate 설정
        self.traj_pub.publish(msg)  # PX4로 publish

    def publish_zero_velocity(self, status: str, reset_controller: bool = False):
        if reset_controller:
            self.controller.reset()
        else:
            self.controller.limiter.last = np.zeros(3)

        self.publish_velocity_setpoint(np.zeros(3), self.current_yaw, 0.0)
        self.publish_debug_status(status)

    def compute_dt(self) -> float:  # dt 계산 함수
        now = self.get_clock().now().nanoseconds / 1e9  # 현재 ROS 시간을 초 단위로 계산
        dt = now - self.last_time  # 이전 loop와 시간 차 계산
        self.last_time = now  # 마지막 시간 갱신

        if dt <= 0.0 or dt > 0.5:  # dt가 이상하면(시간을 역행했거나 계산이 이상함 or 제어 loop가 너무 오래 멈췄거나 지연됨)
            dt = self.cfg.control_period  # 기본 제어 주기 사용
            self.controller.reset()  # PID 튐 방지를 위해 리셋

        return dt  # dt 반환

    def target_loss_duration(self) -> float:  # 마지막 valid target 이후 경과 시간 계산
        if self.last_valid_stamp <= 0.0:  # 아직 valid target이 한 번도 없으면
            return float("inf")  # 무한대로 처리
        return time.monotonic() - self.last_valid_stamp  # 마지막 valid target 이후 시간 반환

    def handle_invalid_or_lost_target(self) -> bool:  # valid=0 또는 target lost 대응
        loss_s = self.target_loss_duration()  # target loss duration 계산

        if self.filtered_target_ned is not None and loss_s <= self.cfg.short_loss_s:  # 짧은 손실이면
            self.publish_zero_velocity("SHORT_TARGET_LOSS_HOVER")  # zero velocity로 hover 유지
            return True  # 이번 loop 처리 완료

        if loss_s <= self.cfg.long_loss_s:  # 중간 길이 손실이면
            self.publish_zero_velocity("TARGET_LOSS_HOLD_REQUEST")  # hold 요청 상태로 zero velocity
            return True  # 이번 loop 처리 완료

        self.publish_zero_velocity("FAILSAFE_TARGET_LOST_LONG_REQUEST_MISSION_MANAGER")  # 긴 손실이면 failsafe 요청
        return True  # 이번 loop 처리 완료

    def compute_landing_descent_speed(self, z_error: float) -> float:  # 목표 z에 가까워질 수록 하강 속도를 줄이는 함수

        z_error = abs(float(z_error))

        if z_error <= self.cfg.landing_finish_z_error_m:    # 목표 z에 충분히 가까우면 더 이상 하강하지 않음
            return 0.0

        if z_error >= self.cfg.landing_slow_zone_m:  # 감속 구간 밖, 즉 아직 지면에서 멀면 최대 하강 속도 사용
            return self.cfg.landing_descent_speed_max_ned

        
        ratio = (    # 감속 구간 안에서는 z_error가 작아질수록 속도를 선형적으로 줄임
            (z_error - self.cfg.landing_finish_z_error_m)
            / (self.cfg.landing_slow_zone_m - self.cfg.landing_finish_z_error_m)
        )

        ratio = float(np.clip(ratio, 0.0, 1.0))

        descent_speed = (
            self.cfg.landing_descent_speed_min_ned
            + ratio * (
                self.cfg.landing_descent_speed_max_ned
                - self.cfg.landing_descent_speed_min_ned
            )
        )

        return float(descent_speed)


    def check_stable(self, pos_error: np.ndarray, yaw_error: float) -> bool:  # 정렬 안정 상태 판단
        xy_error = float(np.linalg.norm(pos_error[:2]))  # xy 평면 오차 계산
        z_error = float(abs(pos_error[2]))  # z축 오차 계산
        yaw_ok = True if not math.isfinite(self.target_yaw) else abs(yaw_error) < self.cfg.align_yaw_threshold_rad  # yaw 목표가 있으면 yaw 기준 적용

        if self.mission_state == "RESCUE_APPROACH":  # 조난자 구조 접근이면
            stable_now = (
                xy_error < self.cfg.align_xy_threshold_m and
                z_error < self.cfg.align_z_threshold_m and
                yaw_ok
            )  # xy, z, yaw 모두 만족해야 안정

        elif self.mission_state == "VERTIPORT_LANDING":  # 버티포트 착륙이면
            stable_now = (
                xy_error < self.cfg.align_xy_threshold_m and
                yaw_ok
            )  # 착륙 정렬에서는 xy와 yaw 중심으로 안정 판단

        else:  # 다른 상태면
            stable_now = False  # 안정 상태 아님

        if stable_now:  # 이번 loop에서 안정 조건을 만족하면
            self.stable_count += 1  # stable counter 증가
        else:  # 안정 조건을 만족하지 못하면
            self.stable_count = 0  # stable counter 초기화

        return self.stable_count >= self.stable_required_count  # 연속 유지 시간이 충족되었는지 반환

    
    

    def apply_mission_specific_logic(
        self,
        vel_cmd: np.ndarray,
        pos_error: np.ndarray,
        yaw_error: float,
        stable: bool
    ) -> np.ndarray:  # mission별 추가 제어 로직 적용

        vel_cmd = vel_cmd.copy()  # 원본 속도 명령을 복사

        if self.mission_state == "RESCUE_APPROACH":


            # RESCUE_DONE을 받으면 stable 조건과 무관하게 복귀 상승 시작
            if self.internal_phase in ["IDLE", "ALIGN", "RESCUE_ALIGNING", "RESCUE_STABLE_READY"]:
                if self.rescue_done:
                    vel_cmd[:] = 0.0
                    self.internal_phase = "RESCUE_ASCENDING"
                    self.controller.reset()
                    self.publish_debug_status(
                        "RESCUE_DONE_ASCEND_START",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )
                    return vel_cmd
        
            # 1. 아직 상승 단계가 아니면, 조난자 위 hover 지점으로 정렬
            if self.internal_phase in ["IDLE", "ALIGN", "RESCUE_ALIGNING", "RESCUE_STABLE_READY"]:
                if self.rescue_done and self.internal_phase not in ["RESCUE_ASCENDING", "RESCUE_DONE"]:
                    vel_cmd[:] = 0.0
                    self.internal_phase = "RESCUE_ASCENDING"
                    self.controller.reset()
                    self.publish_debug_status(
                    "RESCUE_DONE_ASCEND_START",
                    pos_error,
                    yaw_error,
                    vel_cmd
                    )
                    return vel_cmd

                if stable:
                    vel_cmd[:] = 0.0
                    self.internal_phase = "RESCUE_STABLE_READY"

                    # 구조 완료 신호를 받으면 상승 단계로 전환
                    if self.rescue_done:
                        self.internal_phase = "RESCUE_ASCENDING"
                        self.controller.reset()
                        self.publish_debug_status(
                            "RESCUE_DONE_ASCEND_START",
                            pos_error,
                            yaw_error,
                            vel_cmd
                        )
                    else:
                        self.publish_debug_status(
                            "RESCUE_STABLE_READY_WAITING_FOR_DONE",
                            pos_error,
                            yaw_error,
                            vel_cmd
                        )

                    return vel_cmd

                else:
                    self.internal_phase = "RESCUE_ALIGNING"
                    self.publish_debug_status(
                        "RESCUE_ALIGNING",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )
                    return vel_cmd

            # 2. 구조 완료 후 원래 높이까지 상승
            if self.internal_phase == "RESCUE_ASCENDING":

                # 원래 높이를 저장하지 못했다면 안전하게 현재 위치에서 hover
                if self.rescue_return_z is None:
                    vel_cmd[:] = 0.0
                    self.publish_debug_status(
                        "RESCUE_ASCEND_NO_RETURN_ALT_HOVER",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )
                    return vel_cmd

                z_to_return = self.rescue_return_z - self.current_pos[2]

                # 원래 높이에 충분히 가까워지면 상승 종료
                if abs(z_to_return) < self.cfg.rescue_return_z_tolerance_m:
                    vel_cmd[:] = 0.0
                    self.internal_phase = "RESCUE_DONE"
                    self.publish_status("RESCUE_RETURN_ALT_REACHED")
                    self.publish_debug_status(
                        "RESCUE_RETURN_ALT_REACHED",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )
                    return vel_cmd

                z_to_return = self.rescue_return_z - self.current_pos[2]

                vz_return = 0.4 * z_to_return
                vz_return = float(np.clip(
                    vz_return,
                    -self.cfg.rescue_ascent_speed_ned,
                    self.cfg.rescue_ascent_speed_ned
                ))

                vel_cmd[0] = 0.0
                vel_cmd[1] = 0.0
                vel_cmd[2] = vz_return
            
                self.publish_debug_status(
                    "RESCUE_ASCENDING_TO_RETURN_ALT",
                    pos_error,
                    yaw_error,
                    vel_cmd
                )
                return vel_cmd

            # 3. 상승 완료 후 정지 유지
            if self.internal_phase == "RESCUE_DONE":
                vel_cmd[:] = 0.0
                self.publish_status("RESCUE_DONE_HOLD")
                # supervision이 놓쳤을 경우를 대비해 완료 상태를 계속 알림

                self.publish_debug_status(
                    "RESCUE_DONE_HOLD",
                    pos_error,
                    yaw_error,
                    vel_cmd
                )
                return vel_cmd

        if self.mission_state == "VERTIPORT_LANDING":  # 버티포트 착륙 상태라면
            xy_error = float(np.linalg.norm(pos_error[:2]))  # xy 오차 계산
            z_error = float(abs(pos_error[2]))  # z 오차 계산

            if self.internal_phase in ["IDLE", "ALIGN"]:
                self.internal_phase = "ALIGN"

                if stable:
                    self.internal_phase = "DESCEND"
                    self.stable_count = 0
                    self.publish_debug_status(
                        "VERTIPORT_ALIGN_COMPLETE_DESCEND_START",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )
                else:
                    self.publish_debug_status(
                        "VERTIPORT_ALIGNING",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )

                vel_cmd[2] = 0.0
                return vel_cmd

            if self.internal_phase == "DESCEND":  # 하강 단계라면
                if xy_error > self.cfg.landing_xy_threshold_m:  # 하강 중 xy 오차가 커지면
                    vel_cmd[2] = 0.0  # 하강 정지
                    self.internal_phase = "ALIGN"  # 다시 정렬 단계로 복귀
                    self.stable_count = 0  # stable counter 초기화
                    self.publish_debug_status("VERTIPORT_XY_ERROR_TOO_LARGE_REALIGN",
                    pos_error,
                    yaw_error,
                    vel_cmd
                    )  # 상태 publish
                    return vel_cmd  # 재정렬 명령 반환

                if z_error < self.cfg.landing_finish_z_error_m:  # 목표 z에 충분히 가까우면
                    vel_cmd[:] = 0.0  # 정지
                    self.internal_phase = "LANDING_TARGET_REACHED"  # 착륙 목표 도달 상태
                    self.publish_status("VERTIPORT_LANDING_TARGET_REACHED")
                    # MissionManager가 읽는 착륙 완료 신호

                    self.publish_debug_status(
                        "VERTIPORT_LANDING_TARGET_REACHED",
                        pos_error,
                        yaw_error,
                        vel_cmd
                    )  # 상태 publish
                    return vel_cmd  # 정지 명령 반환

                dynamic_descent_speed = self.compute_landing_descent_speed(z_error)

                vel_cmd[2] = min(
                    max(vel_cmd[2], 0.0),
                    dynamic_descent_speed
                )

                self.publish_debug_status(
                    f"VERTIPORT_DESCENDING_DYNAMIC_SPEED_{dynamic_descent_speed:.2f}",
                    pos_error,
                    yaw_error,
                    vel_cmd
                )

                return vel_cmd

            if self.internal_phase == "LANDING_TARGET_REACHED":  # 착륙 목표 도달 이후라면
                vel_cmd[:] = 0.0  # 계속 정지
                self.publish_status("VERTIPORT_HOLD_AFTER_REACHED")
                self.publish_debug_status("VERTIPORT_HOLD_AFTER_REACHED",
                pos_error,
                yaw_error,
                vel_cmd
                )  # 도달 후 hold 상태 publish
                return vel_cmd  # 정지 명령 반환

        return vel_cmd  # 다른 경우에는 원래 명령 그대로 반환

    def control_loop(self):  # 20Hz 메인 제어 루프
        dt = self.compute_dt()  # dt 계산

        if not self.is_active_state():  # 현재 mission state가 이 노드의 담당 상태가 아니면
            self.controller.reset()  # PID 리셋
            self.target_filter.reset()  # target filter 리셋
            self.stable_count = 0  # stable counter 리셋
            self.internal_phase = "IDLE"  # 내부 상태 IDLE
            self.publish_debug_status(f"INACTIVE_MISSION_STATE_{self.mission_state}")  # inactive 상태 publish
            return  # 이 노드는 setpoint를 보내지 않고 종료


        if not self.local_ready:  # PX4 현재 위치를 아직 못 받았으면
            self.publish_zero_velocity("NO_LOCAL_POSITION_HOVER")  # zero velocity publish
            return  # 제어 종료

        # ============================================================
        # RESCUE_DONE 수신 후에는 target/ned가 끊겨도 복귀 상승 수행
        # ============================================================
        if self.mission_state == "RESCUE_APPROACH" and self.rescue_done:
            if self.internal_phase not in ["RESCUE_ASCENDING", "RESCUE_DONE"]:
                self.internal_phase = "RESCUE_ASCENDING"
                self.controller.reset()
                self.stable_count = 0
                self.publish_debug_status("RESCUE_DONE_ASCEND_START")

            if self.rescue_return_z is None:
                self.publish_zero_velocity("RESCUE_ASCEND_NO_RETURN_ALT_HOVER")
                return
 
            z_to_return = self.rescue_return_z - self.current_pos[2]

            if abs(z_to_return) < self.cfg.rescue_return_z_tolerance_m:
                vel_cmd = np.zeros(3, dtype=float)
                self.internal_phase = "RESCUE_DONE"

                # mission_manager가 이 문자열을 보고 GENERATOR_RETURN으로 넘어가야 함
                self.publish_status("RESCUE_RETURN_ALT_REACHED")

                self.publish_velocity_setpoint(
                    vel_cmd,
                    self.current_yaw,
                    0.0
                )

                # 주의: mission_manager가 exact match를 쓰면
                # 아래 debug가 완료 status를 덮어쓸 수 있음.
                # mission_manager를 startswith 방식으로 고치지 않았다면 이 줄은 빼는 게 안전함.
                # self.publish_debug_status("RESCUE_RETURN_ALT_REACHED", vel_cmd=vel_cmd)

                return

            vz_return = 0.4 * z_to_return
            vz_return = float(np.clip(
                vz_return,
                -self.cfg.rescue_ascent_speed_ned,
                self.cfg.rescue_ascent_speed_ned
            ))

            vel_cmd = np.zeros(3, dtype=float)
            vel_cmd[2] = vz_return

            self.publish_velocity_setpoint(
                vel_cmd,
                self.current_yaw,
                0.0
            )

            self.publish_debug_status(
                "RESCUE_ASCENDING_TO_RETURN_ALT",
                vel_cmd=vel_cmd
            )

            return

        # ============================================================
        # 여기부터는 기존 target 기반 정밀제어 로직
        # ============================================================

        if self.filtered_target_ned is None or not self.target_valid_msg:  # filtered target이 없거나 valid=0이면
            self.handle_invalid_or_lost_target()  # valid=0 정책 적용
            return  # 제어 종료

        loss_s = self.target_loss_duration()  # 마지막 valid target 이후 시간 계산

        if loss_s > self.cfg.short_loss_s:  # valid target이 오래되었으면
            self.handle_invalid_or_lost_target()  # target lost 정책 적용
            return  # 제어 종료

        target_pos = self.filtered_target_ned.copy()  # 무조건 NED 절대좌표 target 사용
        if self.mission_state == "RESCUE_APPROACH":
            target_pos[2] = self.filtered_target_ned[2] - self.cfg.rescue_hover_height_m

        pos_error = target_pos - self.current_pos

        yaw_sp = self.target_yaw if math.isfinite(self.target_yaw) else self.current_yaw  # target yaw가 없으면 현재 yaw 유지
        yaw_error = wrap_pi(yaw_sp - self.current_yaw)  # yaw 오차 계산

        stable = self.check_stable(pos_error, yaw_error)  # 연속 정렬 안정 여부 계산

        vel_cmd, yaw_rate_cmd = self.controller.compute(
            pos_error,
            self.current_vel,
            yaw_error,
            dt
        )  # Cascade PID로 velocity setpoint 계산

        vel_cmd = self.apply_mission_specific_logic(
            vel_cmd,
            pos_error,
            yaw_error,
            stable
        )  # 조난자/버티포트 mission별 추가 로직 적용

        self.publish_velocity_setpoint(
            vel_cmd,
            yaw_sp,
            yaw_rate_cmd
        )  # PX4로 최종 velocity setpoint publish

def main(args=None):  # ROS2 실행 진입점
    rclpy.init(args=args)  # ROS2 초기화
    node = PX4PrecisionMissionNode()  # 노드 생성

    try:  # 안전 종료 처리
        rclpy.spin(node)  # 노드 실행 유지
    except KeyboardInterrupt:  # Ctrl+C 종료 시
        pass  # 예외 무시
    finally:  # 종료 시 반드시 실행
        node.destroy_node()  # 노드 제거
        rclpy.shutdown()  # ROS2 종료


if __name__ == "__main__":  # 파일 직접 실행 시
    main()  # main 실행
