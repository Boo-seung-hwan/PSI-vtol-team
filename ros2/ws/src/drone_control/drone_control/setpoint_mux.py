import math
import time
from typing import Optional
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleLocalPosition


def has_finite(values) -> bool:
    # 배열 안에 NaN이 아닌 유효한 값이 하나라도 있는지 확인하는 함수
    # position, velocity, acceleration 중 어떤 setpoint가 사용되는지 판단할 때 사용

    return any(math.isfinite(float(v)) for v in values)


class SetpointMux(Node):
    # generator_tracking과 vertiport_tracking이 만든 setpoint 중 하나만 선택하여 PX4로 보내는 노드
    # PX4로 /fmu/in/trajectory_setpoint를 직접 publish하는 유일한 노드가 되어야 함

    def __init__(self):
        super().__init__("setpoint_mux")
        # ROS2 노드 이름 설정

        self.source = "HOLD"
        # 현재 선택된 setpoint source
        # 가능한 값: GENERATOR, VERTIPORT, HOLD

        self.generator_sp: Optional[TrajectorySetpoint] = None
        # generator_tracking에서 받은 마지막 setpoint 저장

        self.vertiport_sp: Optional[TrajectorySetpoint] = None
        # vertiport_tracking에서 받은 마지막 setpoint 저장

        self.last_generator_sp_time = 0.0
        # generator setpoint를 마지막으로 받은 시간

        self.last_vertiport_sp_time = 0.0
        # vertiport setpoint를 마지막으로 받은 시간

        self.current_pos = np.zeros(3)
        # PX4 현재 위치 저장 [x, y, z]

        self.current_yaw = 0.0
        # PX4 현재 yaw 저장

        self.local_ready = False
        # 현재 위치를 한 번이라도 받았는지 여부

        self.setpoint_timeout_s = 0.5
        # 선택된 source의 setpoint가 0.5초 이상 안 들어오면 stale로 판단하고 hold로 대체

        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # PX4로 publish하는 topic에 사용할 QoS
        # /fmu/in/offboard_control_mode, /fmu/in/trajectory_setpoint에 사용

        qos_px4_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # PX4에서 subscribe하는 topic에 사용할 QoS
        # /fmu/out/vehicle_local_position에 사용
        
        qos_tracking = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.source_sub = self.create_subscription(
            String,
            "/setpoint_mux/source",
            self.source_callback,
            10
        )
        # mission_manager가 선택한 source를 구독
        # 예: GENERATOR, VERTIPORT, HOLD

        self.generator_sp_sub = self.create_subscription(
            TrajectorySetpoint,
            "/generator_tracking/trajectory_setpoint",
            self.generator_sp_callback,
            qos_tracking
        )
        # generator_tracking이 만든 trajectory setpoint 구독

        self.vertiport_sp_sub = self.create_subscription(
            TrajectorySetpoint,
            "/vertiport_tracking/trajectory_setpoint",
            self.vertiport_sp_callback,
            qos_tracking
        )
        # vertiport_tracking이 만든 trajectory setpoint 구독

        self.local_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_position_callback,
            qos_px4_sub
        )
        # PX4 현재 위치 구독
        # HOLD setpoint 생성에 사용

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            qos_pub
        )
        # PX4 OffboardControlMode publisher
        # 선택된 setpoint가 position인지 velocity인지 PX4에 알려줌

        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            qos_pub
        )
        # PX4로 최종 trajectory setpoint를 publish하는 publisher
        # 이 topic은 setpoint_mux만 publish해야 함

        self.status_pub = self.create_publisher(
            String,
            "/setpoint_mux/status",
            10
        )
        # setpoint_mux 상태 확인용 publisher

        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )
        # 20Hz로 Offboard heartbeat와 trajectory setpoint publish

        self.get_logger().info("Setpoint Mux started.")
        self.get_logger().info("PX4 receives trajectory_setpoint only from setpoint_mux.")

    def now_us(self) -> int:
        # PX4 timestamp용 현재 시간을 microsecond 단위로 반환

        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_status(self, text: str):
        # setpoint_mux 상태를 문자열로 publish

        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def source_callback(self, msg: String):
        # mission_manager가 보낸 source 선택 메시지를 받을 때 실행

        source = msg.data.strip().upper()
        # 문자열 정리 후 대문자로 변환

        if source not in ["GENERATOR", "VERTIPORT", "HOLD"]:
            # 허용되지 않은 source이면

            self.get_logger().warn(f"Unknown mux source: {source}")
            # 경고 로그 출력

            return
            # source 변경하지 않음

        if source != self.source:
            # source가 바뀐 경우

            self.get_logger().info(f"Mux source changed: {self.source} -> {source}")
            # 변경 로그 출력

        self.source = source
        # 현재 source 저장

    def generator_sp_callback(self, msg: TrajectorySetpoint):
        # generator_tracking setpoint 수신 callback

        self.generator_sp = msg
        # 마지막 generator setpoint 저장

        self.last_generator_sp_time = time.monotonic()
        # 수신 시간 저장

    def vertiport_sp_callback(self, msg: TrajectorySetpoint):
        # vertiport_tracking setpoint 수신 callback

        self.vertiport_sp = msg
        # 마지막 vertiport setpoint 저장

        self.last_vertiport_sp_time = time.monotonic()
        # 수신 시간 저장

    def local_position_callback(self, msg: VehicleLocalPosition):
        # PX4 현재 위치 수신 callback

        self.current_pos = np.array([msg.x, msg.y, msg.z], dtype=float)
        # 현재 NED 위치 저장

        self.current_yaw = float(msg.heading)
        # 현재 yaw 저장

        self.local_ready = True
        # 현재 위치 수신 완료 표시

    def is_stale(self, stamp: float) -> bool:
        # setpoint가 너무 오래되었는지 확인

        if stamp <= 0.0:
            # 아직 해당 setpoint를 받은 적이 없다면

            return True
            # stale로 판단

        return time.monotonic() - stamp > self.setpoint_timeout_s
        # 마지막 수신 이후 timeout_s보다 오래 지났으면 stale

    def make_hold_setpoint(self) -> Optional[TrajectorySetpoint]:
        # 현재 위치를 유지하는 hold setpoint 생성

        if not self.local_ready:
            # 현재 위치를 아직 받지 못했다면

            return None
            # hold setpoint 생성 불가

        msg = TrajectorySetpoint()
        # TrajectorySetpoint 메시지 생성

        msg.timestamp = self.now_us()
        # timestamp 설정

        msg.position = [
            float(self.current_pos[0]),
            float(self.current_pos[1]),
            float(self.current_pos[2]),
        ]
        # 현재 위치를 목표 위치로 설정
        # 즉 현재 위치 hold

        msg.velocity = [math.nan, math.nan, math.nan]
        # velocity setpoint는 사용하지 않음

        msg.acceleration = [math.nan, math.nan, math.nan]
        # acceleration setpoint는 사용하지 않음

        msg.yaw = float(self.current_yaw)
        # 현재 yaw 유지

        msg.yawspeed = math.nan
        # yaw rate setpoint는 사용하지 않음

        return msg
        # hold setpoint 반환

    def select_setpoint(self) -> tuple[Optional[TrajectorySetpoint], str]:
        # mission_manager가 선택한 source에 따라 PX4로 보낼 setpoint를 선택

        if self.source == "GENERATOR":
            # generator_tracking setpoint를 사용할 경우

            if self.generator_sp is None or self.is_stale(self.last_generator_sp_time):
                # generator setpoint가 없거나 오래되었으면

                return self.make_hold_setpoint(), "GENERATOR_STALE_HOLD"
                # 현재 위치 hold로 대체

            return self.generator_sp, "GENERATOR_SELECTED"
            # 정상 generator setpoint 선택

        if self.source == "VERTIPORT":
            # vertiport_tracking setpoint를 사용할 경우

            if self.vertiport_sp is None or self.is_stale(self.last_vertiport_sp_time):
                # vertiport setpoint가 없거나 오래되었으면

                return self.make_hold_setpoint(), "VERTIPORT_STALE_HOLD"
                # 현재 위치 hold로 대체

            return self.vertiport_sp, "VERTIPORT_SELECTED"
            # 정상 vertiport setpoint 선택

        return self.make_hold_setpoint(), "HOLD_SELECTED"
        # source가 HOLD이면 현재 위치 hold setpoint 선택

    def infer_offboard_mode(self, sp: TrajectorySetpoint) -> tuple[bool, bool, bool]:
        # 선택된 TrajectorySetpoint를 보고 PX4 OffboardControlMode를 결정
        # position 값이 있으면 position mode
        # velocity 값이 있으면 velocity mode
        # acceleration 값이 있으면 acceleration mode

        use_position = has_finite(sp.position)
        # position 배열에 유효한 값이 있는지 확인

        use_velocity = has_finite(sp.velocity)
        # velocity 배열에 유효한 값이 있는지 확인

        use_acceleration = has_finite(sp.acceleration)
        # acceleration 배열에 유효한 값이 있는지 확인

        if use_position:
            # position setpoint가 있으면 position control 사용

            return True, False, False

        if use_velocity:
            # position setpoint가 없고 velocity 값이 있으면 velocity control 사용

            return False, True, False

        if use_acceleration:
            # acceleration만 있으면 acceleration control 사용

            return False, False, True

        return True, False, False
        # 아무 유효값도 없으면 안전하게 position mode로 hold 시도

    def publish_offboard_mode(self, sp: TrajectorySetpoint):
        # 선택된 setpoint에 맞는 OffboardControlMode를 PX4로 publish

        use_position, use_velocity, use_acceleration = self.infer_offboard_mode(sp)
        # setpoint 내용을 보고 사용할 제어 모드 판단

        msg = OffboardControlMode()
        # OffboardControlMode 메시지 생성

        msg.timestamp = self.now_us()
        # timestamp 설정

        msg.position = use_position
        # position setpoint 사용 여부

        msg.velocity = use_velocity
        # velocity setpoint 사용 여부

        msg.acceleration = use_acceleration
        # acceleration setpoint 사용 여부

        msg.attitude = False
        # attitude 직접 제어는 사용하지 않음

        msg.body_rate = False
        # body rate 직접 제어는 사용하지 않음

        if hasattr(msg, "thrust_and_torque"):
            # px4_msgs 버전에 따라 필드가 있을 수 있음

            msg.thrust_and_torque = False
            # thrust/torque 직접 제어 미사용

        if hasattr(msg, "direct_actuator"):
            # px4_msgs 버전에 따라 필드가 있을 수 있음

            msg.direct_actuator = False
            # actuator 직접 제어 미사용

        self.offboard_pub.publish(msg)
        # PX4로 OffboardControlMode publish

    def copy_setpoint(self, src: TrajectorySetpoint) -> TrajectorySetpoint:
        # 받은 setpoint를 복사하되 timestamp는 현재 시간으로 새로 설정
        # 오래된 timestamp 때문에 PX4가 무시하지 않도록 하기 위함

        msg = TrajectorySetpoint()
        # 새 TrajectorySetpoint 생성

        msg.timestamp = self.now_us()
        # 현재 시간으로 timestamp 갱신

        msg.position = list(src.position)
        # position 복사

        msg.velocity = list(src.velocity)
        # velocity 복사

        msg.acceleration = list(src.acceleration)
        # acceleration 복사

        msg.yaw = float(src.yaw)
        # yaw 복사

        msg.yawspeed = float(src.yawspeed)
        # yawspeed 복사

        return msg
        # 복사된 setpoint 반환

    def control_loop(self):
        # setpoint_mux 메인 루프
        # 선택된 setpoint를 PX4로 전달

        selected_sp, status = self.select_setpoint()
        # 현재 source에 맞는 setpoint 선택

        if selected_sp is None:
            # 현재 위치도 없고 사용할 setpoint도 없다면

            self.publish_status("NO_LOCAL_POSITION")
            # 상태 publish

            return
            # 이번 loop 종료

        output_sp = self.copy_setpoint(selected_sp)
        # timestamp를 현재 시간으로 갱신한 setpoint 생성

        self.publish_offboard_mode(output_sp)
        # setpoint에 맞는 OffboardControlMode publish

        self.traj_pub.publish(output_sp)
        # PX4로 최종 TrajectorySetpoint publish

        self.publish_status(status)
        # mux 상태 publish


def main(args=None):
    # ROS2 실행 진입점

    rclpy.init(args=args)
    # ROS2 초기화

    node = SetpointMux()
    # SetpointMux 노드 생성

    try:
        rclpy.spin(node)
        # 노드 실행 유지
    except KeyboardInterrupt:
        # Ctrl+C 종료 시

        pass
    finally:
        # 종료 시 반드시 실행

        node.destroy_node()
        # 노드 제거

        if rclpy.ok():
            rclpy.shutdown()
        # ROS2 종료
         

if __name__ == "__main__":
    main()