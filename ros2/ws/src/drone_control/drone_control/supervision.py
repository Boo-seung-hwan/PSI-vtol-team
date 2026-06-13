import time
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionState(Enum):
    # 전체 임무 상태 정의

    WAIT_START = auto()
    # 임무 시작 전 대기 상태

    GENERATOR_OUTBOUND = auto()
    # 출발 경로 추종 상태
 

    RESCUE_TRACKING = auto()
    # 조난자 구조 정밀제어 상태
  

    GENERATOR_RETURN = auto()
    # 복귀 경로 추종 상태
   

    VERTIPORT_TRACKING = auto()
    # 버티포트 정밀 착륙 상태
   

    MISSION_DONE = auto()
    # 전체 임무 완료 상태

    FAILSAFE = auto()
    # 비상 상태


class MissionManager(Node):
    # 전체 임무 순서를 관리하는 상위 노드
   

    def __init__(self):
        super().__init__("mission_manager")
        # ROS2 노드 이름 설정

        self.state = MissionState.WAIT_START
        # 현재 mission state 저장

        self.state_start_time = time.monotonic()
        # 현재 상태가 시작된 시간 저장

        self.generator_status = ""
        # generator_tracking에서 마지막으로 받은 상태 문자열

        self.vertiport_status = ""
        # vertiport_tracking에서 마지막으로 받은 상태 문자열

        self.last_generator_status_time = 0.0
        # generator_tracking status를 마지막으로 받은 시간

        self.last_vertiport_status_time = 0.0
        # vertiport_tracking status를 마지막으로 받은 시간

        self.status_timeout_s = 2.0
        # 하위 노드에서 status가 2초 이상 안 오면 timeout 경고

        self.mission_state_pub = self.create_publisher(
            String,
            "/mission/state",
            10
        )
        # generator_tracking, vertiport_tracking에게 현재 임무 상태를 알려주는 publisher

        self.mux_source_pub = self.create_publisher(
            String,
            "/setpoint_mux/source",
            10
        )
        # setpoint_mux에게 어떤 setpoint를 PX4로 보낼지 알려주는 publisher

        self.status_pub = self.create_publisher(
            String,
            "/mission/status",
            10
        )
        # mission_manager 자체 상태를 외부에서 확인하기 위한 publisher

        self.generator_status_sub = self.create_subscription(
            String,
            "/generator_tracking/status",
            self.generator_status_callback,
            10
        )
        # generator_tracking의 진행 상태를 구독

        self.vertiport_status_sub = self.create_subscription(
            String,
            "/vertiport_tracking/status",
            self.vertiport_status_callback,
            10
        )
        # vertiport_tracking의 진행 상태를 구독

        self.start_sub = self.create_subscription(
            String,
            "/mission/start",
            self.start_callback,
            10
        )
        # 임무 시작 명령 구독
        # START를 받으면 임무 시작

        self.abort_sub = self.create_subscription(
            String,
            "/mission/abort",
            self.abort_callback,
            10
        )
        # 임무 중단 또는 비상 명령 구독

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )
        # 10Hz로 mission manager loop 실행

        self.get_logger().info("Mission Manager started.")
        self.get_logger().info("Publish START to /mission/start to begin mission.")

    def start_callback(self, msg: String):
        # /mission/start topic으로 시작 명령이 들어오면 실행

        command = msg.data.strip().upper()
        # 문자열 앞뒤 공백 제거 후 대문자로 변환

        if command == "START" and self.state == MissionState.WAIT_START:
            # START 명령이고 현재 대기 상태라면

            self.change_state(MissionState.GENERATOR_OUTBOUND)
            # 출발 경로 추종 상태로 전환

    def abort_callback(self, msg: String):
        # /mission/abort topic으로 중단 명령이 들어오면 실행

        command = msg.data.strip().upper()

        if command in ["ABORT", "STOP", "FAILSAFE"]:
            # 중단 명령이면

            self.get_logger().warn(f"Mission abort command received: {command}")
            # 경고 로그 출력

            self.change_state(MissionState.FAILSAFE)
            # 비상 상태로 전환

    def generator_status_callback(self, msg: String):
        # generator_tracking에서 status 메시지를 받을 때 실행

        self.generator_status = msg.data.strip()
        # generator_tracking의 마지막 상태 저장

        self.last_generator_status_time = time.monotonic()
        # status 수신 시간 저장

    def vertiport_status_callback(self, msg: String):
        # vertiport_tracking에서 status 메시지를 받을 때 실행

        self.vertiport_status = msg.data.strip()
        # vertiport_tracking의 마지막 상태 저장

        self.last_vertiport_status_time = time.monotonic()
        # status 수신 시간 저장

    def publish_string(self, pub, text: str):
        # String 메시지를 publish하는 공통 함수

        msg = String()
        msg.data = text
        pub.publish(msg)

    def publish_mission_state(self, text: str):
        # 하위 제어 노드들에게 현재 임무 상태를 publish

        self.publish_string(self.mission_state_pub, text)

    def publish_mux_source(self, text: str):
        # setpoint_mux에게 사용할 setpoint source를 publish

        self.publish_string(self.mux_source_pub, text)

    def publish_status(self, text: str):
        # mission manager 상태를 publish

        self.publish_string(self.status_pub, text)

    def change_state(self, new_state: MissionState):
        # mission state를 변경하는 함수

        if new_state == self.state:
            # 같은 상태로 바꾸는 경우에는 아무 것도 하지 않음

            return

        self.get_logger().info(
            f"Mission state changed: {self.state.name} -> {new_state.name}"
        )
        # 상태 전환 로그 출력

        self.state = new_state
        # 현재 상태 갱신

        self.state_start_time = time.monotonic()
        # 새 상태 시작 시간 저장

        self.generator_status = ""
        # 이전 상태의 generator status 초기화

        self.vertiport_status = ""
        # 이전 상태의 vertiport status 초기화

        self.last_generator_status_time = 0.0
        # generator status 수신 시간 초기화

        self.last_vertiport_status_time = 0.0
        # vertiport status 수신 시간 초기화

    def elapsed(self) -> float:
        # 현재 상태에 들어온 뒤 경과 시간 반환

        return time.monotonic() - self.state_start_time

    def generator_timeout(self) -> bool:
        # generator_tracking status가 일정 시간 이상 들어오지 않았는지 확인

        if self.last_generator_status_time <= 0.0:
            # 아직 status를 한 번도 받지 못했다면

            return self.elapsed() > self.status_timeout_s
            # 상태 진입 후 timeout_s가 지나면 timeout으로 판단

        return time.monotonic() - self.last_generator_status_time > self.status_timeout_s
        # 마지막 status 이후 timeout_s보다 오래 지났으면 True

    def vertiport_timeout(self) -> bool:
        # vertiport_tracking status가 일정 시간 이상 들어오지 않았는지 확인

        if self.last_vertiport_status_time <= 0.0:
            # 아직 status를 한 번도 받지 못했다면

            return self.elapsed() > self.status_timeout_s

        return time.monotonic() - self.last_vertiport_status_time > self.status_timeout_s

    def outbound_done(self) -> bool:
        # 출발 경로 추종이 완료되었는지 판단

        return self.generator_status in [
            "GENERATOR_OUTBOUND_DONE",
            "ARRIVED_REP",
            "OUTBOUND_DONE",
        ]

    def rescue_done(self) -> bool:
        # 조난자 구조 정밀제어가 완료되었는지 판단
        status = self.vertiport_status.split("|", 1)[0].strip()

        return status in [
            "RESCUE_RETURN_ALT_REACHED",
            "RESCUE_DONE_HOLD",
            "RESCUE_TRACKING_DONE",
        ]

    def return_done(self) -> bool:
        # 복귀 경로 추종이 완료되었는지 판단

        return self.generator_status in [
            "GENERATOR_RETURN_DONE",
            "ARRIVED_WP1",
            "RETURN_DONE",
        ]

    def landing_done(self) -> bool:
        # 버티포트 정밀 착륙이 완료되었는지 판단

        return self.vertiport_status in [
            "VERTIPORT_LANDING_TARGET_REACHED",
            "VERTIPORT_HOLD_AFTER_REACHED",
            "VERTIPORT_TRACKING_DONE",
        ]

    def control_loop(self):
        # Mission Manager 메인 루프
        # 현재 상태에 따라 mission_state와 mux_source를 publish하고,
        # 완료 조건을 확인해 다음 상태로 전환한다.

        if self.state == MissionState.WAIT_START:
            # 임무 시작 전 대기 상태

            self.publish_mission_state("INACTIVE")
            # 하위 제어 노드 비활성화

            self.publish_mux_source("HOLD")
            # setpoint_mux는 현재 위치 hold

            self.publish_status("WAIT_START")
            # mission manager 상태 publish

            return

        if self.state == MissionState.GENERATOR_OUTBOUND:
            # 출발 경로 추종 단계

            self.publish_mission_state("GENERATOR_OUTBOUND")
            # generator_tracking에게 출발 경로 추종 상태임을 알림

            self.publish_mux_source("GENERATOR")
            # setpoint_mux가 generator_tracking setpoint를 선택하도록 함

            self.publish_status("GENERATOR_OUTBOUND_ACTIVE")
            # mission manager 상태 publish

            if self.generator_timeout():
                # generator_tracking status가 timeout이면

                self.get_logger().warn("generator_tracking status timeout during outbound.")
                # 경고 로그 출력

            if self.outbound_done():
                # REP 또는 구조 진입점까지 도착하면

                self.change_state(MissionState.RESCUE_TRACKING)
                # 조난자 구조 정밀제어 상태로 전환

            return

        if self.state == MissionState.RESCUE_TRACKING:
            # 조난자 구조 정밀제어 단계

            self.publish_mission_state("RESCUE_APPROACH")
            # vertiport_tracking에게 조난자 구조 접근 상태임을 알림

            self.publish_mux_source("VERTIPORT")
            # setpoint_mux가 vertiport_tracking setpoint를 선택하도록 함

            self.publish_status("RESCUE_TRACKING_ACTIVE")
            # mission manager 상태 publish

            if self.vertiport_timeout():
                # vertiport_tracking status timeout 확인

                self.get_logger().warn("vertiport_tracking status timeout during rescue.")
                # 경고 로그 출력

            if self.rescue_done():
                # 구조 완료 후 원래 고도 상승까지 완료되면

                self.change_state(MissionState.GENERATOR_RETURN)
                # 복귀 경로 추종 상태로 전환

            return

        if self.state == MissionState.GENERATOR_RETURN:
            # 복귀 경로 추종 단계

            self.publish_mission_state("GENERATOR_RETURN")
            # generator_tracking에게 복귀 경로 추종 상태임을 알림

            self.publish_mux_source("GENERATOR")
            # setpoint_mux가 generator_tracking setpoint를 선택하도록 함

            self.publish_status("GENERATOR_RETURN_ACTIVE")
            # mission manager 상태 publish

            if self.generator_timeout():
                # generator_tracking timeout 확인

                self.get_logger().warn("generator_tracking status timeout during return.")

            if self.return_done():
                # WP1 또는 버티포트 접근 지점까지 복귀 완료되면

                self.change_state(MissionState.VERTIPORT_TRACKING)
                # 버티포트 정밀 착륙 상태로 전환

            return

        if self.state == MissionState.VERTIPORT_TRACKING:
            # 버티포트 정밀 착륙 단계

            self.publish_mission_state("VERTIPORT_LANDING")
            # vertiport_tracking에게 버티포트 착륙 상태임을 알림

            self.publish_mux_source("VERTIPORT")
            # setpoint_mux가 vertiport_tracking setpoint를 선택하도록 함

            self.publish_status("VERTIPORT_TRACKING_ACTIVE")
            # mission manager 상태 publish

            if self.vertiport_timeout():
                # vertiport_tracking timeout 확인

                self.get_logger().warn("vertiport_tracking status timeout during landing.")

            if self.landing_done():
                # 착륙 목표 도달 상태를 받으면

                self.change_state(MissionState.MISSION_DONE)
                # 전체 임무 완료 상태로 전환

            return

        if self.state == MissionState.MISSION_DONE:
            # 임무 완료 상태

            self.publish_mission_state("MISSION_DONE")
            # 하위 제어 노드에게 임무 완료 상태 알림

            self.publish_mux_source("HOLD")
            # setpoint_mux는 hold setpoint 사용

            self.publish_status("MISSION_DONE")
            # mission manager 상태 publish

            return

        if self.state == MissionState.FAILSAFE:
            # 비상 상태

            self.publish_mission_state("FAILSAFE")
            # 하위 제어 노드에게 비상 상태 알림

            self.publish_mux_source("HOLD")
            # 우선 현재 위치 hold
            # 추후 RTH, LAND 등으로 연결 가능

            self.publish_status("FAILSAFE_HOLD")
            # mission manager 상태 publish

            return


def main(args=None):
    # ROS2 실행 진입점

    rclpy.init(args=args)
    # ROS2 초기화

    node = MissionManager()
    # MissionManager 노드 생성

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

        rclpy.shutdown()
        # ROS2 종료


if __name__ == "__main__":
    main()