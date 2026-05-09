import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition


class OffboardController(Node):
    def __init__(self):
        super().__init__('offboard_controller')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile
        )

        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile
        )

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )

        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v4',
            self.vehicle_status_callback,
            qos_profile
        )

        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback,
            qos_profile
        )

        self.last_arm_request_time = 0.0

        self.vehicle_status = None
        self.vehicle_local_position = None
        self.offboard_setpoint_counter = 0
        self.arm_sent = False

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.relative_waypoints = [
            (0.0, 0.0, -2.0),
            (2.0, 0.0, -2.0),
            (2.0, 2.0, -2.0),
            (0.0, 2.0, -2.0),
            (0.0, 0.0, -2.0),
        ]

        self.current_wp_idx = 0
        self.wp_reached_time = None
        self.acceptance_radius = 0.3
        self.hold_time = 2.0
        self.mission_complete = False

        self.last_logged_wp_idx = -1
        self.final_wp_logged = False

        self.reset_stable_since = None
        self.reset_release_delay = 1.0

        # home 기준점
        self.home_position_initialized = False
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0

        # reset counter 감지용
        self.prev_xy_reset_counter = None
        self.prev_z_reset_counter = None
        self.prev_heading_reset_counter = None

        # reset 발생 시 현재 위치 hold
        self.reset_detected = False
        self.reset_hold_x = 0.0
        self.reset_hold_y = 0.0
        self.reset_hold_z = 0.0

        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def initialize_home_position(self):
        self.home_x = self.vehicle_local_position.x
        self.home_y = self.vehicle_local_position.y
        self.home_z = self.vehicle_local_position.z
        self.home_position_initialized = True

        self.prev_xy_reset_counter = self.vehicle_local_position.xy_reset_counter
        self.prev_z_reset_counter = self.vehicle_local_position.z_reset_counter
        self.prev_heading_reset_counter = self.vehicle_local_position.heading_reset_counter

        self.get_logger().info(
            f"Home position initialized at "
            f"({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f})"
        )

    def check_local_position_reset(self, now):
        xy_changed = (
            self.prev_xy_reset_counter is not None and
            self.vehicle_local_position.xy_reset_counter != self.prev_xy_reset_counter
        )
        z_changed = (
            self.prev_z_reset_counter is not None and
            self.vehicle_local_position.z_reset_counter != self.prev_z_reset_counter
        )
        heading_changed = (
            self.prev_heading_reset_counter is not None and
            self.vehicle_local_position.heading_reset_counter != self.prev_heading_reset_counter
        )

        changed = xy_changed or z_changed or heading_changed

        if changed:
            if not self.reset_detected:
                self.get_logger().warn("Local position reset detected. Entering reset-hold.")
            else:
                self.get_logger().warn("Local position reset changed again while holding.")

            self.reset_detected = True
            self.reset_hold_x = self.vehicle_local_position.x
            self.reset_hold_y = self.vehicle_local_position.y
            self.reset_hold_z = self.vehicle_local_position.z
            self.reset_stable_since = now
        else:
            if self.reset_detected and self.reset_stable_since is None:
                self.reset_stable_since = now

        self.prev_xy_reset_counter = self.vehicle_local_position.xy_reset_counter
        self.prev_z_reset_counter = self.vehicle_local_position.z_reset_counter
        self.prev_heading_reset_counter = self.vehicle_local_position.heading_reset_counter

    def get_absolute_target(self):
        rel_x, rel_y, rel_z = self.relative_waypoints[self.current_wp_idx]
        return (
            self.home_x + rel_x,
            self.home_y + rel_y,
            self.home_z + rel_z
        )

    def timer_callback(self):
        if self.vehicle_status is None or self.vehicle_local_position is None:
            if self.vehicle_status is None:
                self.get_logger().warn('Waiting for vehicle_status...')
            if self.vehicle_local_position is None:
                self.get_logger().warn('Waiting for vehicle_local_position...')
            return

        now = self.get_clock().now().nanoseconds / 1e9
        self.publish_offboard_control_mode()

        # OFFBOARD 진입 후에 home 기준점 1회 캡처
        if (
            not self.home_position_initialized and
            self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
            self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        ):
            self.initialize_home_position()

        # 기준점이 잡힌 뒤에는 reset counter 감시
        if self.home_position_initialized:
            self.check_local_position_reset(now)

        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        current_z = self.vehicle_local_position.z

        # reset 발생 시 현재 위치 hold
        if self.reset_detected:
            if (
                self.reset_stable_since is not None and
                (now - self.reset_stable_since) < self.reset_release_delay
            ):
                self.reset_hold_x = current_x
                self.reset_hold_y = current_y
                self.reset_hold_z = current_z

            target = (self.reset_hold_x, self.reset_hold_y, self.reset_hold_z)
            self.publish_trajectory_setpoint(*target)

            dx = current_x - target[0]
            dy = current_y - target[1]
            dz = current_z - target[2]

            if self.offboard_setpoint_counter % 10 == 0:
                self.get_logger().error(
                    f"RESET HOLD | current=({current_x:.2f}, {current_y:.2f}, {current_z:.2f}), "
                    f"hold_target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), "
                    f"err=({dx:.2f}, {dy:.2f}, {dz:.2f})"
                )

            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                    if now - self.last_arm_request_time > 1.0:
                        self.arm()
                        self.last_arm_request_time = now
                        self.get_logger().info(
                            f"Sending ARM request again during reset-hold, "
                            f"arming_state={self.vehicle_status.arming_state}"
                        )

            if (
                self.reset_stable_since is not None and
                (now - self.reset_stable_since) >= self.reset_release_delay
            ):
                self.get_logger().warn("Local position reset stabilized. Re-initializing home position.")
                self.initialize_home_position()
                self.reset_detected = False
                self.wp_reached_time = None
                self.last_logged_wp_idx = -1

            self.offboard_setpoint_counter += 1
            return

        # home 기준 상대 waypoint → 절대 target 생성
        if self.home_position_initialized:
            target = self.get_absolute_target()
        else:
            # 아직 OFFBOARD 진입 전이면 현재 위치 부근 유지
            target = (current_x, current_y, current_z)

        self.publish_trajectory_setpoint(*target)

        dx = current_x - target[0]
        dy = current_y - target[1]
        dz = current_z - target[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if self.current_wp_idx != self.last_logged_wp_idx:
            self.get_logger().info(
                f"Heading to waypoint {self.current_wp_idx + 1}/{len(self.relative_waypoints)} | "
                f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
            )
            self.last_logged_wp_idx = self.current_wp_idx

        if self.offboard_setpoint_counter % 10 == 0:
            self.get_logger().info(
                f"nav_state={self.vehicle_status.nav_state}, "
                f"arming_state={self.vehicle_status.arming_state}, "
                f"wp={self.current_wp_idx + 1}/{len(self.relative_waypoints)}, "
                f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), "
                f"current=({current_x:.2f}, {current_y:.2f}, {current_z:.2f}), "
                f"error={dist:.2f} m"
            )

        if self.home_position_initialized:
            if dist < self.acceptance_radius:
                if self.wp_reached_time is None:
                    self.wp_reached_time = now
                    self.get_logger().info(
                        f"Waypoint {self.current_wp_idx + 1} entered acceptance radius "
                        f"({dist:.2f} m < {self.acceptance_radius:.2f} m). "
                        f"Holding for {self.hold_time:.1f} s."
                    )
                elif now - self.wp_reached_time > self.hold_time:
                    if self.current_wp_idx < len(self.relative_waypoints) - 1:
                        self.current_wp_idx += 1
                        self.wp_reached_time = None
                        self.get_logger().info(
                            f"Waypoint reached. Advancing to waypoint "
                            f"{self.current_wp_idx + 1}/{len(self.relative_waypoints)}."
                        )
                    else:
                        if not self.final_wp_logged:
                            self.mission_complete = True
                            self.get_logger().info("Final waypoint reached and hold complete. Sending LAND.")
                            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                            self.final_wp_logged = True
            else:
                self.wp_reached_time = None

        if self.offboard_setpoint_counter < 50:
            self.offboard_setpoint_counter += 1
            return

        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.set_offboard_mode()
            self.get_logger().info(
                f"Requesting OFFBOARD, nav_state={self.vehicle_status.nav_state}, "
                f"arming_state={self.vehicle_status.arming_state}"
            )
            self.offboard_setpoint_counter += 1
            return

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                if now - self.last_arm_request_time > 1.0:
                    self.arm()
                    self.last_arm_request_time = now
                    self.get_logger().info(
                        f"Sending ARM request again, arming_state={self.vehicle_status.arming_state}"
                    )
                self.offboard_setpoint_counter += 1
                return

        self.offboard_setpoint_counter += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.vehicle_command_publisher.publish(msg)

    def set_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0
        )

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = OffboardController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
