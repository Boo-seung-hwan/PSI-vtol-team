#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC_DIR="$REPO_ROOT/ros2/ws/src"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] SRC_DIR=$SRC_DIR"

if [ ! -d "$SRC_DIR" ]; then
  echo "[ERROR] ROS2 src dir not found: $SRC_DIR"
  exit 1
fi

echo "[1/4] PX4 main/1.17 topic name patches"

sed -i 's#"/fmu/out/vehicle_local_position"#"/fmu/out/vehicle_local_position_v1"#g' \
  "$SRC_DIR/drone_control/drone_control/setpoint_mux.py" \
  "$SRC_DIR/drone_control/drone_control/precision_landing_test_input.py" \
  "$SRC_DIR/drone_control/drone_control/generator_tracking.py" \
  "$SRC_DIR/drone_control/drone_control/vertiport_tracking.py" \
  "$SRC_DIR/drone_control/drone_control/precision_landing_controller.py" \
  "$SRC_DIR/drone_vision/drone_vision/target_camera_to_ned.py"

sed -i 's#"/fmu/out/vehicle_status"#"/fmu/out/vehicle_status_v1"#g' \
  "$SRC_DIR/drone_control/drone_control/generator_tracking.py"

echo "[2/4] Add RealSense 16UC1 depth scaling patch"

python3 - <<PY
from pathlib import Path

p = Path("$SRC_DIR/drone_vision/drone_vision/target_depth_estimator.py")
text = p.read_text()

old = '''depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth, dtype=np.float32)'''

new = '''depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth_raw, dtype=np.float32)

        # RealSense depth image is often 16UC1, where values are in millimeters.
        # Convert all depth values to meters before using them.
        if msg.encoding in ("16UC1", "mono16"):
            depth *= 0.001
        elif msg.encoding in ("32FC1",):
            pass
        else:
            self.get_logger().warn_once(
                f"Unknown depth encoding: {msg.encoding}. Assuming depth is already in meters."
            )'''

if "RealSense depth image is often 16UC1" in text:
    print("[INFO] depth scale patch already exists")
elif old in text:
    text = text.replace(old, new)
    p.write_text(text)
    print("[INFO] depth scale patch applied")
else:
    print("[WARN] expected depth conversion block not found. Check target_depth_estimator.py manually.")
PY

echo "[3/4] Normalize line endings CRLF -> LF"

python3 - <<PY
from pathlib import Path

files = [
    "$SRC_DIR/drone_control/drone_control/setpoint_mux.py",
    "$SRC_DIR/drone_control/drone_control/precision_landing_test_input.py",
    "$SRC_DIR/drone_control/drone_control/generator_tracking.py",
    "$SRC_DIR/drone_control/drone_control/vertiport_tracking.py",
    "$SRC_DIR/drone_control/drone_control/precision_landing_controller.py",
    "$SRC_DIR/drone_vision/drone_vision/target_camera_to_ned.py",
    "$SRC_DIR/drone_vision/drone_vision/target_depth_estimator.py",
]

for f in files:
    p = Path(f)
    data = p.read_bytes()
    new = data.replace(b"\\r\\n", b"\\n")
    if new != data:
        p.write_bytes(new)
        print(f"[INFO] converted CRLF -> LF: {p}")
PY

echo "[4/4] Verify patches"

grep -R '"/fmu/out/vehicle_local_position"' -n "$SRC_DIR/drone_control" "$SRC_DIR/drone_vision" || true
grep -R '"/fmu/out/vehicle_status"' -n "$SRC_DIR/drone_control" || true

echo
echo "[INFO] New PX4 topic references:"
grep -R '"/fmu/out/vehicle_local_position_v1"' -n "$SRC_DIR/drone_control" "$SRC_DIR/drone_vision" || true
grep -R '"/fmu/out/vehicle_status_v1"' -n "$SRC_DIR/drone_control" || true

echo
echo "[DONE] Jetson patches applied."
