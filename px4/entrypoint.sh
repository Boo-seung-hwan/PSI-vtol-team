#!/bin/bash
set -e

git config --global --add safe.directory /workspace/px4/PX4-Autopilot
git config --global --add safe.directory '*'

if [ ! -d /workspace/px4/PX4-Autopilot ]; then
  echo "[INFO] Cloning PX4-Autopilot..."
  git clone https://github.com/PX4/PX4-Autopilot.git /workspace/px4/PX4-Autopilot
fi

cd /workspace/px4/PX4-Autopilot

#git submodule deinit -f --all || true
#git submodule sync --recursive
#git submodule update --init --recursive

export PX4_UXRCE_DDS_PORT=${PX4_UXRCE_DDS_PORT:-8888}
export PX4_UXRCE_DDS_AG_IP=${PX4_UXRCE_DDS_AG_IP:-microxrce-agent}

export PX4_GZ_WORLD=${PX4_GZ_WORLD:-default}
export PX4_MAKE_TARGET=${PX4_MAKE_TARGET:-gz_standard_vtol}

export PX4_GZ_WORLD_PATH=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds
export PX4_GZ_MODELS=${PX4_GZ_MODELS:-/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models}
export PX4_GZ_WORLDS=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds

export GZ_SIM_RESOURCE_PATH=/workspace/custom_models:/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models:/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH}
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH}


echo "[INFO] Applying custom Gazebo models..."

PX4_GZ_MODEL_DIR="/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models"

if [ -d /workspace/custom_models/standard_vtol ]; then
  echo "[INFO] Override PX4 standard_vtol with custom standard_vtol"
  rm -rf "${PX4_GZ_MODEL_DIR}/standard_vtol"
  cp -r /workspace/custom_models/standard_vtol "${PX4_GZ_MODEL_DIR}/standard_vtol"
fi

if [ -d /workspace/custom_models/mono_cam ]; then
  echo "[INFO] Override PX4 mono_cam with custom mono_cam"
  rm -rf "${PX4_GZ_MODEL_DIR}/mono_cam"
  cp -r /workspace/custom_models/mono_cam "${PX4_GZ_MODEL_DIR}/mono_cam"
fi

echo "[INFO] Verifying custom camera model:"
grep -n "mono_cam\|mono_cam_joint\|mono_cam::camera_link" "${PX4_GZ_MODEL_DIR}/standard_vtol/model.sdf" || true
grep -n "/vtol/camera" "${PX4_GZ_MODEL_DIR}/mono_cam/model.sdf" || true

echo "[INFO] PX4_UXRCE_DDS_AG_IP=${PX4_UXRCE_DDS_AG_IP}"
echo "[INFO] PX4_UXRCE_DDS_PORT=${PX4_UXRCE_DDS_PORT}"
echo "[INFO] PX4_GZ_WORLD=${PX4_GZ_WORLD}"
echo "[INFO] PX4_MAKE_TARGET=${PX4_MAKE_TARGET}"
echo "[INFO] GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"

echo "[INFO] Starting Gazebo first..."
gz sim --verbose=4 -r -s /workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds/${PX4_GZ_WORLD}.sdf &
GZ_PID=$!

export MAV_BROADCAST=1

echo "[INFO] Waiting for Gazebo create service..."
for i in $(seq 1 60); do
  if gz service -l | grep -q "/world/${PX4_GZ_WORLD}/create"; then
    echo "[INFO] Gazebo create service is ready."
    break
  fi
  sleep 1
done


echo "[INFO] Starting PX4 SITL..."
export PX4_GZ_STANDALONE=1

WINDOWS_HOST_IP=$(grep -oP '(?<=host\()\d+\.\d+\.\d+\.\d+(?=\))' /etc/resolv.conf)

echo "[INFO] WINDOWS_HOST_IP=${WINDOWS_HOST_IP}"

RCS_FILE="/workspace/px4/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/rcS"

echo "[DEBUG_XRCE] patching uxrce_dds_client agent host"

sed -i \
  's|uxrce_dds_client start -t udp -p $uxrce_dds_port $uxrce_dds_ns|uxrce_dds_client start -t udp -h microxrce-agent -p $uxrce_dds_port $uxrce_dds_ns|' \
  "$RCS_FILE"

echo "[DEBUG_XRCE] rcS uXRCE line after patch:"
grep -n "uxrce_dds_client start" "$RCS_FILE" || true

echo "[DEBUG_QGC] running entrypoint: /workspace/px4/entrypoint.sh"
echo "[DEBUG_QGC] QGC_HOST_IP=${QGC_HOST_IP}"
echo "[DEBUG_QGC] RCS_FILE=${RCS_FILE}"

if [ -z "$QGC_HOST_IP" ]; then
  echo "[ERROR_QGC] QGC_HOST_IP is empty. Run ./start.sh, not docker compose up."
  exit 1
fi

sed -i '/AUTO_QGC_MAVLINK_START/,/AUTO_QGC_MAVLINK_END/d' "$RCS_FILE"
sed -i '/host\.docker\.internal/d' "$RCS_FILE"

{
  echo ""
  echo "# >>> AUTO_QGC_MAVLINK_START"
  echo "mavlink start -x -u 14558 -r 400000 -t $QGC_HOST_IP -o 14560"
  echo "# <<< AUTO_QGC_MAVLINK_END"
  echo ""
} >> "$RCS_FILE"

echo "[DEBUG_QGC] rcS block after patch:"
grep -n "AUTO_QGC_MAVLINK\|14558\|14560\|$QGC_HOST_IP" "$RCS_FILE" || true

make px4_sitl ${PX4_MAKE_TARGET}

wait $GZ_PID
