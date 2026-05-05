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
export PX4_MAKE_TARGET=${PX4_MAKE_TARGET:-gz_x500}

export PX4_GZ_WORLD_PATH=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds
export PX4_GZ_MODELS=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models
export PX4_GZ_WORLDS=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds

export GZ_SIM_RESOURCE_PATH=/workspace/custom_models:/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models:/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH}
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH}

echo "[INFO] PX4_UXRCE_DDS_AG_IP=${PX4_UXRCE_DDS_AG_IP}"
echo "[INFO] PX4_UXRCE_DDS_PORT=${PX4_UXRCE_DDS_PORT}"
echo "[INFO] PX4_GZ_WORLD=${PX4_GZ_WORLD}"
echo "[INFO] PX4_MAKE_TARGET=${PX4_MAKE_TARGET}"
echo "[INFO] GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"

echo "[INFO] Starting Gazebo first..."
gz sim --verbose=1 -r -s /workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds/${PX4_GZ_WORLD}.sdf &
GZ_PID=$!

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

# Auto-detect Windows/WSL host IP for QGroundControl.
# In most WSL2 setups, /etc/resolv.conf nameserver points to the Windows host.
export QGC_IP=${QGC_IP:-$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)}

echo "[INFO] QGC_IP=${QGC_IP}"

RCS_FILE="/workspace/px4/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/rcS"
QGC_BLOCK_START="# >>> AUTO_QGC_MAVLINK_START"
QGC_BLOCK_END="# <<< AUTO_QGC_MAVLINK_END"

# Remove old injected block to avoid duplicates.
sed -i "/${QGC_BLOCK_START}/,/${QGC_BLOCK_END}/d" "$RCS_FILE"

cat <<EOF >> "$RCS_FILE"

${QGC_BLOCK_START}
if [ -n "\$QGC_IP" ]; then
    echo "[INFO] Starting extra MAVLink route to QGC at \$QGC_IP:14550"
    mavlink start -x -u 14558 -r 400000 -t \$QGC_IP -o 14550
fi
${QGC_BLOCK_END}
EOF

make px4_sitl ${PX4_MAKE_TARGET}

wait $GZ_PID
