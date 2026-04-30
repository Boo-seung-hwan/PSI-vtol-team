#!/bin/bash
set -e

git config --global --add safe.directory /workspace/px4/PX4-Autopilot
git config --global --add safe.directory '*'

if [ ! -d /workspace/px4/PX4-Autopilot ]; then
  echo "[INFO] Cloning PX4-Autopilot..."
  git clone https://github.com/PX4/PX4-Autopilot.git /workspace/px4/PX4-Autopilot
fi

cd /workspace/px4/PX4-Autopilot

git submodule deinit -f --all || true
git submodule sync --recursive
git submodule update --init --recursive


export PX4_UXRCE_DDS_PORT=${PX4_UXRCE_DDS_PORT:-8888}
export PX4_UXRCE_DDS_AG_IP=${PX4_UXRCE_DDS_AG_IP:-microxrce-agent}

export GZ_SIM_RESOURCE_PATH=/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models:/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH}
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH}

echo "[INFO] PX4_UXRCE_DDS_AG_IP=${PX4_UXRCE_DDS_AG_IP}"
echo "[INFO] PX4_UXRCE_DDS_PORT=${PX4_UXRCE_DDS_PORT}"
echo "[INFO] GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
echo "[INFO] Building and starting PX4 SITL..."

make px4_sitl gz_x500
