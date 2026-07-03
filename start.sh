#!/usr/bin/env bash
set -e

MODE="${1:-auto}"

export QGC_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')

echo "[INFO] QGC_HOST_IP=${QGC_HOST_IP}"
echo "[INFO] Requested mode: ${MODE}"

if [ "${MODE}" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="gpu"
  else
    MODE="cpu"
  fi
fi

case "${MODE}" in
  cpu)
    echo "[INFO] Starting in CPU/headless-compatible mode"
    docker compose -f compose.yaml -f compose.cpu.yaml up -d
    ;;

  gpu)
    echo "[INFO] Starting in NVIDIA GPU mode"
    docker compose -f compose.yaml -f compose.gpu.yaml up -d
    ;;

  *)
    echo "[ERROR] Unknown mode: ${MODE}"
    echo "Usage: ./start.sh [auto|cpu|gpu]"
    exit 1
    ;;
esac
