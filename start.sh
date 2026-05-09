#!/usr/bin/env bash
set -e

export QGC_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')

echo "[INFO] QGC_HOST_IP=$QGC_HOST_IP"
docker compose up -d
