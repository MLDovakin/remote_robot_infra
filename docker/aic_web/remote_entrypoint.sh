#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export AIC_RESULTS_DIR="${AIC_RESULTS_DIR:-/workspace/aic_results}"
export AIC_RUNS_DIR="${AIC_RUNS_DIR:-/workspace/aic_runs}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"

if [ -f "/opt/ros/${ROS_DISTRO:-}/setup.bash" ]; then
  # Expose ROS 2 Python packages and command-line tools to the web runner.
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
fi

mkdir -p "$AIC_RESULTS_DIR" "$AIC_RUNS_DIR"

ZENOH_PID=""
if [ "${RMW_IMPLEMENTATION:-}" = "rmw_zenoh_cpp" ] && command -v ros2 >/dev/null 2>&1; then
  ros2 run rmw_zenoh_cpp rmw_zenohd >/tmp/rmw_zenohd.log 2>&1 &
  ZENOH_PID=$!
fi

Xvfb "$DISPLAY" -screen 0 "${AIC_SCREEN:-1600x900x24}" -nolisten tcp &
XVFB_PID=$!

sleep 1

fluxbox >/tmp/fluxbox.log 2>&1 &
FLUXBOX_PID=$!

x11vnc -display "$DISPLAY" -forever -shared -nopw -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
X11VNC_PID=$!

websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
NOVNC_PID=$!

cleanup() {
  if [ -n "$ZENOH_PID" ]; then
    kill "$ZENOH_PID" 2>/dev/null || true
  fi
  kill "$NOVNC_PID" "$X11VNC_PID" "$FLUXBOX_PID" "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

exec python3 /opt/aic_web/server.py
