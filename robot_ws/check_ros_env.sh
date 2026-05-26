#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH

set +u
source /opt/ros/humble/setup.bash
source /root/gpufree-data/code/robot_ws/install/setup.bash
set -u

python3 - <<'PY'
import sys
import rclpy

print("python:", sys.version.split()[0])
print("executable:", sys.executable)
print("rclpy:", rclpy.__file__)
PY
