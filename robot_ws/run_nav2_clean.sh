#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin

set +u
source /opt/ros/humble/setup.bash
source /root/gpufree-data/code/robot_ws/install/local_setup.bash
set -u

exec /opt/ros/humble/bin/ros2 launch robot_navigation nav2_launch.py \
  mode:=isaac \
  use_sim_time:=true \
  map:=/root/gpufree-data/code/robot_ws/src/robot_navigation/maps/expff_map.yaml \
  params_file:=/root/gpufree-data/code/robot_ws/src/robot_navigation/config/nav2_params.yaml \
  "$@"
