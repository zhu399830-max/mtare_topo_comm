#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_RESOURCE_PATH=/parity
world_path="$1"
output_path="$2"
sensor_count="$3"
log_dir="$4"
mkdir -p "$log_dir"
cleanup() {
  if [[ -n "${gazebo_pid:-}" ]]; then kill "$gazebo_pid" 2>/dev/null || true; fi
  if [[ -n "${ros_pid:-}" ]]; then kill "$ros_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
roscore >"$log_dir/roscore.log" 2>&1 &
ros_pid=$!
for _ in $(seq 1 100); do
  if rostopic list >/dev/null 2>&1; then break; fi
  sleep 0.1
done
gzserver --verbose -s libgazebo_ros_api_plugin.so "$world_path" >"$log_dir/gzserver.log" 2>&1 &
gazebo_pid=$!
python3 /workspace/tools/v3/gazebo/collect_fixed_lidar_scans.py --output "$output_path" --sensor-count "$sensor_count" --warmup 2 --scans 3 --timeout 120
