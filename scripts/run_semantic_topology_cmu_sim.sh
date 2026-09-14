#!/usr/bin/env bash
# Run CMU's simulator/local planner with our frozen semantic-topology global node.
# No tare_planner/explore_*.launch is started: /way_point belongs to our node.
set -Eeuo pipefail

WORLD="${1:-indoor}"
MODE="${2:-shadow}" # shadow | closed_loop
RUN_ID="${3:-$(date +%Y%m%d_%H%M%S)_semantic_${WORLD}_${MODE}}"
RUN_SECONDS="${4:-600}"
KEEP_CONTAINER="${KEEP_CONTAINER:-0}"
RVIZ="${RVIZ:-false}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
case "$WORLD" in campus|forest|garage|indoor|tunnel|unseen_mine|external_cave|subtgraph_operational_01|subtgraph_operational_02) ;; *) echo "unsupported world: $WORLD" >&2; exit 22 ;; esac
case "$MODE" in shadow|closed_loop) ;; *) echo "mode must be shadow or closed_loop" >&2; exit 22 ;; esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/results/semantic_topology_cmu_sim/$RUN_ID"
IMAGE="mtare-semantic-runtime:local"
CKPT="$ROOT_DIR/results/semantic_bottleneck_role/20260807_polar_semantic_v5_final/semantic_bottleneck_underground_deployment_epoch70.pt"
CONTAINER="semantic_topology_${WORLD}_${RANDOM}"
[[ -f "$CKPT" ]] || { echo "checkpoint missing: $CKPT" >&2; exit 23; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite $OUT_DIR" >&2; exit 24; }
mkdir -p "$OUT_DIR"/{logs,artifacts,bags}

cleanup() {
  [[ "$KEEP_CONTAINER" == 1 ]] || docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$CONTAINER" --privileged --gpus all \
  -e DISPLAY="${DISPLAY:-:0}" -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /etc/localtime:/etc/localtime:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$ROOT_DIR:/workspace:ro" -v "$OUT_DIR:/output:rw" \
  "$IMAGE" bash -lc 'sleep infinity' >/dev/null
printf '%s\n' "$CONTAINER" >"$OUT_DIR/container_name.txt"

run() { docker exec "$CONTAINER" bash -lc "$1"; }
ROS='source /opt/ros/noetic/setup.bash'
SIM="$ROS && source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash"

SYSTEM_LAUNCH="system_${WORLD}.launch"
SYSTEM_EXTRA_ARGS=""
if [[ "$WORLD" == unseen_mine ]]; then
  run "cp /workspace/integration/cmu_unseen_mine/unseen_mine.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/unseen_mine.world"
  SYSTEM_LAUNCH="system_tunnel.launch"
  SYSTEM_EXTRA_ARGS="world_name:=unseen_mine"
  cp "$ROOT_DIR/integration/cmu_unseen_mine/protocol.json" "$OUT_DIR/artifacts/held_out_world_protocol.json"
elif [[ "$WORLD" == external_cave ]]; then
  run "cp /workspace/external/gazebo_cave_world/worlds/cave_world.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/external_cave.world"
  SYSTEM_LAUNCH="system.launch"
  SYSTEM_EXTRA_ARGS="world_name:=external_cave vehicleX:=10 vehicleY:=-21 vehicleYaw:=0"
  cp "$ROOT_DIR/integration/external_cave/protocol.json" "$OUT_DIR/artifacts/held_out_world_protocol.json"
elif [[ "$WORLD" == subtgraph_operational_01 ]]; then
  run "cp /workspace/integration/subtgraph_operational_01/subtgraph_operational_01.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/subtgraph_operational_01.world"
  SYSTEM_LAUNCH="system.launch"
  SYSTEM_EXTRA_ARGS="world_name:=subtgraph_operational_01 vehicleX:=0 vehicleY:=40 terrainZ:=62.2 vehicleYaw:=0"
  cp "$ROOT_DIR/integration/subtgraph_operational_01/protocol.json" "$OUT_DIR/artifacts/held_out_world_protocol.json"
elif [[ "$WORLD" == subtgraph_operational_02 ]]; then
  run "cp /workspace/integration/subtgraph_operational_02/subtgraph_operational_02.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/subtgraph_operational_02.world"
  SYSTEM_LAUNCH="system.launch"
  SYSTEM_EXTRA_ARGS="world_name:=subtgraph_operational_02 vehicleX:=0 vehicleY:=40 terrainZ:=62.2 vehicleYaw:=0"
  cp "$ROOT_DIR/integration/subtgraph_operational_02/protocol.json" "$OUT_DIR/artifacts/held_out_world_protocol.json"
fi

run "$ROS && roscore >/output/logs/roscore.log 2>&1 &" 
sleep 4
if [[ "$WORLD" == external_cave ]]; then
  SIM="$SIM && export GAZEBO_MODEL_PATH=/workspace/external/gazebo_cave_world/worlds/models:\${GAZEBO_MODEL_PATH:-}"
fi
run "$SIM && roslaunch --wait vehicle_simulator $SYSTEM_LAUNCH $SYSTEM_EXTRA_ARGS robot_id:=0 rviz:=false vis_tools:=$RVIZ >/output/logs/simulator.log 2>&1 &"
sleep 18
run "$ROS && timeout 30 rostopic echo -n 1 /registered_scan/header" >"$OUT_DIR/artifacts/registered_scan_header.txt"
run "$ROS && timeout 30 rostopic echo -n 1 /state_estimation_at_scan/header" >"$OUT_DIR/artifacts/state_at_scan_header.txt"
run "$ROS && rostopic list" >"$OUT_DIR/artifacts/topics.txt"
run "python3 -c 'import torch; print(torch.__version__)'" >"$OUT_DIR/artifacts/torch_version.txt"
# Record the exact two streams used by the original TARE baseline.  This is
# required for trajectory and scan-coverage comparisons on equal time windows.
run "$ROS; nohup rosbag record --lz4 -O /output/bags/${WORLD}.bag /registered_scan /state_estimation_at_scan >/output/logs/rosbag.log 2>&1 </dev/null & echo \$! >/output/artifacts/rosbag.pid"

NODE_ARGS="--checkpoint /workspace/results/semantic_bottleneck_role/20260807_polar_semantic_v5_final/semantic_bottleneck_underground_deployment_epoch70.pt --output /output/artifacts --device cpu"
[[ "$MODE" == shadow ]] && NODE_ARGS="$NODE_ARGS --shadow"
run "$ROS && export PYTHONPATH=/workspace:\${PYTHONPATH:-} && python3 /workspace/learning/structural_learning/tools/semantic_topology_global_node.py $NODE_ARGS >/output/logs/semantic_global.log 2>&1 &"
if [[ "$RVIZ" == true ]]; then
  run "$ROS && nohup rviz -d /workspace/configs/semantic_topology_live.rviz >/output/logs/semantic_rviz.log 2>&1 </dev/null &"
fi
if [[ "$GAZEBO_GUI" == true ]]; then
  run "nohup gzclient >/output/logs/gzclient.log 2>&1 </dev/null &"
fi
sleep 30
run "$ROS && rosnode list" >"$OUT_DIR/artifacts/nodes.txt"
run "$ROS && timeout 15 rostopic hz /registered_scan -w 10" >"$OUT_DIR/artifacts/registered_scan_hz.txt" 2>&1 || true
sleep "$RUN_SECONDS"
run "if test -f /output/artifacts/rosbag.pid; then kill -INT \$(cat /output/artifacts/rosbag.pid) 2>/dev/null || true; fi"
sleep 8
test -f "$OUT_DIR/artifacts/topology_snapshot.json"
python3 - <<PY
import json
from pathlib import Path
p=Path('$OUT_DIR/artifacts/topology_snapshot.json')
j=json.loads(p.read_text())
Path('$OUT_DIR/summary.json').write_text(json.dumps({'status':'COMPLETED','status_scope':'execution_only','world':'$WORLD','mode':'$MODE','run_seconds':int('$RUN_SECONDS'),'container':'$CONTAINER','container_retained':bool(int('$KEEP_CONTAINER')),'topology':{'node_count':j['topology']['node_count'],'edge_count':j['topology']['edge_count']},'checkpoint':'$CKPT'},indent=2))
PY
echo "$OUT_DIR"
