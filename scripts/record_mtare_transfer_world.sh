#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD=""
DURATION_SEC="300"
CONTAINER_IP="10.0.2.96"
RUN_ID=""
NETWORK=""
IMAGE="caochao/mtare-open-source@sha256:e67b37b6c8084cb6ab92d9faf48c6b2f77c64f8f79eee914a40b586883bc709e"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world) WORLD="$2"; shift 2 ;;
    --duration) DURATION_SEC="$2"; shift 2 ;;
    --container-ip) CONTAINER_IP="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 22 ;;
  esac
done

case "$WORLD" in
  campus|forest|garage|indoor|tunnel|unseen_mine|external_cave|subtgraph_operational_01) ;;
  *) echo "unsupported --world: $WORLD" >&2; exit 22 ;;
esac
[[ "$DURATION_SEC" =~ ^[0-9]+$ ]] && (( DURATION_SEC >= 30 )) || { echo "--duration must be an integer >= 30" >&2; exit 22; }
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${WORLD}}"
RUN_DIR="${ROOT_DIR}/results/mtare_transfer_recordings/${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to overwrite $RUN_DIR" >&2; exit 23; }
mkdir -p "$RUN_DIR"/{bags,logs,runtime,world_audit}
CONTAINER="mtare_transfer_${WORLD}_$$_${RANDOM}"
RUN_COMPLETE="false"
SIMULATION_LAUNCH="system_${WORLD}.launch"
SIMULATION_ARGS=""
PLANNER_LAUNCH="explore_${WORLD}.launch"
if [[ "$WORLD" == unseen_mine ]]; then
  SIMULATION_LAUNCH="system_tunnel.launch"
  SIMULATION_ARGS="world_name:=unseen_mine"
  PLANNER_LAUNCH="explore_tunnel.launch"
elif [[ "$WORLD" == external_cave ]]; then
  SIMULATION_LAUNCH="system.launch"
  SIMULATION_ARGS="world_name:=external_cave vehicleX:=10 vehicleY:=-21 vehicleYaw:=0"
  PLANNER_LAUNCH="explore_tunnel.launch"
elif [[ "$WORLD" == subtgraph_operational_01 ]]; then
  SIMULATION_LAUNCH="system.launch"
  SIMULATION_ARGS="world_name:=subtgraph_operational_01 vehicleX:=0 vehicleY:=40 terrainZ:=62.2 vehicleYaw:=0"
  PLANNER_LAUNCH="explore_tunnel.launch"
fi

cleanup() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    docker exec "$CONTAINER" bash -lc '
      if [[ -f /tmp/transfer_pids.tsv ]]; then
        while IFS=$'"'"'\t'"'"' read -r pid name; do
          [[ "$name" == "rosbag" ]] && kill -INT "$pid" 2>/dev/null || true
        done < /tmp/transfer_pids.tsv
        sleep 5
        tac /tmp/transfer_pids.tsv | while IFS=$'"'"'\t'"'"' read -r pid name; do kill -TERM "$pid" 2>/dev/null || true; done
      fi' >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  if [[ "$RUN_COMPLETE" != "true" && ! -f "$RUN_DIR/status.json" ]]; then
    printf '{"status":"FAILED","world":"%s","reason":"see logs"}\n' "$WORLD" >"$RUN_DIR/status.json"
  fi
}
trap cleanup EXIT

cat >"$RUN_DIR/config.yaml" <<EOF
run_id: ${RUN_ID}
world: ${WORLD}
duration_sec: ${DURATION_SEC}
robot_count: 1
container_ip: ${CONTAINER_IP}
network: ${NETWORK}
image: ${IMAGE}
simulation_launch: vehicle_simulator/${SIMULATION_LAUNCH} ${SIMULATION_ARGS}
planner_launch: tare_planner/${PLANNER_LAUNCH}
topics:
  - /registered_scan
  - /state_estimation_at_scan
model_training_use: forbidden
seed_status: UNCONFIRMED_SIMULATION_RUNTIME_RANDOMNESS
EOF
date -Is >"$RUN_DIR/start_time.txt"
docker image inspect "$IMAGE" >"$RUN_DIR/runtime/docker_image.json"

NETWORK_ARGS=()
if [[ -n "$NETWORK" ]]; then
  NETWORK_ARGS=(--network "$NETWORK" --ip "$CONTAINER_IP")
fi
docker run -d --name "$CONTAINER" --privileged --gpus all "${NETWORK_ARGS[@]}" \
  -e DISPLAY="${DISPLAY:-}" -e QT_X11_NO_MITSHM=1 \
  -v /etc/localtime:/etc/localtime:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$ROOT_DIR:/workspace:ro" -v "$RUN_DIR:/data:rw" "$IMAGE" bash -lc 'sleep infinity' >/dev/null

if [[ "$WORLD" == unseen_mine ]]; then
  docker exec "$CONTAINER" bash -lc 'cp /workspace/integration/cmu_unseen_mine/unseen_mine.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/unseen_mine.world'
  cp "$ROOT_DIR/integration/cmu_unseen_mine/protocol.json" "$RUN_DIR/world_audit/held_out_world_protocol.json"
elif [[ "$WORLD" == external_cave ]]; then
  docker exec "$CONTAINER" bash -lc 'cp /workspace/external/gazebo_cave_world/worlds/cave_world.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/external_cave.world'
  cp "$ROOT_DIR/integration/external_cave/protocol.json" "$RUN_DIR/world_audit/held_out_world_protocol.json"
elif [[ "$WORLD" == subtgraph_operational_01 ]]; then
  docker exec "$CONTAINER" bash -lc 'cp /workspace/integration/subtgraph_operational_01/subtgraph_operational_01.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/subtgraph_operational_01.world'
  cp "$ROOT_DIR/integration/subtgraph_operational_01/protocol.json" "$RUN_DIR/world_audit/held_out_world_protocol.json"
fi

start_bg() {
  local name="$1"
  local command="$2"
  local pid
  pid="$(docker exec "$CONTAINER" bash -lc "nohup bash -lc $(printf '%q' "$command") >'/data/logs/${name}.stdout' 2>'/data/logs/${name}.stderr' & echo \$!")"
  printf '%s\t%s\n' "$pid" "$name" | docker exec -i "$CONTAINER" bash -lc 'cat >> /tmp/transfer_pids.tsv'
  printf '%s\t%s\n' "$pid" "$name" >>"$RUN_DIR/runtime/pids.tsv"
}

ROS_BASE="source /opt/ros/noetic/setup.bash"
SIM_ENV="$ROS_BASE && source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash"
if [[ "$WORLD" == external_cave ]]; then
  SIM_ENV="$SIM_ENV && export GAZEBO_MODEL_PATH=/workspace/external/gazebo_cave_world/worlds/models:\${GAZEBO_MODEL_PATH:-}"
fi
PLANNER_ENV="$ROS_BASE && source /home/docker-user/mtare/tare_system/devel/setup.bash"
start_bg roscore "$ROS_BASE && roscore"
for _ in $(seq 1 30); do
  docker exec "$CONTAINER" bash -lc "$ROS_BASE && rosnode list >/dev/null 2>&1" && break
  sleep 1
done
docker exec "$CONTAINER" bash -lc "$ROS_BASE && rosnode list >/dev/null 2>&1" || { echo ROSCORE_NOT_READY >&2; exit 24; }

start_bg rosbag "$ROS_BASE && rosbag record --lz4 -O /data/bags/${WORLD}.bag /registered_scan /state_estimation_at_scan"
sleep 2
start_bg simulator "$SIM_ENV && roslaunch --wait vehicle_simulator $SIMULATION_LAUNCH $SIMULATION_ARGS robot_id:=0 rviz:=false vis_tools:=false"
sleep 15

docker exec "$CONTAINER" bash -lc "$ROS_BASE && timeout 45 rostopic echo -n 1 /registered_scan/header" >"$RUN_DIR/world_audit/registered_scan_header.txt" 2>&1 || { echo REGISTERED_SCAN_NO_MESSAGES >&2; exit 25; }
docker exec "$CONTAINER" bash -lc "$ROS_BASE && timeout 30 rostopic echo -n 1 /state_estimation_at_scan/header" >"$RUN_DIR/world_audit/state_at_scan_header.txt" 2>&1 || { echo STATE_AT_SCAN_NO_MESSAGES >&2; exit 25; }
start_bg planner "$PLANNER_ENV && roslaunch --wait tare_planner $PLANNER_LAUNCH robot_id:=0 robot_num:=1 test_id:=0001 rviz:=false rosbag_record:=false"
sleep 5

docker exec "$CONTAINER" bash -lc "$ROS_BASE && rostopic list" >"$RUN_DIR/runtime/rostopic_list.txt" 2>"$RUN_DIR/runtime/rostopic_list.stderr"
grep -Fxq /registered_scan "$RUN_DIR/runtime/rostopic_list.txt" || { echo REGISTERED_SCAN_MISSING >&2; exit 25; }
grep -Fxq /state_estimation_at_scan "$RUN_DIR/runtime/rostopic_list.txt" || { echo STATE_AT_SCAN_MISSING >&2; exit 25; }

docker exec "$CONTAINER" bash -lc "$ROS_BASE && timeout 12 rostopic hz /registered_scan" >"$RUN_DIR/runtime/registered_scan_hz.txt" 2>&1 || true
if [[ "$WORLD" == unseen_mine ]]; then
  docker exec "$CONTAINER" bash -lc "sha256sum /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/unseen_mine.world" >"$RUN_DIR/world_audit/assets.sha256"
elif [[ "$WORLD" == external_cave ]]; then
  docker exec "$CONTAINER" bash -lc "sha256sum /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/external_cave.world /workspace/external/gazebo_cave_world/worlds/models/cave_world/meshes/cave_world.obj" >"$RUN_DIR/world_audit/assets.sha256"
elif [[ "$WORLD" == subtgraph_operational_01 ]]; then
  docker exec "$CONTAINER" bash -lc "sha256sum /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/subtgraph_operational_01.world /workspace/external/SubTGraph/benchmark/operational/01/subtgraph.obj" >"$RUN_DIR/world_audit/assets.sha256"
else
  docker exec "$CONTAINER" bash -lc "sha256sum /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/world/${WORLD}.world /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/mesh/${WORLD}/preview/pointcloud.ply" >"$RUN_DIR/world_audit/assets.sha256"
fi
docker exec "$CONTAINER" bash -lc "cat /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/launch/$SIMULATION_LAUNCH" >"$RUN_DIR/world_audit/system_${WORLD}.launch"
docker exec "$CONTAINER" bash -lc "cat /home/docker-user/mtare/tare_system/src/tare_planner/launch/$PLANNER_LAUNCH" >"$RUN_DIR/world_audit/explore_${WORLD}.launch"

sleep "$DURATION_SEC"
docker exec "$CONTAINER" bash -lc '
  while IFS=$'"'"'\t'"'"' read -r pid name; do
    [[ "$name" == "rosbag" ]] && kill -INT "$pid" 2>/dev/null || true
  done < /tmp/transfer_pids.tsv'
sleep 8
docker exec "$CONTAINER" bash -lc "$ROS_BASE && rosbag info /data/bags/${WORLD}.bag" >"$RUN_DIR/world_audit/rosbag_info.txt" 2>&1
docker exec "$CONTAINER" bash -lc "$ROS_BASE && rosbag info --yaml /data/bags/${WORLD}.bag" >"$RUN_DIR/world_audit/rosbag_info.yaml" 2>&1
docker exec "$CONTAINER" bash -lc "$ROS_BASE && rosnode list" >"$RUN_DIR/runtime/rosnode_list_end.txt" 2>&1 || true
date -Is >"$RUN_DIR/end_time.txt"
cat >"$RUN_DIR/status.json" <<EOF
{"status":"RECORDED","world":"${WORLD}","bag":"bags/${WORLD}.bag"}
EOF
RUN_COMPLETE="true"
echo "$RUN_DIR"
