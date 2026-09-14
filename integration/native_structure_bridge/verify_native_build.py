#!/usr/bin/env python3
"""Compile the frozen native overlay and synthetic contracts, never start ROS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from prepare_native_overlay import IMAGE

SHELL = r'''
set -e
python3 /bridge/prepare_native_overlay.py --source-root /home/docker-user/mtare/tare_system/src/tare_planner --output-root /tmp/native-structure-overlay
cp /tmp/native-structure-overlay/src/grid_world/grid_world.cpp /home/docker-user/mtare/tare_system/src/tare_planner/src/grid_world/grid_world.cpp
cp /tmp/native-structure-overlay/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp /home/docker-user/mtare/tare_system/src/tare_planner/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp
cp -r /tmp/native-structure-overlay/include/native_structure_bridge /home/docker-user/mtare/tare_system/src/tare_planner/include/
source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/tare_system/devel/setup.bash
g++ --version
g++ -std=c++14 -Wall -Wextra -Werror /bridge/core_contract_test.cpp -o /tmp/core_test
/tmp/core_test
g++ -std=c++14 -I/opt/ros/noetic/include /bridge/ros_transport_contract_test.cpp -L/opt/ros/noetic/lib -Wl,-rpath-link,/opt/ros/noetic/lib -lroscpp -lroscpp_serialization -lrosconsole -lrostime -lcpp_common -o /tmp/transport_test
/tmp/transport_test
cmake --build /home/docker-user/mtare/tare_system/build --target tare_planner_node -- -j2
sha256sum /home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node
'''


def verify(output, *, planning_paths=False):
    output = output.resolve()
    if output.exists():
        raise ValueError("Build evidence directory must be new")
    bridge = Path(__file__).resolve().parent
    files = sorted(p for p in bridge.iterdir() if p.is_file())
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    observed_image = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if observed_image != IMAGE:
        raise ValueError("Frozen image mismatch")
    shell = SHELL
    if planning_paths:
        shell = shell.replace('--output-root /tmp/native-structure-overlay',
                              '--output-root /tmp/native-structure-overlay --planning-paths', 1)
        shell = shell.replace('cmake --build',
            'g++ -std=c++14 -Wall -Wextra -Werror /bridge/planning_paths_contract_test.cpp -o /tmp/path_test\n/tmp/path_test\ncmake --build', 1)
    command = ["docker", "run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "4",
        "--entrypoint", "/bin/bash", "-v", str(bridge) + ":/bridge:ro", IMAGE, "-c", shell]
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "source_manifest.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    start = time.monotonic()
    with (output / "build.log").open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == source_hashes[p.name] for p in files)
    raw_log = (output / "build.log").read_bytes()
    summary = dict(schema_version="native_structure_software_build_v1", image=IMAGE,
        exit_code=result.returncode, elapsed_s=time.monotonic() - start, source_unchanged=unchanged,
        full_native_target="tare_planner_node", ros_started=False, simulation_started=False,
        real_observations_read=0, cpp_core_and_disabled_transport_tested=(
            b'native_route_core_contract PASS' in raw_log and b'native_route_ros_transport PASS' in raw_log),
        status="SOFTWARE_BUILD_PASS" if result.returncode == 0 and unchanged else "SOFTWARE_BUILD_FAIL",
        log_sha256=hashlib.sha256(raw_log).hexdigest())
    if planning_paths:
        summary.update(planning_paths_enabled=True,
            planning_paths_contract_tested=b'native_planning_paths PASS' in raw_log)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if result.returncode or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planning-paths", action="store_true")
    args = parser.parse_args()
    verify(args.output, planning_paths=args.planning_paths)
