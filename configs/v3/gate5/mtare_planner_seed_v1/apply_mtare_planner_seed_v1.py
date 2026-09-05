#!/usr/bin/env python3
"""Apply the hash-bound M-TARE planner-seed patch to a source tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


EXPECTED = {
    "CMakeLists.txt": "a3a5ce35b433e4f5d4b0876ae20f9f96e16a825cfd8cbef668af473520bcc22d",
    "src/tare_planner_node/tare_planner_node.cpp": "73c9d9d1cfb5c9aeaca30e27f79f6e93b5d5c78c7f4e9285373c35436097bd91",
    "src/local_coverage_planner/local_coverage_planner.cpp": "b9b6d7206bd948cd2f8abf84b01cf23f2754f16fef3bcb6dff27d94d0aad1a92",
    "src/grid_world/grid_world.cpp": "1e41fb08d6f6d7ca00a60d553f8020b1bc0ac5bef8c6cbe906cb012cb86d6ac3",
    "src/utils/PBS_utils.cpp": "6a96b47fcd787969c776e107190e0ed46f85ce77e1d5d696b8328716a57738db",
    "src/utils/CBS_utils.cpp": "0dad444b85f0fa5ef82186f78c1f962485449c76a21bea7edb9f4468676074f8",
}

HEADER = """#pragma once

#include <cstdint>
#include <random>

namespace planner_seed_ns
{
constexpr std::uint32_t kLocalCoverageViewpointStream = 0x4c435650U;
constexpr std::uint32_t kGridWorldRobotSubsetStream = 0x47575253U;
constexpr std::uint32_t kPBSTieBreakStream = 0x50425354U;
constexpr std::uint32_t kCBSTieBreakStream = 0x43425354U;

void Initialize(std::uint32_t seed);
bool IsInitialized();
std::uint32_t BaseSeed();
std::mt19937& Generator(std::uint32_t stream_id);
bool UniformBinary(std::uint32_t stream_id);
}
"""

SOURCE = """#include <utils/planner_seed.h>

#include <map>
#include <mutex>
#include <stdexcept>

namespace planner_seed_ns
{
namespace
{
std::mutex seed_mutex;
bool initialized = false;
std::uint32_t base_seed = 0U;
std::map<std::uint32_t, std::mt19937> generators;
}

void Initialize(std::uint32_t seed)
{
  std::lock_guard<std::mutex> guard(seed_mutex);
  if (initialized)
  {
    throw std::logic_error("planner seed was initialized more than once");
  }
  base_seed = seed;
  initialized = true;
  generators.clear();
}

bool IsInitialized()
{
  std::lock_guard<std::mutex> guard(seed_mutex);
  return initialized;
}

std::uint32_t BaseSeed()
{
  std::lock_guard<std::mutex> guard(seed_mutex);
  if (!initialized)
  {
    throw std::logic_error("planner seed has not been initialized");
  }
  return base_seed;
}

std::mt19937& Generator(std::uint32_t stream_id)
{
  std::lock_guard<std::mutex> guard(seed_mutex);
  if (!initialized)
  {
    throw std::logic_error("planner seed has not been initialized");
  }
  auto found = generators.find(stream_id);
  if (found == generators.end())
  {
    std::seed_seq sequence{ base_seed, stream_id, 0x4d544152U, 0x45534545U };
    found = generators.emplace(stream_id, std::mt19937(sequence)).first;
  }
  return found->second;
}

bool UniformBinary(std::uint32_t stream_id)
{
  std::uniform_int_distribution<int> distribution(0, 1);
  return distribution(Generator(stream_id)) != 0;
}
}
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one replacement in {relative}, found {count}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_mtare_planner_seed_v1.py TARE_PLANNER_ROOT")
    root = Path(sys.argv[1]).resolve()
    observed = {relative: digest(root / relative) for relative in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError(f"frozen source hash drift: {observed}")

    header = root / "include/utils/planner_seed.h"
    source = root / "src/utils/planner_seed.cpp"
    if header.exists() or source.exists():
        raise FileExistsError("planner seed files already exist; refusing a repeated patch")
    header.write_text(HEADER, encoding="utf-8")
    source.write_text(SOURCE, encoding="utf-8")

    replace_once(
        root,
        "CMakeLists.txt",
        "target_link_libraries(misc_utils ${catkin_LIBRARIES} ${PCL_LIBRARIES})\n",
        "target_link_libraries(misc_utils ${catkin_LIBRARIES} ${PCL_LIBRARIES})\n\n"
        "add_library(planner_seed src/utils/planner_seed.cpp)\n"
        "add_dependencies(planner_seed ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})\n"
        "target_link_libraries(planner_seed ${catkin_LIBRARIES})\n",
    )
    for target, old_suffix in (
        ("local_coverage_planner", "${catkin_LIBRARIES} viewpoint_manager"),
        ("grid_world", "${catkin_LIBRARIES} viewpoint_manager tsp_solver keypose_graph exploration_path knowledge_base multi_robot_exploration_manager robot rendezvous_manager time_budget_manager pursuit_mdp TOPwTVR TOPwTVR_utils"),
        ("PBS_utils", "${catkin_LIBRARIES}"),
        ("CBS_utils", "${catkin_LIBRARIES}"),
        ("tare_planner_node", "${catkin_LIBRARIES} sensor_coverage_planner_ground"),
    ):
        replace_once(
            root,
            "CMakeLists.txt",
            f"target_link_libraries({target} {old_suffix})",
            f"target_link_libraries({target} {old_suffix} planner_seed)",
        )

    replace_once(
        root,
        "src/tare_planner_node/tare_planner_node.cpp",
        '#include <ros/ros.h>\n#include "sensor_coverage_planner/sensor_coverage_planner_ground.h"',
        '#include <ros/ros.h>\n#include <cstdint>\n\n#include "sensor_coverage_planner/sensor_coverage_planner_ground.h"\n#include "utils/planner_seed.h"',
    )
    replace_once(
        root,
        "src/tare_planner_node/tare_planner_node.cpp",
        '  ros::NodeHandle private_node_handle("~");\n\n  sensor_coverage_planner_3d_ns::SensorCoveragePlanner3D tare_planner',
        '  ros::NodeHandle private_node_handle("~");\n\n'
        '  int planner_seed = -1;\n'
        '  if (!private_node_handle.getParam("planner_seed", planner_seed) || planner_seed < 0)\n'
        '  {\n'
        '    ROS_FATAL("A non-negative private parameter \'~planner_seed\' is required for reproducible execution");\n'
        '    return 2;\n'
        '  }\n'
        '  planner_seed_ns::Initialize(static_cast<std::uint32_t>(planner_seed));\n'
        '  ROS_INFO_STREAM("M-TARE deterministic planner seed: " << planner_seed_ns::BaseSeed());\n\n'
        '  sensor_coverage_planner_3d_ns::SensorCoveragePlanner3D tare_planner',
    )
    replace_once(
        root,
        "src/local_coverage_planner/local_coverage_planner.cpp",
        '#include "local_coverage_planner/local_coverage_planner.h"',
        '#include "local_coverage_planner/local_coverage_planner.h"\n#include "utils/planner_seed.h"',
    )
    replace_once(
        root,
        "src/local_coverage_planner/local_coverage_planner.cpp",
        "  std::random_device rd;\n  std::mt19937 gen(rd());",
        "  std::mt19937& gen = planner_seed_ns::Generator(planner_seed_ns::kLocalCoverageViewpointStream);",
    )
    replace_once(
        root,
        "src/grid_world/grid_world.cpp",
        "#include <utils/pursuit_mdp.h>",
        "#include <utils/pursuit_mdp.h>\n#include <utils/planner_seed.h>",
    )
    replace_once(
        root,
        "src/grid_world/grid_world.cpp",
        "  std::random_device rd;\n  std::mt19937 gen(rd());\n  int32_t min_sample",
        "  std::mt19937& gen = planner_seed_ns::Generator(planner_seed_ns::kGridWorldRobotSubsetStream);\n  int32_t min_sample",
    )
    replace_once(
        root,
        "src/grid_world/grid_world.cpp",
        "{\n  std::random_device rd;\n  std::mt19937 gen(rd());\n  int node_num = robot_route.size();",
        "{\n  int node_num = robot_route.size();",
    )
    for relative, namespace, stream in (
        ("src/utils/PBS_utils.cpp", "PBS", "kPBSTieBreakStream"),
        ("src/utils/CBS_utils.cpp", "CBS", "kCBSTieBreakStream"),
    ):
        replace_once(root, relative, f"#include <utils/{namespace}_utils.h>", f"#include <utils/{namespace}_utils.h>\n#include <utils/planner_seed.h>")
        replace_once(root, relative, "  return rand() % 2;", f"  return planner_seed_ns::UniformBinary(planner_seed_ns::{stream});")

    leftovers = []
    for path in (root / "src").rglob("*"):
        if path.suffix not in {".cpp", ".h"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "std::random_device" in text or "rand()" in text:
            leftovers.append(str(path.relative_to(root)))
    if leftovers:
        raise RuntimeError(f"uncontrolled random sources remain: {leftovers}")
    print(json.dumps({"status": "PATCHED", "source_hashes": observed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
