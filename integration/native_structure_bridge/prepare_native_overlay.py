#!/usr/bin/env python3
"""Create a new two-source overlay for the exact audited M-TARE image.

This is a mechanical source transform, never an in-place installer, experiment,
ROS launch, or world reader. Build/recording authority remains with the runner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

IMAGE = "sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c"
EXPECTED = {
    "src/grid_world/grid_world.cpp": "97c95395fd7de4ec811c767502411554f3d5677e6b88cc24479a814531666c66",
    "src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp": "84ea295196c7e250d6a126601695517277ac5aa9d9c0e3ebf5bcb4c0e427b6bf",
}

STATUS_RULES = {
    923: "global_neighborhood_or_connectivity_heuristic",
    963: "local_nontraversable_share_with_others",
    1026: "local_selected_viewpoint_unseen_to_exploring",
    1052: "local_residual_coverage_below_threshold",
    1067: "local_no_candidate_single_visit_no_graph_heuristic",
    1080: "local_no_candidate_near_robot_heuristic",
    1091: "local_selected_traversable_viewpoint_reclaim",
    1109: "local_high_residual_coverage_reopened",
    1138: "almost_covered_left_local_horizon_heuristic",
}


def annotate_status_rules(text):
    """Tag actual unchanged native branches; COVERED alone is not completion."""
    rows, found = [], set()
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        match = re.match(r"^(\s*)SetCellStatus\((i|cell_ind), CellStatus::\w+\);\s*$", line)
        if match:
            if number not in STATUS_RULES:
                raise ValueError("Unclassified native status callsite: " + str(number))
            found.add(number)
            details = '""'
            if number == 1052:
                details = ('",\\\"candidate_count\\\":" + std::to_string(candidate_count) + '
                    '",\\\"above_small_threshold_count\\\":" + std::to_string(above_small_threshold_count) + '
                    '",\\\"above_frontier_threshold_count\\\":" + std::to_string(above_frontier_threshold_count) + '
                    '",\\\"native_cell_exploring_to_covered_threshold\\\":" + std::to_string(kCellExploringToCoveredThr)')
            rows.append(match[1] + 'native_structure_bridge::bridge().setStatusReason("' + STATUS_RULES[number] + '", ' +
                        ('true' if number == 1052 else 'false') + ', ' + details + ');\n')
        rows.append(line)
    if found != set(STATUS_RULES):
        raise ValueError("Native status callsite inventory changed")
    return ''.join(rows)

GRID_HOOK = r'''
  // Native structure bridge V1: advisory permutation AFTER the native VRP.
  // No candidate/status/edge mutation. Special mission branches are untouched.
  if (native_structure_bridge::bridge().enabled())
  {
    if (robots.size() == 1 && kRobotID == 0 && !relay_comms_ &&
        !(kRendezvous && go_to_rendezvous_) && !return_home_ && !wait_ &&
        ordered_cell_indices.size() > 2)
    {
      try
      {
        std::map<int, int> advice_indices;
        int depot = GetCellInd(robots[kRobotID].in_comms_position_);
        advice_indices.emplace(depot, kRobotID);
        for (std::size_t i = 0; i < global_exploring_cell_indices_.size(); ++i)
          native_structure_bridge::require(advice_indices.emplace(
              global_exploring_cell_indices_[i], static_cast<int>(robots.size() + i)).second,
              "AMBIGUOUS_NATIVE_CELL_DM_BINDING");
        native_structure_bridge::Edges advice_edges;
        std::map<int, std::vector<double>> advice_positions;
        std::map<int, int> advice_statuses;
        for (const auto& from : advice_indices)
        {
          auto position = GetCellPosition(from.first);
          advice_positions[from.first] = {position.x, position.y, position.z};
          advice_statuses[from.first] = static_cast<int>(GetCellStatus(from.first));
          for (const auto& to : advice_indices)
          {
            const int native_cost = distance_matrices_no_comms[kRobotID][from.second][to.second];
            const int traversable_cost = distance_matrices_with_traversability[kRobotID][from.second][to.second];
            if (native_cost >= 0 && native_cost < misc_utils_ns::INF_DISTANCE &&
                traversable_cost >= 0 && traversable_cost < misc_utils_ns::INF_DISTANCE)
              advice_edges[{from.first, to.first}] = native_cost;
          }
        }
        auto advice_decision = native_structure_bridge::bridge().consider(
            ordered_cell_indices, global_exploring_cell_indices_, advice_edges, advice_positions,
            advice_statuses, {robot_position_.x, robot_position_.y, robot_position_.z});
        if (advice_decision.changed)
        {
          native_structure_bridge::require(advice_decision.advised_cost_units <= std::numeric_limits<int>::max(),
              "NATIVE_COST_CACHE_OVERFLOW");
          ordered_cell_indices = advice_decision.route;
          no_comms_global_plan_[kRobotID] = ordered_cell_indices;
          vrp_cost_.longest_route_length_ = static_cast<int>(advice_decision.advised_cost_units);
        }
      }
      catch (const std::exception& error)
      {
        native_structure_bridge::bridge().feedback("NATIVE_SNAPSHOT_REJECTED",
            ",\"reason\":" + native_structure_bridge::quote(error.what()));
        ROS_ERROR_STREAM("Native structure bridge retained original route: " << error.what());
      }
    }
    else native_structure_bridge::bridge().feedback("NATIVE_MISSION_BYPASS");
  }

'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Patch anchor is not unique: " + old[:100])
    return text.replace(old, new, 1)


def transformed_sources(source_root, *, planning_paths=False):
    result = {}
    for name, expected in EXPECTED.items():
        raw = (source_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Source SHA mismatch, no overlay created: " + name)
        result[name] = raw.decode()
    name = "src/grid_world/grid_world.cpp"
    text = replace_once(annotate_status_rules(result[name]), "#include <utils/planner_seed.h>\n",
        "#include <utils/planner_seed.h>\n#include <native_structure_bridge/native_route_ros.hpp>\n")
    text = replace_once(text,
        "    std::vector<int>& ordered_cell_indices, const std::unique_ptr<keypose_graph_ns::KeyposeGraph>& keypose_graph)\n{\n",
        "    std::vector<int>& ordered_cell_indices, const std::unique_ptr<keypose_graph_ns::KeyposeGraph>& keypose_graph)\n{\n"
        "  native_structure_bridge::bridge().beginNativeCycle();\n")
    text = replace_once(text, "  // Get the global exploration path\n", GRID_HOOK + "  // Get the global exploration path\n")
    text = replace_once(text, "  subspaces_->GetCell(cell_ind).SetStatus(status);\n",
        "  subspaces_->GetCell(cell_ind).SetStatus(status);\n"
        "  if (native_structure_bridge::bridge().enabled())\n  {\n"
        "    auto center = GetCellPosition(cell_ind);\n"
        "    native_structure_bridge::bridge().regionStatus(cell_ind, static_cast<int>(prev_cell_status),\n"
        "        static_cast<int>(status), robot_id, {center.x, center.y, center.z});\n  }\n")
    text = replace_once(text, "  robot_position_ = robot_position;\n  int robot_cell_ind = GetCellInd(robot_position_);\n",
        "  robot_position_ = robot_position;\n  int robot_cell_ind = GetCellInd(robot_position_);\n"
        "  if (native_structure_bridge::bridge().enabled() && robot_cell_ind >= 0)\n  {\n"
        "    auto center = GetCellPosition(robot_cell_ind);\n"
        "    native_structure_bridge::bridge().regionVisit(robot_cell_ind, cur_robot_cell_ind_,\n"
        "        {center.x, center.y, center.z}, {robot_position_.x, robot_position_.y, robot_position_.z});\n  }\n")
    result[name] = text
    name = "src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp"
    text = replace_once(result[name], "#include <tare_msgs/ExplorationInfo.h>\n",
        "#include <tare_msgs/ExplorationInfo.h>\n#include <native_structure_bridge/native_route_ros.hpp>\n")
    text = replace_once(text, "  pd_.Initialize(nh, nh_p);\n",
        "  pd_.Initialize(nh, nh_p);\n  native_structure_bridge::bridge().configure(nh, nh_p);\n")
    text = replace_once(text, "  pd_.robot_position_ = state_estimation_msg->pose.pose.position;\n",
        "  pd_.robot_position_ = state_estimation_msg->pose.pose.position;\n"
        "  native_structure_bridge::bridge().observePose(state_estimation_msg->header.stamp.toNSec(),\n"
        "      {pd_.robot_position_.x, pd_.robot_position_.y, pd_.robot_position_.z},\n"
        "      {state_estimation_msg->pose.pose.orientation.x, state_estimation_msg->pose.pose.orientation.y,\n"
        "       state_estimation_msg->pose.pose.orientation.z, state_estimation_msg->pose.pose.orientation.w},\n"
        "      state_estimation_msg->header.frame_id, state_estimation_msg->child_frame_id, pp_.sub_state_estimation_topic_);\n")
    text = replace_once(text, "  // For using LOAM interfaces\n  // RotatePointCloud<pcl::PointXYZ>(registered_scan_tmp);",
        "  native_structure_bridge::bridge().observeSource(registered_scan_msg->header.stamp.toNSec());\n\n"
        "  // For using LOAM interfaces\n  // RotatePointCloud<pcl::PointXYZ>(registered_scan_tmp);")
    text = replace_once(text, "  pcl::fromROSMsg(*free_paths_msg, *free_paths_tmp);\n",
        "  pcl::fromROSMsg(*free_paths_msg, *free_paths_tmp);\n"
        "  native_structure_bridge::bridge().feedback(\"FREE_PATHS\",\n"
        "      \",\\\"count\\\":\" + std::to_string(free_paths_tmp->points.size()));\n")
    text = replace_once(text, "  misc_utils_ns::Publish<geometry_msgs::PointStamped>(waypoint_pub_, waypoint, kWorldFrameID);\n",
        "  misc_utils_ns::Publish<geometry_msgs::PointStamped>(waypoint_pub_, waypoint, kWorldFrameID);\n"
        "  native_structure_bridge::bridge().feedback(\"WAYPOINT\",\n"
        "      \",\\\"xyz_m\\\":\" + native_structure_bridge::array(std::vector<double>{\n"
        "          waypoint.point.x, waypoint.point.y, waypoint.point.z}));\n"
        "  if (native_structure_bridge::bridge().enabled())\n  {\n"
        "    int region_id = pd_.grid_world_->GetCellInd(waypoint.point);\n"
        "    if (region_id >= 0)\n    {\n"
        "      auto center = pd_.grid_world_->GetCellPosition(region_id);\n"
        "      native_structure_bridge::bridge().regionDispatch(region_id,\n"
        "          static_cast<int>(pd_.grid_world_->GetCellStatus(region_id)), {center.x, center.y, center.z},\n"
        "          {waypoint.point.x, waypoint.point.y, waypoint.point.z},\n"
        "          {pd_.robot_position_.x, pd_.robot_position_.y, pd_.robot_position_.z});\n"
        "    }\n    else native_structure_bridge::bridge().feedback(\"WAYPOINT_OUTSIDE_NATIVE_GRID\");\n  }\n")
    text = replace_once(text, "    PublishExplorationState();\n",
        "    PublishExplorationState();\n"
        "    native_structure_bridge::bridge().feedback(\"NATIVE_EXECUTION_STATE\",\n"
        "        \",\\\"native_finish\\\":\" + std::string(exploration_finished_ ? \"true\" : \"false\") +\n"
        "        \",\\\"at_home\\\":\" + std::string(at_home_ ? \"true\" : \"false\") +\n"
        "        \",\\\"global_route\\\":\" + native_structure_bridge::array(global_cell_tsp_order) +\n"
        "        \",\\\"local_viewpoint_count\\\":\" + std::to_string(viewpoint_count));\n")
    if planning_paths:
        text = replace_once(text, '#include <native_structure_bridge/native_route_ros.hpp>\n',
            '#include <native_structure_bridge/native_route_ros.hpp>\n'
            '#include <native_structure_bridge/native_planning_paths.hpp>\n')
        anchor = '        GetLookAheadPoint(pd_.exploration_path_, global_path, pd_.lookahead_point_, follow_local_path_from_start);\n'
        text = replace_once(text, anchor, anchor + r'''
    // Record actual native path objects; do not repair or use the by-value
    // GetNextGlobalSubspace output as a direction/task identity.
    if (native_structure_bridge::bridge().enabled())
    {
      try {
        native_structure_bridge::bridge().feedback("NATIVE_PLANNING_PATHS",
            native_structure_bridge::planningPathsFields(global_path, pd_.exploration_path_,
                pd_.lookahead_point_, lookahead_point_update_, follow_local_path_from_start));
      } catch (const std::exception& error) {
        native_structure_bridge::bridge().feedback("NATIVE_PLANNING_PATHS_REJECTED",
            ",\"reason\":" + native_structure_bridge::quote(error.what()));
      }
    }
''')
    result[name] = text
    return result


def prepare(source_root, output_root, *, planning_paths=False):
    if output_root.exists():
        raise ValueError("Overlay target must not exist")
    replacements = transformed_sources(source_root, planning_paths=planning_paths)
    own = Path(__file__).resolve().parent
    for filename in ["native_route_advice.hpp", "native_route_ros.hpp"]:
        replacements["include/native_structure_bridge/" + filename] = (own / filename).read_text()
    if planning_paths:
        replacements['include/native_structure_bridge/native_planning_paths.hpp'] = (own / 'native_planning_paths.hpp').read_text()
    # A bulk mechanical rewrite into a NEW overlay only; original files remain read-only.
    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, content in replacements.items():
        target = output_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        outputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = dict(schema_version="native_structure_source_overlay_v1", source_image=IMAGE,
        original_sha256=EXPECTED, output_sha256=outputs, source_originals_modified=False,
        local_planners_modified=False, extra_waypoint_publisher=False,
        native_bug_GetNextGlobalSubspace_modified=False, simulation_executed=False)
    if planning_paths:
        manifest.update(planning_path_evidence='native_planning_paths_v1', planned_paths_are_executed_edges=False)
    (output_root / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--planning-paths", action="store_true", help="Optional evidence only; default preserves old overlay")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output_root, planning_paths=args.planning_paths), indent=2))
