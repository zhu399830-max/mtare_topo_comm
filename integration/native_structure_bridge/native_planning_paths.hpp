#pragma once
// Evidence only. No native planner mutations, identity inference or ROS calls.
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace native_structure_bridge {
template <class Path> std::string planningPathJson(const Path& path) {
  std::ostringstream out; out << std::setprecision(17) << '[';
  bool first = true;
  for (const auto& node : path.nodes_) {
    if (!first) out << ',';
    first = false;
    out << "{\"xyz_m\":[";
    for (int i = 0; i < 3; ++i) {
      const double value = node.position_[i];
      if (!std::isfinite(value)) throw std::runtime_error("NONFINITE_NATIVE_PATH");
      if (i) out << ',';
      out << value;
    }
    out << "],\"native_node_type\":" << static_cast<int>(node.type_)
        << ",\"native_subspace_index\":" << node.global_subspace_index_ << '}';
  }
  out << ']'; return out.str();
}

template <class Path, class Point> std::string planningPathsFields(
    const Path& global, const Path& exploration, const Point& lookahead,
    bool updated, bool from_start) {
  // A recording bound, not path truncation or a planning constraint.
  const bool complete = global.nodes_.size() + exploration.nodes_.size() <= 16384;
  std::ostringstream out; out << std::setprecision(17)
      << ",\"path_schema\":\"native_planning_paths_v1\",\"coordinate_frame\":\"map\""
      << ",\"global_node_count\":" << global.nodes_.size()
      << ",\"exploration_node_count\":" << exploration.nodes_.size()
      << ",\"paths_complete\":" << (complete ? "true" : "false")
      << ",\"global_path\":" << (complete ? planningPathJson(global) : "null")
      << ",\"exploration_path\":" << (complete ? planningPathJson(exploration) : "null")
      << ",\"lookahead_xyz_m\":[";
  for (int i = 0; i < 3; ++i) {
    const double value = lookahead[i];
    if (!std::isfinite(value)) throw std::runtime_error("NONFINITE_NATIVE_LOOKAHEAD");
    if (i) out << ',';
    out << value;
  }
  out << "],\"lookahead_updated\":" << (updated ? "true" : "false")
      << ",\"follow_local_path_from_start\":" << (from_start ? "true" : "false")
      << ",\"evidence_scope\":\"native_planned_paths_not_executed_traversal\"";
  return out.str();
}
}  // namespace native_structure_bridge
