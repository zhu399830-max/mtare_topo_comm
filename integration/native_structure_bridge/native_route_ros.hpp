#pragma once
// Existing dependencies only: ROS std_msgs and Boost property_tree (header-only).
#include "native_route_advice.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <deque>
#include <iomanip>
#include <memory>
#include <sstream>

namespace native_structure_bridge {
inline std::string nativeStatusName(int status) {
  // Frozen grid_world/grid_world.h CellStatus, not inferred semantic labels.
  static const char* names[] = {"UNSEEN", "EXPLORING", "COVERED", "COVERED_BY_OTHERS", "NOGO", "EXPLORING_BY_OTHERS"};
  return status >= 0 && status < 6 ? names[status] : "UNKNOWN_NATIVE_STATUS";
}
inline std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
template <class T> inline std::string array(const std::vector<T>& values) {
  std::ostringstream out; out << std::setprecision(17) << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']'; return out.str();
}
template <> inline std::string array(const std::vector<std::string>& values) {
  std::ostringstream out; out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  out << ']'; return out.str();
}
inline Cost integer(const boost::property_tree::ptree& node) {
  require(node.empty(), "INTEGER_SHAPE");
  const std::string value = node.data();
  require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "INTEGER_VALUE");
  std::size_t read = 0; Cost result = std::stoll(value, &read);
  require(read == value.size(), "INTEGER_VALUE"); return result;
}
inline std::vector<int> readIds(const boost::property_tree::ptree& node) {
  require(node.data().empty(), "ID_ARRAY");
  std::vector<int> result;
  for (const auto& item : node) {
    require(item.first.empty(), "ID_ARRAY"); Cost id = integer(item.second);
    require(id <= std::numeric_limits<int>::max(), "ID_RANGE"); result.push_back(static_cast<int>(id));
  }
  return result;
}
inline std::vector<std::string> readStrings(const boost::property_tree::ptree& node) {
  require(node.data().empty(), "STRING_ARRAY");
  std::vector<std::string> result;
  for (const auto& item : node) {
    require(item.first.empty() && item.second.empty(), "STRING_ARRAY"); result.push_back(item.second.data());
  }
  return result;
}
inline Advice parseAdvice(const std::string& text) {
  require(text.size() <= 1048576, "ADVICE_MESSAGE_SIZE");
  boost::property_tree::ptree tree; std::istringstream stream(text);
  boost::property_tree::read_json(stream, tree);
  const std::set<std::string> expected = {"schema_version", "epoch", "snapshot_stamp_ns",
      "source_frame_keys", "candidate_ids", "route", "claimed_cost_units", "evidence_refs"};
  std::set<std::string> found;
  for (const auto& item : tree) require(found.insert(item.first).second, "DUPLICATE_JSON_KEY");
  require(found == expected, "ADVICE_FIELDS");
  require(tree.get<std::string>("schema_version") == "native_route_advice_v1", "ADVICE_SCHEMA");
  require(tree.get_child("epoch").empty(), "ADVICE_EPOCH");
  Advice a; a.epoch = tree.get<std::string>("epoch");
  a.snapshot_stamp_ns = integer(tree.get_child("snapshot_stamp_ns"));
  a.source_frame_keys = readStrings(tree.get_child("source_frame_keys"));
  a.candidate_ids = readIds(tree.get_child("candidate_ids")); a.route = readIds(tree.get_child("route"));
  a.claimed_cost_units = integer(tree.get_child("claimed_cost_units"));
  a.evidence_refs = readStrings(tree.get_child("evidence_refs")); return a;
}
inline std::string snapshotJson(const Snapshot& s, const std::map<int, std::vector<double>>& positions,
    const std::map<int, int>& statuses, const std::vector<double>& robot_position) {
  std::ostringstream out; out << std::setprecision(17);
  out << "{\"schema_version\":\"native_route_snapshot_v1\",\"epoch\":" << quote(s.epoch)
      << ",\"stamp_ns\":" << s.stamp_ns << ",\"max_age_ns\":" << s.max_age_ns
      << ",\"max_extra_cost_units\":" << s.max_extra_cost_units
      << ",\"source_frame_keys\":" << array(s.source_frame_keys)
      << ",\"candidate_ids\":" << array(s.candidate_ids) << ",\"original_route\":" << array(s.original_route)
      << ",\"original_cost_units\":" << s.original_cost_units << ",\"native_edges\":[";
  bool first = true;
  for (const auto& edge : s.native_edges) {
    if (!first) out << ','; first = false;
    out << '[' << edge.first.first << ',' << edge.first.second << ',' << edge.second << ']';
  }
  out << "],\"candidate_positions_m\":{"; first = true;
  for (const auto& item : positions) {
    if (!first) out << ','; first = false;
    out << quote(std::to_string(item.first)) << ':' << array(item.second);
  }
  out << "},\"candidate_records\":["; first = true;
  for (int id : s.candidate_ids) {
    if (!first) out << ','; first = false;
    out << "{\"id\":" << id << ",\"position_xyz_m\":" << array(positions.at(id))
        << ",\"status\":" << statuses.at(id) << ",\"status_name\":" << quote(nativeStatusName(statuses.at(id))) << '}';
  }
  out << "],\"robot_position_xyz_m\":" << array(robot_position) << ",\"coordinate_frame\":\"map\""
      << ",\"cost_semantics\":\"native_integer_distance_matrix_including_offsets_not_metres\"}";
  return out.str();
}

class RosBridge {
 public:
  void configure(ros::NodeHandle& nh, ros::NodeHandle& nh_p) {
    nh_p.param<std::string>("native_structure_bridge_mode", mode_, "disabled");
    require(mode_ == "disabled" || mode_ == "shadow" || mode_ == "active", "BRIDGE_MODE");
    if (mode_ == "disabled") return;  // No wait, pubs, subscription or candidate changes.
    int wait_ms = -1, age_ms = -1;
    double extra_ratio = -1;
    require(nh_p.getParam("native_structure_wait_ms", wait_ms) && wait_ms >= 0 && wait_ms <= 100,
        "EXPLICIT_WAIT_POLICY_REQUIRED_MAX_100MS");
    require(nh_p.getParam("native_structure_max_age_ms", age_ms) && age_ms > 0,
        "EXPLICIT_AGE_POLICY_REQUIRED");
    require(nh_p.getParam("native_structure_max_extra_cost_ratio", extra_ratio) && extra_ratio == 0.10,
        "EXPLICIT_FROZEN_COST_RATIO_010_REQUIRED");
    wait_sec_ = wait_ms / 1000.; max_age_ns_ = Cost(age_ms) * 1000000;
    session_ = std::to_string(ros::WallTime::now().toNSec());
    snapshot_pub_ = nh.advertise<std_msgs::String>("native_structure/candidates", 2, false);
    decision_pub_ = nh.advertise<std_msgs::String>("native_structure/decision", 10, false);
    feedback_pub_ = nh.advertise<std_msgs::String>("native_structure/feedback", 1000, false);
    advice_nh_.reset(new ros::NodeHandle(nh)); advice_nh_->setCallbackQueue(&queue_);
    advice_sub_ = advice_nh_->subscribe("native_structure/advice", 2, &RosBridge::adviceCallback, this);
  }
  bool enabled() const { return mode_ != "disabled"; }
  void beginNativeCycle() {
    if (!enabled()) return;
    // Home/mission/empty-route cycles have no structural advice epoch. Never
    // attribute their later native waypoint to the preceding model decision.
    last_epoch_.clear(); expected_epoch_.clear(); received_.reset();
  }
  void observeSource(std::uint64_t stamp_ns) {
    if (!enabled()) return;
    const std::string key = "registered_scan:" + std::to_string(stamp_ns);
    if (!sources_.empty() && key == sources_.back()) return;
    sources_.push_back(key); if (sources_.size() > 5) sources_.pop_front();
    source_stamp_ns_ = stamp_ns;
  }
  void observePose(std::uint64_t stamp_ns, const std::vector<double>& xyz,
      const std::vector<double>& xyzw, const std::string& frame_id, const std::string& child_frame_id,
      const std::string& source_topic) {
    if (!enabled()) return;
    previous_pose_stamp_ns_ = pose_stamp_ns_; pose_stamp_ns_ = stamp_ns; ++pose_sequence_;
    pose_xyz_ = xyz; pose_xyzw_ = xyzw; pose_frame_ = frame_id; pose_child_frame_ = child_frame_id;
    pose_source_key_ = source_topic + ":" + std::to_string(stamp_ns);
    feedback("NATIVE_POSE", poseFields(xyz));
  }
  void regionVisit(int candidate_id, int previous_region_id, const std::vector<double>& center,
      const std::vector<double>& robot_position) {
    if (!enabled()) return;
    feedback("NATIVE_REGION_VISIT", ",\"candidate_id\":" + std::to_string(candidate_id) +
        ",\"previous_region_id\":" + std::to_string(previous_region_id) +
        ",\"position_xyz_m\":" + array(center) + poseFields(robot_position) + dispatchFields(candidate_id) +
        ",\"pose_evidence_kind\":\"native_state_estimation_cell_membership\",\"continuous_trajectory_verified\":false");
  }
  void regionDispatch(int candidate_id, int status, const std::vector<double>& center,
      const std::vector<double>& waypoint, const std::vector<double>& robot_position) {
    if (!enabled()) return;
    const std::string id = feedback("NATIVE_REGION_DISPATCH", ",\"candidate_id\":" + std::to_string(candidate_id) +
        ",\"position_xyz_m\":" + array(center) + ",\"waypoint_xyz_m\":" + array(waypoint) +
        ",\"status\":" + std::to_string(status) + ",\"status_name\":" + quote(nativeStatusName(status)) +
        poseFields(robot_position) + ",\"dispatch_scope\":\"published_native_waypoint_region_not_route_first_cell\"");
    dispatches_[candidate_id] = {last_epoch_, id};
  }
  void setStatusReason(const std::string& reason, bool coverage_rule_satisfied, const std::string& details = "") {
    if (!enabled()) return;
    status_reason_ = reason; status_coverage_ = coverage_rule_satisfied; status_details_ = details;
  }
  void regionStatus(int candidate_id, int previous_status, int status, int actor_robot_id,
      const std::vector<double>& center) {
    if (!enabled()) return;
    const std::string reason = status_reason_, details = status_details_;
    const bool coverage = status_coverage_;
    status_reason_ = "unspecified_native_caller"; status_details_.clear(); status_coverage_ = false;
    if (previous_status == status) return;
    feedback("NATIVE_REGION_STATUS", ",\"candidate_id\":" + std::to_string(candidate_id) +
        ",\"position_xyz_m\":" + array(center) + ",\"previous_status\":" + std::to_string(previous_status) +
        ",\"status\":" + std::to_string(status) + ",\"status_name\":" + quote(nativeStatusName(status)) +
        ",\"previous_status_name\":" + quote(nativeStatusName(previous_status)) +
        ",\"actor_robot_id\":" + std::to_string(actor_robot_id) +
        ",\"native_rule\":" + quote(reason) + ",\"native_coverage_rule_satisfied\":" + (coverage ? "true" : "false") +
        dispatchFields(candidate_id) + details + ",\"task_semantics\":\"native_region\",\"direction_complete\":false,\"physical_exploration_verified\":false");
  }
  Decision consider(const std::vector<int>& route, const std::vector<int>& candidate_ids,
      const Edges& edges, const std::map<int, std::vector<double>>& positions,
      const std::map<int, int>& statuses, const std::vector<double>& robot_position) {
    Decision decision; decision.route = route;
    if (!enabled()) { decision.reason = "DISABLED"; return decision; }
    Snapshot s; s.epoch = session_ + ":" + std::to_string(++epoch_counter_);
    s.stamp_ns = source_stamp_ns_; s.max_age_ns = max_age_ns_;
    s.source_frame_keys.assign(sources_.begin(), sources_.end());
    s.original_route = route; s.candidate_ids = candidate_ids; s.native_edges = edges;
    last_epoch_ = s.epoch;
    try {
      require(s.source_frame_keys.size() == 5, "FIVE_SOURCE_FRAMES_REQUIRED");
      s.original_cost_units = routeCost(route, edges);
      s.max_extra_cost_units = s.original_cost_units / 10;  // floor(0.10 * native cost), no floating rounding.
      validateSnapshot(s);
      expected_epoch_ = s.epoch; received_.reset(); parse_error_.clear();
      std_msgs::String message; message.data = snapshotJson(s, positions, statuses, robot_position);
      snapshot_pub_.publish(message);
      const auto deadline = ros::WallTime::now() + ros::WallDuration(wait_sec_);
      // Only the dedicated advice callback queue is serviced; native sensor /
      // execution callbacks cannot re-enter this snapshot while waiting.
      do {
        queue_.callAvailable(ros::WallDuration(0));
        if (received_ || !parse_error_.empty() || ros::WallTime::now() >= deadline) break;
        const double remaining = (deadline - ros::WallTime::now()).toSec();
        if (remaining > 0) queue_.callAvailable(ros::WallDuration(std::min(.005, remaining)));
      } while (ros::ok());
      decision = validate(s, received_.get(), static_cast<Cost>(ros::Time::now().toNSec()));
      if (!parse_error_.empty()) decision.reason = parse_error_;
      if (!received_ && parse_error_.empty()) decision.reason = "ADVICE_TIMEOUT";
      if (mode_ == "shadow" && decision.accepted) {
        decision.route = route; decision.changed = false; decision.reason = "SHADOW_" + decision.reason;
      }
    } catch (const std::exception& error) {
      decision.route = route; decision.accepted = false; decision.changed = false;
      decision.reason = std::string("SNAPSHOT_OR_TRANSPORT_ERROR:") + error.what();
      ROS_ERROR_STREAM("Native structure bridge preserves original route: " << decision.reason);
    }
    publishDecision(s, decision); expected_epoch_.clear(); received_.reset(); return decision;
  }
  std::string feedback(const std::string& event, const std::string& fields = "") {
    if (!enabled()) return "";
    const std::string evidence_id = session_ + ":event:" + std::to_string(++event_index_);
    std_msgs::String message; message.data = "{\"schema_version\":\"native_route_feedback_v1\",\"epoch\":" +
        quote(last_epoch_) + ",\"stamp_ns\":" + std::to_string(ros::Time::now().toNSec()) +
        ",\"session_id\":" + quote(session_) + ",\"event_index\":" + std::to_string(event_index_) +
        ",\"evidence_id\":" + quote(evidence_id) + ",\"source_frame_keys\":" +
        array(std::vector<std::string>(sources_.begin(), sources_.end())) +
        ",\"event\":" + quote(event) + fields + "}"; feedback_pub_.publish(message); return evidence_id;
  }
 private:
  std::string poseFields(const std::vector<double>& robot_position) const {
    return ",\"robot_position_xyz_m\":" + array(robot_position) +
        ",\"robot_orientation_xyzw\":" + array(pose_xyzw_) + ",\"pose_stamp_ns\":" + std::to_string(pose_stamp_ns_) +
        ",\"previous_pose_stamp_ns\":" + std::to_string(previous_pose_stamp_ns_) +
        ",\"pose_sequence\":" + std::to_string(pose_sequence_) + ",\"pose_source_key\":" + quote(pose_source_key_) +
        ",\"pose_frame_id\":" + quote(pose_frame_) + ",\"pose_child_frame_id\":" + quote(pose_child_frame_) +
        ",\"pose_binding_exact\":" + (pose_sequence_ > 0 && robot_position == pose_xyz_ ? "true" : "false");
  }
  std::string dispatchFields(int candidate_id) const {
    auto found = dispatches_.find(candidate_id);
    return ",\"dispatch_epoch\":" + quote(found == dispatches_.end() ? "" : found->second.first) +
        ",\"dispatch_evidence_id\":" + quote(found == dispatches_.end() ? "" : found->second.second);
  }
  void adviceCallback(const std_msgs::String::ConstPtr& message) {
    try {
      Advice advice = parseAdvice(message->data);
      if (advice.epoch != expected_epoch_) {
        feedback("REJECTED_ADVICE_EPOCH", ",\"advice_epoch\":" + quote(advice.epoch)); return;
      }
      if (!received_) received_.reset(new Advice(std::move(advice)));
    } catch (const std::exception& error) { parse_error_ = std::string("ADVICE_PARSE_ERROR:") + error.what(); }
  }
  void publishDecision(const Snapshot& s, const Decision& d) {
    std_msgs::String message;
    message.data = "{\"schema_version\":\"native_route_decision_v1\",\"epoch\":" + quote(s.epoch) +
        ",\"mode\":" + quote(mode_) + ",\"accepted\":" + (d.accepted ? "true" : "false") +
        ",\"changed\":" + (d.changed ? "true" : "false") + ",\"reason\":" + quote(d.reason) +
        ",\"original_route\":" + array(s.original_route) + ",\"route\":" + array(d.route) +
        ",\"original_cost_units\":" + std::to_string(d.original_cost_units) +
        ",\"advised_cost_units\":" + std::to_string(d.advised_cost_units) +
        ",\"source_frame_keys\":" + array(s.source_frame_keys) +
        ",\"advised_route\":" + array(received_ ? received_->route : std::vector<int>{}) +
        ",\"evidence_refs\":" + array(received_ ? received_->evidence_refs : std::vector<std::string>{}) +
        ",\"native_candidates_preserved\":true,\"native_edges_changed\":false}";
    decision_pub_.publish(message);
  }
  std::string mode_ = "disabled", session_, expected_epoch_, last_epoch_, parse_error_;
  std::string status_reason_ = "unspecified_native_caller", status_details_, pose_frame_, pose_child_frame_, pose_source_key_;
  bool status_coverage_ = false;
  std::vector<double> pose_xyz_, pose_xyzw_;
  std::map<int, std::pair<std::string, std::string>> dispatches_;
  double wait_sec_ = 0;
  Cost max_age_ns_ = 0, source_stamp_ns_ = 0;
  std::uint64_t epoch_counter_ = 0;
  std::uint64_t event_index_ = 0, pose_sequence_ = 0, pose_stamp_ns_ = 0, previous_pose_stamp_ns_ = 0;
  std::deque<std::string> sources_;
  std::unique_ptr<Advice> received_;
  ros::CallbackQueue queue_;
  std::unique_ptr<ros::NodeHandle> advice_nh_;
  ros::Publisher snapshot_pub_, decision_pub_, feedback_pub_;
  ros::Subscriber advice_sub_;
};
// One inline function-local singleton is shared by both patched translation units.
inline RosBridge& bridge() { static RosBridge value; return value; }
}  // namespace native_structure_bridge
