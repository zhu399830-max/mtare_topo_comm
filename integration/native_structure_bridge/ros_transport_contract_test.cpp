#include "native_route_ros.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
using namespace native_structure_bridge;
int main() {
  // Default-disabled mode must work without ros::init(), ROS master or a wait.
  RosBridge b;
  b.observePose(1, {0, 0, 0}, {0, 0, 0, 1}, "map", "sensor", "/state_estimation_at_scan");
  b.regionVisit(1, 0, {1, 0, 0}, {0, 0, 0});
  b.regionDispatch(1, 1, {1, 0, 0}, {1, 0, 0}, {0, 0, 0});
  b.setStatusReason("local_residual_coverage_below_threshold", true);
  b.regionStatus(1, 1, 2, 0, {1, 0, 0});
  assert(nativeStatusName(2) == "COVERED" && nativeStatusName(5) == "EXPLORING_BY_OTHERS");
  auto start = std::chrono::steady_clock::now();
  auto d = b.consider({0, 1, 0}, {}, {}, {}, {}, {});
  assert(!b.enabled() && d.reason == "DISABLED" && d.route == std::vector<int>({0, 1, 0}));
  assert(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < .05);
  std::string raw = R"({"schema_version":"native_route_advice_v1","epoch":"s:1","snapshot_stamp_ns":1,"source_frame_keys":["scan:1"],"candidate_ids":[1],"route":[0,1,0],"claimed_cost_units":20,"evidence_refs":[]})";
  auto a = parseAdvice(raw); assert(a.epoch == "s:1" && a.route.size() == 3);
  bool rejected = false;
  try { parseAdvice(raw.substr(0, raw.size() - 1) + ",\"epoch\":\"s:2\"}"); }
  catch (const std::exception&) { rejected = true; }
  assert(rejected);
  rejected = false;
  auto malformed = raw; malformed.replace(malformed.find("[1]"), 3, "[true]");
  try { parseAdvice(malformed); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
  Snapshot s; s.epoch = "s:1"; s.stamp_ns = 1; s.max_age_ns = 500000000;
  s.original_route = {0, 1, 0}; s.candidate_ids = {1}; s.source_frame_keys = {"scan:1"};
  s.native_edges = {{{0, 1}, 10}, {{1, 0}, 10}}; s.original_cost_units = 20;
  auto output = snapshotJson(s, {{0, {0, 0, 0}}, {1, {1, 2, 3}}}, {{0, 0}, {1, 1}}, {0, 0, 0});
  boost::property_tree::ptree tree; std::istringstream input(output); boost::property_tree::read_json(input, tree);
  assert(tree.get_child("candidate_records").size() == 1);
  assert(tree.get_child("candidate_records").front().second.get<std::string>("status_name") == "EXPLORING");
  assert(tree.get_child("source_frame_keys").size() == 1);
  const double precise_pose = 1.2345678901234567;
  boost::property_tree::ptree precise_tree;
  std::istringstream precise_json("{\"pose\":" + array(std::vector<double>{precise_pose}) + "}");
  boost::property_tree::read_json(precise_json, precise_tree);
  assert(precise_tree.get_child("pose").front().second.get_value<double>() == precise_pose);
  std::cout << "native_route_ros_transport PASS; disabled has no ROS master and no wait\n";
}
