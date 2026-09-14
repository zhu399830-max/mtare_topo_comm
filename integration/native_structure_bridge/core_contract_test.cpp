#include "native_route_advice.hpp"
#include <cassert>
#include <iostream>
using namespace native_structure_bridge;
int main() {
  Snapshot s; s.epoch = "session:7"; s.stamp_ns = 1000; s.max_age_ns = 100;
  s.source_frame_keys = {"registered_scan:900", "registered_scan:1000"};
  s.candidate_ids = {1, 2}; s.original_route = {0, 1, 2, 0};
  for (int a : {0, 1, 2}) for (int b : {0, 1, 2}) s.native_edges[{a, b}] = a == b ? 0 : 10;
  s.original_cost_units = 30;
  assert(validate(s, nullptr, 1001).route == s.original_route);
  Advice a; a.epoch = s.epoch; a.snapshot_stamp_ns = s.stamp_ns;
  a.source_frame_keys = s.source_frame_keys; a.candidate_ids = s.candidate_ids;
  a.route = {0, 2, 1, 0}; a.claimed_cost_units = 30; a.evidence_refs = {"verification:1"};
  assert(validate(s, &a, 1001).changed);
  a.epoch = "old"; assert(validate(s, &a, 1001).route == s.original_route); a.epoch = s.epoch;
  assert(validate(s, &a, 1101).reason == "ADVICE_NOT_FRESH");
  a.source_frame_keys[0] = "wrong"; assert(!validate(s, &a, 1001).accepted); a.source_frame_keys = s.source_frame_keys;
  a.candidate_ids.pop_back(); assert(!validate(s, &a, 1001).accepted); a.candidate_ids = s.candidate_ids;
  a.route = {0, 1, 1, 0}; assert(!validate(s, &a, 1001).accepted); a.route = {0, 2, 1, 0};
  a.claimed_cost_units = 29; assert(!validate(s, &a, 1001).accepted); a.claimed_cost_units = 30;
  a.evidence_refs.clear(); assert(!validate(s, &a, 1001).accepted); a.evidence_refs = {"verification:1"};
  s.native_edges.erase({0, 2}); assert(validate(s, &a, 1001).reason == "UNREACHABLE_NATIVE_EDGE");
  assert(s.original_route == std::vector<int>({0, 1, 2, 0}));
  std::cout << "native_route_core_contract PASS; no ROS, no observations, no control\n";
}
