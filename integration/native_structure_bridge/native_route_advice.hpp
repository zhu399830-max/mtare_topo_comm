#pragma once
// ROS-free conservative route advice. Native integer costs are not metres.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace native_structure_bridge {
using Cost = std::int64_t;
using Edges = std::map<std::pair<int, int>, Cost>;
struct Snapshot {
  std::string epoch;
  Cost stamp_ns = 0, max_age_ns = 0, max_extra_cost_units = 0, original_cost_units = 0;
  std::vector<std::string> source_frame_keys;
  std::vector<int> candidate_ids, original_route;
  Edges native_edges;
};
struct Advice {
  std::string epoch;
  Cost snapshot_stamp_ns = 0, claimed_cost_units = 0;
  std::vector<std::string> source_frame_keys, evidence_refs;
  std::vector<int> candidate_ids, route;
};
struct Decision {
  bool accepted = false, changed = false;
  std::string reason = "NO_ADVICE";
  std::vector<int> route;
  Cost original_cost_units = 0, advised_cost_units = -1;
};
inline void require(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
template <class T> inline std::vector<T> sorted(std::vector<T> values) {
  std::sort(values.begin(), values.end()); return values;
}
template <class T> inline bool unique(const std::vector<T>& values) {
  return std::set<T>(values.begin(), values.end()).size() == values.size();
}
inline bool sourcesValid(const std::vector<std::string>& values, bool allow_empty = false) {
  if ((!allow_empty && values.empty()) || !unique(values)) return false;
  for (const auto& s : values) if (s.empty() || s.size() > 512) return false;
  return true;
}
inline bool idsValid(const std::vector<int>& values) {
  for (int value : values) if (value < 0) return false;
  return true;
}
inline Cost routeCost(const std::vector<int>& route, const Edges& edges) {
  Cost total = 0;
  for (std::size_t i = 1; i < route.size(); ++i) {
    auto found = edges.find({route[i - 1], route[i]});
    require(found != edges.end(), "UNREACHABLE_NATIVE_EDGE");
    require(found->second >= 0, "NEGATIVE_NATIVE_COST");
    require(total <= std::numeric_limits<Cost>::max() - found->second, "COST_OVERFLOW");
    total += found->second;
  }
  return total;
}
inline void validateSnapshot(const Snapshot& s) {
  require(!s.epoch.empty(), "SNAPSHOT_EPOCH");
  require(s.stamp_ns >= 0 && s.max_age_ns > 0 && s.max_extra_cost_units >= 0, "SNAPSHOT_TIME_POLICY");
  require(sourcesValid(s.source_frame_keys), "SNAPSHOT_SOURCES");
  require(idsValid(s.candidate_ids) && unique(s.candidate_ids), "SNAPSHOT_CANDIDATES");
  require(idsValid(s.original_route) && s.original_route.size() >= 2, "SNAPSHOT_ROUTE");
  require(s.original_route.front() == s.original_route.back(), "SNAPSHOT_ENDPOINTS");
  for (int id : s.original_route)
    require(id == s.original_route.front() ||
        std::find(s.candidate_ids.begin(), s.candidate_ids.end(), id) != s.candidate_ids.end(),
        "SNAPSHOT_ROUTE_MEMBERS");
  require(routeCost(s.original_route, s.native_edges) == s.original_cost_units, "ORIGINAL_COST_MISMATCH");
}
inline Decision validate(const Snapshot& s, const Advice* a, Cost now_ns) {
  validateSnapshot(s);  // A malformed native snapshot is an integration defect.
  Decision d; d.route = s.original_route; d.original_cost_units = s.original_cost_units;
  if (!a) return d;
  try {
    require(a->epoch == s.epoch, "EPOCH_MISMATCH");
    require(a->snapshot_stamp_ns == s.stamp_ns, "STAMP_MISMATCH");
    require(now_ns >= s.stamp_ns && now_ns - s.stamp_ns <= s.max_age_ns, "ADVICE_NOT_FRESH");
    require(sourcesValid(a->source_frame_keys) && a->source_frame_keys == s.source_frame_keys,
        "SOURCE_MISMATCH");
    require(idsValid(a->candidate_ids) && unique(a->candidate_ids) &&
        sorted(a->candidate_ids) == sorted(s.candidate_ids), "CANDIDATE_SET_MISMATCH");
    require(idsValid(a->route) && a->route.size() >= 2 &&
        a->route.front() == s.original_route.front() && a->route.back() == s.original_route.back(),
        "ENDPOINT_MISMATCH");
    require(sorted(a->route) == sorted(s.original_route), "ROUTE_MULTISET_MISMATCH");
    Cost cost = routeCost(a->route, s.native_edges);
    require(cost == a->claimed_cost_units, "ADVISED_COST_MISMATCH");
    require(cost <= s.original_cost_units || cost - s.original_cost_units <= s.max_extra_cost_units,
        "COST_BUDGET_EXCEEDED");
    require(sourcesValid(a->evidence_refs, a->route == s.original_route), "ADVICE_EVIDENCE");
    d.accepted = true; d.changed = a->route != s.original_route;
    d.reason = d.changed ? "ACCEPTED" : "IDENTITY_ADVICE";
    d.route = a->route; d.advised_cost_units = cost;
  } catch (const std::exception& error) { d.reason = error.what(); }
  return d;
}
}  // namespace native_structure_bridge
