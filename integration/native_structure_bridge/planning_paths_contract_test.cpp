#include "native_planning_paths.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>
struct Node { std::array<double,3> position_; int type_; int global_subspace_index_; };
struct Path { std::vector<Node> nodes_; };
int main() {
  Path global{{{{1.,2.,3.},1,12},{{1.,2.,8.},1,13}}}, local{};
  const auto original = global.nodes_;
  const std::array<double,3> lookahead{{4.,5.,6.}};
  auto text = native_structure_bridge::planningPathsFields(global,local,lookahead,false,false);
  assert(text.find("\"paths_complete\":true") != std::string::npos);
  assert(text.find("\"native_subspace_index\":13") != std::string::npos);
  assert(text.find("[1,2,8]") != std::string::npos);
  assert(text.find("\"lookahead_updated\":false") != std::string::npos);
  assert(global.nodes_[0].position_ == original[0].position_);
  global.nodes_.resize(16385);
  text = native_structure_bridge::planningPathsFields(global,local,lookahead,true,true);
  assert(text.find("\"paths_complete\":false") != std::string::npos);
  assert(text.find("\"global_path\":null") != std::string::npos);
  global.nodes_.resize(1);
  global.nodes_[0].position_[0] = std::numeric_limits<double>::quiet_NaN();
  bool failed = false;
  try { native_structure_bridge::planningPathsFields(global,local,lookahead,true,true); }
  catch (const std::runtime_error&) { failed = true; }
  assert(failed);
  std::cout << "native_planning_paths PASS\n";
}
