#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <voxblox_skeleton/skeleton_generator.h>

// Fully known, discrete synthetic occupancy geometry. Not LiDAR integration.
// Exact Euclidean distance to nearest occupied voxel CENTER; solids are fixed
// zero-distance sources. Outside the finite volume remains unobserved.
bool air(int x, int y, int z) {
  return z >= 4 && z <= 10 &&
    ((x >= 4 && x <= 43 && y >= 20 && y <= 26) ||
     (x >= 20 && x <= 26 && y >= 23 && y <= 43));
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  const float resolution = .25f;
  voxblox::Layer<voxblox::EsdfVoxel> layer(resolution, 16);
  std::vector<Eigen::Vector3i> occupied;
  for (int x=0; x<48; ++x) for (int y=0; y<48; ++y) for (int z=0; z<16; ++z) {
    if (!air(x,y,z)) occupied.emplace_back(x,y,z);
    layer.allocateBlockPtrByCoordinates(voxblox::Point(x+.5f,y+.5f,z+.5f)*resolution);
  }
  size_t free_count = 0;
  for (int x=0; x<48; ++x) for (int y=0; y<48; ++y) for (int z=0; z<16; ++z) {
    auto* v = layer.getVoxelPtrByGlobalIndex(voxblox::GlobalIndex(x,y,z));
    CHECK_NOTNULL(v);
    v->observed=true; v->fixed=!air(x,y,z); v->distance=0; v->parent.setZero();
    if (v->fixed) continue;
    ++free_count;
    Eigen::Vector3i index(x,y,z), parent;
    int best=std::numeric_limits<int>::max();
    for (const auto& obstacle:occupied) {
      int squared=(obstacle-index).squaredNorm();
      // Deterministic lexicographic tie break from occupied enumeration.
      if (squared<best) {best=squared;parent=obstacle-index;}
    }
    CHECK_GT(best,0);
    v->parent=parent;
    v->distance=resolution*std::sqrt(static_cast<float>(best));
    auto source=index+parent;
    CHECK(!air(source.x(),source.y(),source.z()));
    CHECK_LT(std::abs(v->distance-resolution*parent.cast<float>().norm()),1e-6);
  }
  voxblox::SkeletonGenerator generator(&layer);
  generator.generateSkeleton();
  generator.generateSparseGraph();
  const auto& graph=generator.getSparseGraph();
  std::vector<int64_t> vertices,edges;
  graph.getAllVertexIds(&vertices); graph.getAllEdgeIds(&edges);
  std::map<int64_t,int> degrees;
  for(auto id:vertices) degrees[id]=0;
  for(auto id:edges) {
    const auto& edge=graph.getEdge(id);
    CHECK(degrees.count(edge.start_vertex)); CHECK(degrees.count(edge.end_vertex));
    ++degrees[edge.start_vertex]; ++degrees[edge.end_vertex];
  }
  std::cout<<"{\"scope\":\"fully_known_discrete_T_not_lidar\",\"free_voxels\":"<<free_count
    <<",\"parent_distance_checks\":"<<free_count<<",\"nodes\":"<<vertices.size()
    <<",\"edges\":"<<edges.size()<<",\"vertices\":[";
  bool first=true;
  for(auto id:vertices) {
    if(!first)std::cout<<",";first=false;
    const auto& p=graph.getVertex(id).point;
    std::cout<<"{\"id\":"<<id<<",\"degree\":"<<degrees[id]<<",\"xyz\":["
      <<p.x()<<","<<p.y()<<","<<p.z()<<"]}";
  }
  std::cout<<"],\"connections\":[";
  first=true;
  for(auto id:edges) {
    if(!first)std::cout<<",";first=false;
    const auto& e=graph.getEdge(id);
    std::cout<<"["<<e.start_vertex<<","<<e.end_vertex<<"]";
  }
  std::cout<<"]}\n";
  return 0;
}
