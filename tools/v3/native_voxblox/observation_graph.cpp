#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox_skeleton/skeleton_generator.h>

int main(int argc,char**argv) {
  google::InitGoogleLogging(argv[0]);
  std::string magic; int frames=0;
  if(!(std::cin>>magic>>frames) || magic!="GSE_RANGE_V1" || frames!=5) return 2;
  std::vector<voxblox::Pointcloud> scans(5);
  std::vector<voxblox::Transformation> poses(5);
  size_t below_min_ray=0;
  for(int f=0;f<5;++f) {
    float x,y,z,yaw;
    if(!(std::cin>>x>>y>>z>>yaw) || !std::isfinite(x) || !std::isfinite(y) ||
       !std::isfinite(z) || !std::isfinite(yaw)) return 2;
    if(f==4 && (x!=0 || y!=0 || z!=0 || yaw!=0)) return 2;
    poses[f].getPosition()=voxblox::Point(x,y,z);
    poses[f].getRotation()=voxblox::Rotation(Eigen::Quaternionf(
      Eigen::AngleAxisf(yaw*M_PI/180.,Eigen::Vector3f::UnitZ())));
    for(int ring=0;ring<16;++ring)for(int col=0;col<720;++col) {
      float range; int valid;
      if(!(std::cin>>range>>valid) || !std::isfinite(range) || range<0 || range>50 ||
         (valid!=0 && valid!=1) || (valid && range<=0)) return 2;
      if(!valid)continue;
      if(range<.1f)++below_min_ray;
      double elevation=(-15.+2.*ring)*M_PI/180.,azimuth=col*.5*M_PI/180.;
      voxblox::Point d(std::cos(elevation)*std::cos(azimuth),
                      std::cos(elevation)*std::sin(azimuth),std::sin(elevation));
      scans[f].push_back(d*range);
    }
  }
  std::cin>>std::ws;
  if(std::cin.peek()!=std::char_traits<char>::eof())return 2;
  voxblox::Layer<voxblox::TsdfVoxel> tsdf(.25,16);
  voxblox::Layer<voxblox::EsdfVoxel> esdf(.25,16);
  voxblox::TsdfIntegratorBase::Config tc;
  tc.integrator_threads=1;tc.max_ray_length_m=50;
  tc.default_truncation_distance=1.;
  voxblox::SimpleTsdfIntegrator ti(tc,&tsdf);
  voxblox::EsdfIntegrator::Config ec;
  ec.full_euclidean_distance=true;ec.add_occupied_crust=false;
  voxblox::EsdfIntegrator ei(ec,&tsdf,&esdf);
  size_t rays=0;
  for(int frame=0;frame<5;++frame) {
    const auto& points=scans[frame];
    voxblox::Colors colors(points.size(),voxblox::Color(128,128,128));
    const auto& pose=poses[frame];
    ti.integratePointCloud(pose,points,colors);rays+=points.size();
    ei.updateFromTsdfLayer(true);
  }
  size_t observed=0,unknown=0,hallucinated=0,source_missing=0;
  voxblox::BlockIndexList blocks;esdf.getAllAllocatedBlocks(&blocks);
  for(const auto& index:blocks) {
    auto b=esdf.getBlockPtrByIndex(index);
    for(size_t i=0;i<b->num_voxels();++i) {
      const auto& v=b->getVoxelByLinearIndex(i);
      if(!v.observed) {++unknown;continue;}
      ++observed;CHECK(std::isfinite(v.distance));
      if(v.hallucinated)++hallucinated;
      auto p=b->computeCoordinatesFromLinearIndex(i);
      auto* source=tsdf.getVoxelPtrByCoordinates(p);
      if(!source || source->weight<ec.min_weight)++source_missing;
    }
  }
  CHECK_EQ(hallucinated,0);CHECK_EQ(source_missing,0);
  voxblox::SkeletonGenerator generator(&esdf);
  generator.generateSkeleton();generator.generateSparseGraph();
  std::vector<int64_t> vertices,edges;
  generator.getSparseGraph().getAllVertexIds(&vertices);
  generator.getSparseGraph().getAllEdgeIds(&edges);
  std::cout<<"{\"scope\":\"fiveframe_candidate_graph_not_verified\",\"rays\":"<<rays
    <<",\"below_min_ray_count\":"<<below_min_ray<<",\"observed_esdf\":"<<observed<<",\"allocated_unknown\":"<<unknown
    <<",\"hallucinated\":"<<hallucinated<<",\"missing_tsdf_support\":"<<source_missing
    <<",\"nodes\":"<<vertices.size()<<",\"edges\":"<<edges.size()<<",\"vertices\":[";
  bool first=true;
  const auto& graph=generator.getSparseGraph();
  for(auto id:vertices) {
    if(!first)std::cout<<",";first=false;
    const auto& p=graph.getVertex(id).point;
    std::cout<<"{\"id\":"<<id<<",\"xyz\":["<<p.x()<<","<<p.y()<<","<<p.z()<<"]}";
  }
  std::cout<<"],\"connections\":[";first=true;
  for(auto id:edges) {
    if(!first)std::cout<<",";first=false;
    const auto& e=graph.getEdge(id);
    std::cout<<"["<<e.start_vertex<<","<<e.end_vertex<<"]";
  }
  std::cout<<"],\"edge_audit\":[";first=true;
  for(auto id:edges) {
    if(!first)std::cout<<",";first=false;
    const auto& e=graph.getEdge(id);
    const auto a=graph.getVertex(e.start_vertex).point;
    const auto b=graph.getVertex(e.end_vertex).point;
    const int steps=std::max(1,static_cast<int>(std::ceil((b-a).norm()/.125f)));
    int unknown_samples=0,below_min_samples=0;
    float minimum=std::numeric_limits<float>::infinity();
    for(int j=0;j<=steps;++j) {
      const voxblox::Point p=a+(b-a)*(static_cast<float>(j)/steps);
      const auto* v=esdf.getVoxelPtrByCoordinates(p);
      if(!v || !v->observed) {++unknown_samples;continue;}
      CHECK(std::isfinite(v->distance));minimum=std::min(minimum,v->distance);
      if(v->distance<generator.getMinGvdDistance())++below_min_samples;
    }
    std::cout<<"{\"edge_id\":"<<id<<",\"endpoints\":["<<e.start_vertex<<","<<e.end_vertex
      <<"],\"samples\":"<<steps+1<<",\"unknown_samples\":"<<unknown_samples
      <<",\"below_0_4m_samples\":"<<below_min_samples<<",\"minimum_sampled_esdf_m\":";
    if(std::isfinite(minimum))std::cout<<minimum;else std::cout<<"null";
    std::cout<<"}";
  }
  std::cout<<"],\"audit_spacing_m\":0.125,\"audit_is_continuous_safety_proof\":false}\n";
}
