#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox_skeleton/skeleton_generator.h>

// Analytic rendering is test input generation only. The integrators receive
// ONLY first-return points and sensor transforms, never these boxes.
float firstExit(const voxblox::Point& o,const voxblox::Point& d) {
  const voxblox::Point low[2]={{-12,-2,-2},{-2,0,-2}};
  const voxblox::Point high[2]={{12,2,2},{2,12,2}};
  std::vector<std::pair<float,float>> intervals;
  for(int b=0;b<2;++b) {
    float near=-1e30f,far=1e30f;
    for(int k=0;k<3;++k) {
      if(std::abs(d[k])<1e-8f) {
        if(o[k]<low[b][k] || o[k]>high[b][k]) {near=1e30f;far=-1e30f;break;}
      } else {
        float a=(low[b][k]-o[k])/d[k],z=(high[b][k]-o[k])/d[k];
        near=std::max(near,std::min(a,z));far=std::min(far,std::max(a,z));
      }
    }
    if(near<=far && far>0)intervals.emplace_back(std::max(near,0.f),far);
  }
  std::sort(intervals.begin(),intervals.end());float end=0;
  for(const auto& i:intervals) {if(i.first>end+1e-5f)break;end=std::max(end,i.second);}
  CHECK_GT(end,0);CHECK_LT(end,50);return end;
}

int main(int argc,char**argv) {
  google::InitGoogleLogging(argv[0]);
  if(argc!=2 || (std::string(argv[1])!="2" && std::string(argv[1])!="5"))return 2;
  const float range_limit=std::stof(argv[1]);
  voxblox::Layer<voxblox::TsdfVoxel> tsdf(.25,16);
  voxblox::Layer<voxblox::EsdfVoxel> esdf(.25,16);
  voxblox::TsdfIntegratorBase::Config tc;
  tc.integrator_threads=1;tc.max_ray_length_m=50;
  tc.default_truncation_distance=1.;
  voxblox::SimpleTsdfIntegrator ti(tc,&tsdf);
  voxblox::EsdfIntegrator::Config ec;
  ec.full_euclidean_distance=true;ec.add_occupied_crust=false;
  ec.max_distance_m=range_limit;ec.default_distance_m=range_limit;
  voxblox::EsdfIntegrator ei(ec,&tsdf,&esdf);
  size_t rays=0;
  for(int frame=0;frame<5;++frame) {
    voxblox::Point origin(-4.f+frame,0,0);
    voxblox::Pointcloud points;
    for(int ring=0;ring<16;++ring)for(int col=0;col<720;++col) {
      double elevation=(-15.+2.*ring)*M_PI/180.,azimuth=col*.5*M_PI/180.;
      voxblox::Point d(std::cos(elevation)*std::cos(azimuth),
                       std::cos(elevation)*std::sin(azimuth),std::sin(elevation));
      points.push_back(d*firstExit(origin,d));
    }
    voxblox::Colors colors(points.size(),voxblox::Color(128,128,128));
    voxblox::Transformation pose;pose.getPosition()=origin;
    ti.integratePointCloud(pose,points,colors);rays+=points.size();
    ei.updateFromTsdfLayer(true);
  }
  size_t observed=0,unknown=0,hallucinated=0,source_missing=0;
  size_t eligible=0,parentless=0,central_eligible=0,central_parentless=0;
  voxblox::BlockIndexList blocks;esdf.getAllAllocatedBlocks(&blocks);
  for(const auto& index:blocks) {
    auto b=esdf.getBlockPtrByIndex(index);
    for(size_t i=0;i<b->num_voxels();++i) {
      const auto& v=b->getVoxelByLinearIndex(i);
      if(!v.observed) {++unknown;continue;}
      ++observed;CHECK(std::isfinite(v.distance));
      if(v.hallucinated)++hallucinated;
      auto p=b->computeCoordinatesFromLinearIndex(i);
      if(v.distance>=.4f && !v.fixed) {
        ++eligible; if(v.parent.isZero())++parentless;
        if(p.norm()<=1.f) {++central_eligible; if(v.parent.isZero())++central_parentless;}
      }
      auto* source=tsdf.getVoxelPtrByCoordinates(p);
      if(!source || source->weight<ec.min_weight)++source_missing;
    }
  }
  CHECK_EQ(hallucinated,0);CHECK_EQ(source_missing,0);CHECK_GT(unknown,0);
  voxblox::SkeletonGenerator generator(&esdf);
  generator.generateSkeleton();generator.generateSparseGraph();
  std::vector<int64_t> vertices,edges;
  generator.getSparseGraph().getAllVertexIds(&vertices);
  generator.getSparseGraph().getAllEdgeIds(&edges);
  size_t central_skeleton=0;
  for(const auto& p:generator.getSkeleton().getSkeletonPoints()) if(p.point.norm()<=1.f)++central_skeleton;
  std::cout<<"{\"range_limit_m\":"<<range_limit
    <<",\"eligible\":"<<eligible<<",\"parentless\":"<<parentless
    <<",\"central_eligible\":"<<central_eligible<<",\"central_parentless\":"<<central_parentless
    <<",\"central_skeleton_points\":"<<central_skeleton
    <<",\"scope\":\"synthetic_fiveframe_lidar_not_research\",\"rays\":"<<rays
    <<",\"observed_esdf\":"<<observed<<",\"allocated_unknown\":"<<unknown
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
  std::cout<<"]}\n";
}

