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
  CHECK_EQ(hallucinated,0);CHECK_EQ(source_missing,0);CHECK_GT(unknown,0);
  voxblox::SkeletonGenerator generator(&esdf);
  generator.generateSkeleton();generator.generateSparseGraph();
  std::vector<int64_t> vertices,edges;
  generator.getSparseGraph().getAllVertexIds(&vertices);
  generator.getSparseGraph().getAllEdgeIds(&edges);
  std::cout<<"{\"scope\":\"synthetic_fiveframe_lidar_not_research\",\"rays\":"<<rays
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
