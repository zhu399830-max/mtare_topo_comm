#include <iostream>
#include <voxblox_skeleton/skeleton_generator.h>

int main() {
  // Empty map only checks native linkage. This is NOT geometry qualification.
  voxblox::Layer<voxblox::EsdfVoxel> layer(0.25, 16);
  voxblox::SkeletonGenerator generator(&layer);
  generator.generateSkeleton();
  std::cout << "NATIVE_EMPTY_MAP_LINK_SMOKE_ONLY\n";
  return 0;
}
