cmake_minimum_required(VERSION 3.16)
project(gse_official_skeleton_smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(V "${ROOT}/third_party/voxblox_reference/voxblox")
set(S "${ROOT}/third_party/mav_voxblox_planning_reference/voxblox_skeleton")
set(P "${ROOT}/build/gse_voxblox_cpu_v1/prefix/usr")
set(G "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${G}")
set(PROTOS "${V}/proto/voxblox/Block.proto" "${V}/proto/voxblox/Layer.proto" "${S}/proto/voxblox_skeleton/skeleton.proto")
set(GENERATED "${G}/voxblox/Block.pb.cc" "${G}/voxblox/Layer.pb.cc" "${G}/voxblox_skeleton/skeleton.pb.cc")
add_custom_command(OUTPUT ${GENERATED}
  COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${P}/lib/x86_64-linux-gnu"
  "${P}/bin/protoc" "--proto_path=${V}/proto" "--proto_path=${S}/proto" "--cpp_out=${G}" ${PROTOS}
  DEPENDS ${PROTOS})
include_directories("${G}" "${G}/voxblox_skeleton" "${V}/include" "${S}/include" "${P}/include"
  "${ROOT}/build/gse_spg_cpu_v1/prefix/usr/include/eigen3"
  "${ROOT}/third_party/minkindr_reference/minkindr/include")
link_directories("${P}/lib/x86_64-linux-gnu")
# Explicit serialization API compatibility, no algorithm edits or source mutation.
file(READ "${S}/src/io/skeleton_io.cpp" SKELETON_IO)
string(FIND "${SKELETON_IO}" "uint32_t tmp_byte_offset" OFFSET_LOCATION)
if(OFFSET_LOCATION LESS 0)
  message(FATAL_ERROR "Pinned skeleton IO compatibility target drift")
endif()
string(REPLACE "uint32_t tmp_byte_offset" "uint64_t tmp_byte_offset" SKELETON_IO "${SKELETON_IO}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/skeleton_io_compat.cpp" "${SKELETON_IO}")
add_library(official_skeleton STATIC ${GENERATED}
  "${V}/src/core/block.cc" "${V}/src/utils/timing.cc"
  "${V}/src/utils/neighbor_tools.cc" "${V}/src/utils/protobuf_utils.cc"
  "${V}/src/utils/voxel_utils.cc"
  "${V}/src/integrator/tsdf_integrator.cc" "${V}/src/integrator/esdf_integrator.cc"
  "${V}/src/integrator/integrator_utils.cc"
  "${S}/src/skeleton.cpp" "${S}/src/skeleton_generator.cpp"
  "${S}/src/skeleton_planner.cpp" "${S}/src/skeleton_serialization.cpp"
  "${S}/src/sparse_graph_planner.cpp" "${S}/src/voxel_template_matcher.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/skeleton_io_compat.cpp")
target_link_libraries(official_skeleton PUBLIC protobuf glog gflags pthread)
add_executable(link_smoke link_smoke.cpp)
target_link_libraries(link_smoke PRIVATE official_skeleton)
set_target_properties(link_smoke PROPERTIES BUILD_RPATH "${P}/lib/x86_64-linux-gnu")
add_executable(synthetic_t synthetic_t.cpp)
target_link_libraries(synthetic_t PRIVATE official_skeleton)
set_target_properties(synthetic_t PROPERTIES BUILD_RPATH "${P}/lib/x86_64-linux-gnu")
add_executable(synthetic_lidar synthetic_lidar.cpp)
target_link_libraries(synthetic_lidar PRIVATE official_skeleton)
set_target_properties(synthetic_lidar PROPERTIES BUILD_RPATH "${P}/lib/x86_64-linux-gnu")
add_executable(synthetic_lidar_edge_audit synthetic_lidar_edge_audit.cpp)
target_link_libraries(synthetic_lidar_edge_audit PRIVATE official_skeleton)
set_target_properties(synthetic_lidar_edge_audit PROPERTIES BUILD_RPATH "${P}/lib/x86_64-linux-gnu")
add_executable(observation_graph observation_graph.cpp)
target_link_libraries(observation_graph PRIVATE official_skeleton)
set_target_properties(observation_graph PROPERTIES BUILD_RPATH "${P}/lib/x86_64-linux-gnu")
