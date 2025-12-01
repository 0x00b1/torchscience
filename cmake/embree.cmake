# cmake/embree.cmake
# Embree ray tracing library configuration

find_package(embree 4 QUIET)

if(embree_FOUND)
  message(STATUS "Found Embree: ${embree_VERSION}")
  target_link_libraries(_csrc PRIVATE embree)
  target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_EMBREE)
else()
  message(STATUS "Embree not found - BVH features disabled")
endif()
