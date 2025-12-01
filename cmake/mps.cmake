# cmake/mps.cmake
# MPS/Metal backend configuration - only included when TORCHSCIENCE_ENABLE_MPS=ON

# Find Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "Found Foundation framework: ${FOUNDATION_FRAMEWORK}")

# Metal shader sources (add files as they are implemented)
set(TORCHSCIENCE_METAL_SOURCES
  # src/torchscience/csrc/metal/special_functions/gamma.metal
  # Add more .metal files as implemented
)

# Only set up Metal compilation if there are Metal sources
list(LENGTH TORCHSCIENCE_METAL_SOURCES METAL_SOURCE_COUNT)
if(METAL_SOURCE_COUNT GREATER 0)
  # Metal compilation: .metal -> .air -> .metallib
  set(METALLIB_OUTPUT "${CMAKE_BINARY_DIR}/torchscience.metallib")
  set(AIR_FILES "")

  foreach(METAL_SOURCE ${TORCHSCIENCE_METAL_SOURCES})
    get_filename_component(METAL_NAME ${METAL_SOURCE} NAME_WE)
    set(AIR_FILE "${CMAKE_BINARY_DIR}/${METAL_NAME}.air")

    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/${METAL_SOURCE} -o ${AIR_FILE}
      DEPENDS ${METAL_SOURCE}
      COMMENT "Compiling ${METAL_SOURCE} to AIR"
    )
    list(APPEND AIR_FILES ${AIR_FILE})
  endforeach()

  add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking Metal library"
  )

  add_custom_target(metallib ALL DEPENDS ${METALLIB_OUTPUT})
  add_dependencies(_csrc metallib)

  # Install metallib alongside the library
  if(SKBUILD)
    install(FILES ${METALLIB_OUTPUT} DESTINATION ${SKBUILD_PROJECT_NAME})
  endif()
endif()

# Link Metal frameworks
target_link_libraries(_csrc PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

# Compile definition to enable MPS code paths
target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_MPS)

message(STATUS "MPS backend enabled")
