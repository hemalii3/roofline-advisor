cmake_minimum_required(VERSION 3.16)
project(roofline_advisor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default; micro-benchmarks are meaningless unoptimized.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(Threads REQUIRED)

# --- OpenMP is optional: used for the multi-threaded CPU bandwidth/compute
# roofline points. Falls back to single-threaded if not found. ---
find_package(OpenMP)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(benchmarks)
add_subdirectory(src)

# Toggle this on once you have AdaptiveCpp available on exa03.
option(ENABLE_SYCL "Build SYCL GPU characterization benchmarks" OFF)
if(ENABLE_SYCL)
  add_subdirectory(benchmarks/sycl)
endif()
