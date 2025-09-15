cmake_minimum_required(VERSION 3.20)

project(BareNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

link_libraries(-lcurand)

# ---- CUDA Toolkit (for cudart) ----
find_package(CUDAToolkit REQUIRED)

# ---- Include paths ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Build the pybind11 extension ----
enable_language(CUDA)

file(GLOB executables
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Remove bindings.cu from the executables
list(FILTER executables EXCLUDE REGEX ".*/bindings\\.(cu|cpp)$")

foreach(exe ${executables})
  get_filename_component(filename ${exe} NAME_WE)
  add_executable(${filename} ${exe})
  set_target_properties(${filename} PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")
  target_compile_options(${filename} PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
  )
  target_include_directories(${filename} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
  )
  target_link_libraries(${filename} PRIVATE CUDA::cudart)
endforeach()
