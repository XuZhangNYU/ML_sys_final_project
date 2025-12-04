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

# ---- pybind11 (find or fetch) ----
find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ---- CUDA Toolkit (for cudart) ----
find_package(CUDAToolkit REQUIRED)

# ---- Include paths ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Build the pybind11 extension ----
enable_language(CUDA)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

pybind11_add_module(bten src/bindings.cu)

target_compile_definitions(bten PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
target_include_directories(bten PRIVATE ${Python3_INCLUDE_DIRS})

target_link_libraries(bten PRIVATE CUDA::cudart)

# for additional CUDA compile options:
#target_compile_options(barenet PRIVATE
#  --expt-relaxed-constexpr
#  # -Xcudafe --diag_suppress=...   
#)
target_compile_options(bten PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC>
)

file(GLOB executables
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Remove bindings.cu from the executables
list(FILTER executables EXCLUDE REGEX ".*/bindings\\.(cu|cpp)$")

foreach(exe ${executables})
  get_filename_component(filename ${exe} NAME_WE)
  add_executable(${filename} ${exe})
  set_target_properties(${filename} PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89")
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