set(CMAKE_CUDA_COMPILER "/ext3/miniconda3/envs/scivid_deepmind/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/ext3/miniconda3/envs/scivid_deepmind/bin/x86_64-conda-linux-gnu-c++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.9.86")
set(CMAKE_CUDA_DEVICE_LINKER "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "14.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/lib/stubs;/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0/include/c++;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0/include/c++/x86_64-conda-linux-gnu;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0/include/c++/backward;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0/include;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0/include-fixed;/ext3/miniconda3/envs/scivid_deepmind/x86_64-conda-linux-gnu/sysroot/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/ext3/miniconda3/envs/scivid_deepmind/lib;/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/lib;/ext3/miniconda3/envs/scivid_deepmind/targets/x86_64-linux/lib/stubs;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc/x86_64-conda-linux-gnu/14.3.0;/ext3/miniconda3/envs/scivid_deepmind/lib/gcc;/ext3/miniconda3/envs/scivid_deepmind/x86_64-conda-linux-gnu/lib;/ext3/miniconda3/envs/scivid_deepmind/x86_64-conda-linux-gnu/sysroot/lib;/ext3/miniconda3/envs/scivid_deepmind/x86_64-conda-linux-gnu/sysroot/usr/lib;/usr/local/cuda/lib64/stubs")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/ext3/miniconda3/envs/scivid_deepmind/bin/x86_64-conda-linux-gnu-ld")
set(CMAKE_AR "/ext3/miniconda3/envs/scivid_deepmind/bin/x86_64-conda-linux-gnu-ar")
set(CMAKE_MT "")
