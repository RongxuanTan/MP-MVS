cmake_minimum_required (VERSION 2.8)
project (MPMVS)

#set CUDA path
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.0")
find_package(CUDA 11.0 REQUIRED ) 
set(OpenCV_DIR "/usr/local/OpenCV3415/share/OpenCV")
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
set(ncnn_DIR "/home/xuan/ncnn/build/install/lib/cmake/ncnn")
set(NCNN_INCLUDE_DIRS /ncnn/build/install/include)
find_package(ncnn REQUIRED)

include_directories(${NCNN_INCLUDE_DIRS})

#-arch: 指定nvcc编译目标GPU的型号
#-code：选项指定了nvcc最后阶段生成的运行时代码的目标架构
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_80,code=sm_80)
message(“CUDA_NVCC_FLAGS:” ${CUDA_NVCC_FLAGS})
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    MPMVS
    main.h    
    PatchMatch.cpp 
    PatchMatch.h 
    PatchMatch.cu 
    main.cpp
    FileIO.h
    FileIO.cpp
    )

# For linking ...
# Specify target & libraries to link it with

target_link_libraries(MPMVS
    ${OpenCV_LIBS}
    ncnn
    )