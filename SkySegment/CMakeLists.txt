message("SkySegment---${PROJECT_SOURCE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/SkySegment/)
CUDA_ADD_LIBRARY(
        SkySegment SHARED
        ./src/SkyRegionDetect.cpp
        ./src/SkyRegionDetect.cu
)
target_link_libraries(SkySegment
        ${OpenCV_LIBS}
        ncnn
        )

