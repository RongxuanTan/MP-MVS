message("SkySegment---${PROJECT_SOURCE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/SkySegment/)
add_library(SkySegment SHARED ./src/SkyRegionDetect.cpp)
target_link_libraries(SkySegment
        ${OpenCV_LIBS}
        ncnn
        )

