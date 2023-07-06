cmake_minimum_required( VERSION 2.8 )

project( STEREOVISION )

find_package( OpenCV 3.4.15 REQUIRED )
find_package( Pangolin REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( stereoVision stereoVision.cpp )

target_link_libraries( stereoVision ${OpenCV_LIBS} )
target_link_libraries( stereoVision ${Pangolin_LIBRARIES})
