cmake_minimum_required( VERSION 2.8 )

project( 2D2D )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( 2d2d pose_estimation.cpp )

target_link_libraries( 2d2d ${OpenCV_LIBS} )
