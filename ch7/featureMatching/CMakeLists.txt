cmake_minimum_required( VERSION 2.8 )

project( ORB_CV )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( orbCV orb_cv.cpp )

target_link_libraries( orbCV ${OpenCV_LIBS} )
