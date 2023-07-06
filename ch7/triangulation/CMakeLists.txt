cmake_minimum_required( VERSION 2.8 )

project( TRIANGULATION )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( tri triangulation.cpp )

target_link_libraries( tri ${OpenCV_LIBS} )
