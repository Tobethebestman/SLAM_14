cmake_minimum_required( VERSION 2.8 )

project( TEST )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( test test.cpp )
target_link_libraries( test ${OpenCV_LIBS} )