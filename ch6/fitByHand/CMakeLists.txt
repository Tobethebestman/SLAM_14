cmake_minimum_required( VERSION 2.8 )

project( GAUSSNEWTON )

find_package( OpenCV 3.4.15 REQUIRED )

include_directories( OpenCV_INCLUDE_DIRS )

add_executable( gaussNewton gaussNewton.cpp )

target_link_libraries( gaussNewton ${OpenCV_LIBS} )
