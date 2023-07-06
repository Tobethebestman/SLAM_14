cmake_minimum_required( VERSION 2.8 )

project( OPTICALFLOW )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Eigen
include_directories( "/usr/include/eigen3" )

add_executable( LK_OpenCV LKByOpenCV.cpp )
add_executable( LK_GaussNewton LKByGaussNewton.cpp )

target_link_libraries( LK_OpenCV ${OpenCV_LIBS} )
target_link_libraries( LK_GaussNewton ${OpenCV_LIBS} )