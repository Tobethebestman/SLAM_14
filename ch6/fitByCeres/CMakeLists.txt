cmake_minimum_required( VERSION 2.8 )

project( CERESCURVEFITTING )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Ceres
find_package( Ceres REQUIRED )
include_directories( ${Ceres_INCLUDE_DIRS} )

# Eigen
include_directories( "/usr/include/eigen3" )

add_executable( ceresFit ceresCurveFitting.cpp )

target_link_libraries( ceresFit ${OpenCV_LIBS} ${CERES_LIBRARIES} )
