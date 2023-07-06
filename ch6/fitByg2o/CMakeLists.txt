cmake_minimum_required( VERSION 2.8 )

project( G2OCURVEFITTING )

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# G2O
find_package( g2o REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

# Eigen
include_directories( "/usr/include/eigen3" )

add_executable( g2oFit g2oCurveFitting.cpp )

# NOTE!!!
target_link_libraries( g2oFit ${OpenCV_LIBS} g2o_core g2o_stuff)
