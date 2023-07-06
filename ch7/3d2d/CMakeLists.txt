cmake_minimum_required( VERSION 2.8 )

project( 3D2D )

# add c++_17 supportion
set( CMAKE_CXX_FLAGS "-std=c++17" )

include_directories( "/usr/include/eigen3" )

# OpenCV
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# G2O
find_package( g2o REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

add_executable( epnp epnp.cpp )
add_executable( hpnp hpnp.cpp )
add_executable( opnp opnp.cpp )

target_link_libraries( epnp ${OpenCV_LIBS} )
target_link_libraries( hpnp ${OpenCV_LIBS} )
target_link_libraries( opnp ${OpenCV_LIBS} g2o_core g2o_stuff )
