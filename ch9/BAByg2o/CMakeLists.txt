cmake_minimum_required( VERSION 2.8 )

project( BABYCERES )

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# eigen
include_directories( "usr/include/eigen3" )

# ceres
find_package( Ceres REQUIRED )
include_directories( ${Ceres_INCLUDE_DIRS} )

# common.cpp
add_library( bal_common common.cpp )

add_executable( bundle_adjustment_ceres bundle_adjustment_ceres.cpp)

target_link_libraries( bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common ) 
