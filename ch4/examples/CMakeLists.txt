cmake_minimum_required( VERSION 2.8 )

add_executable( trajectoryError trajectoryError.cpp )

find_package( Pangolin REQUIRED )
find_package( Sophus REQUIRED )
target_link_libraries( trajectoryError ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ) 