cmake_minimum_required( VERSION 2.8 )

add_executable( plotTrajectory plotTrajectory.cpp )

# Otherwise, after "cmake .." , "make" will have lots of errors  
find_package( Pangolin REQUIRED )
target_link_libraries( plotTrajectory ${Pangolin_LIBRARIES} ${catkin_LIBRARIES} ) 