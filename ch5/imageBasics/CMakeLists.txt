cmake_minimum_required( VERSION 2.8 )

project( IMAGEBASICS )

# add c++_11 supportion
set( CMAKE_CXX_FLAGS "-std=c++11" )

# find package and add header_file
find_package( OpenCV 3.4.15 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# imageBasics.cpp
add_executable( imageBasics imageBasics.cpp )

# link to package
target_link_libraries( imageBasics ${OpenCV_LIBS} )

# undistortImage.cpp
add_executable( undistortImage undistortImage.cpp )

# link to package
target_link_libraries( undistortImage ${OpenCV_LIBS} )
