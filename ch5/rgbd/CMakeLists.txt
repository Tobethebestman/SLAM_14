cmake_minimum_required( VERSION 2.8 )

project( JOINTMAP )

find_package( OpenCV 3.4.15 REQUIRED )
find_package( Pangolin REQUIRED )

include_directories( OpenCV_INCLUDE_DIRS )

add_executable( jointMap jointMap.cpp )

target_link_libraries( jointMap ${OpenCV_LIBS} )
target_link_libraries( jointMap ${Pangolin_LIBRARIES} )
