cmake_minimum_required( VERSION 2.8 )

include_directories( "/usr/include/eigen3" )

add_executable( useSophus useSophus.cpp )

find_package( Sophus REQUIRED )
include_directories( useSophus ${Sophus_INCLUDE_DIRS} )
