# declare the minimum required version of cmake
cmake_minimum_required( VERSION 2.8 )

# declare a cmake project
project( HelloWorld )

# add a library(shared)
add_library( hello SHARED libHello.cpp )

# add an executable program
add_executable( helloWorld main.cpp )

# add an executable program with library
add_executable( useHello useHello.cpp )
target_link_libraries( useHello hello )
