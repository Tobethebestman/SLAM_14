# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yynn/SLAM_14/ch7/featureMatch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yynn/SLAM_14/ch7/featureMatch/build

# Include any dependencies generated for this target.
include CMakeFiles/orbCV.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/orbCV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/orbCV.dir/flags.make

CMakeFiles/orbCV.dir/orb_cv.cpp.o: CMakeFiles/orbCV.dir/flags.make
CMakeFiles/orbCV.dir/orb_cv.cpp.o: ../orb_cv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yynn/SLAM_14/ch7/featureMatch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/orbCV.dir/orb_cv.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orbCV.dir/orb_cv.cpp.o -c /home/yynn/SLAM_14/ch7/featureMatch/orb_cv.cpp

CMakeFiles/orbCV.dir/orb_cv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orbCV.dir/orb_cv.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yynn/SLAM_14/ch7/featureMatch/orb_cv.cpp > CMakeFiles/orbCV.dir/orb_cv.cpp.i

CMakeFiles/orbCV.dir/orb_cv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orbCV.dir/orb_cv.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yynn/SLAM_14/ch7/featureMatch/orb_cv.cpp -o CMakeFiles/orbCV.dir/orb_cv.cpp.s

# Object files for target orbCV
orbCV_OBJECTS = \
"CMakeFiles/orbCV.dir/orb_cv.cpp.o"

# External object files for target orbCV
orbCV_EXTERNAL_OBJECTS =

orbCV: CMakeFiles/orbCV.dir/orb_cv.cpp.o
orbCV: CMakeFiles/orbCV.dir/build.make
orbCV: /usr/local/lib/libopencv_dnn.so.3.4.15
orbCV: /usr/local/lib/libopencv_highgui.so.3.4.15
orbCV: /usr/local/lib/libopencv_ml.so.3.4.15
orbCV: /usr/local/lib/libopencv_objdetect.so.3.4.15
orbCV: /usr/local/lib/libopencv_shape.so.3.4.15
orbCV: /usr/local/lib/libopencv_stitching.so.3.4.15
orbCV: /usr/local/lib/libopencv_superres.so.3.4.15
orbCV: /usr/local/lib/libopencv_videostab.so.3.4.15
orbCV: /usr/local/lib/libopencv_viz.so.3.4.15
orbCV: /usr/local/lib/libopencv_calib3d.so.3.4.15
orbCV: /usr/local/lib/libopencv_features2d.so.3.4.15
orbCV: /usr/local/lib/libopencv_flann.so.3.4.15
orbCV: /usr/local/lib/libopencv_photo.so.3.4.15
orbCV: /usr/local/lib/libopencv_video.so.3.4.15
orbCV: /usr/local/lib/libopencv_videoio.so.3.4.15
orbCV: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
orbCV: /usr/local/lib/libopencv_imgproc.so.3.4.15
orbCV: /usr/local/lib/libopencv_core.so.3.4.15
orbCV: CMakeFiles/orbCV.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yynn/SLAM_14/ch7/featureMatch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable orbCV"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orbCV.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/orbCV.dir/build: orbCV

.PHONY : CMakeFiles/orbCV.dir/build

CMakeFiles/orbCV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/orbCV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/orbCV.dir/clean

CMakeFiles/orbCV.dir/depend:
	cd /home/yynn/SLAM_14/ch7/featureMatch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yynn/SLAM_14/ch7/featureMatch /home/yynn/SLAM_14/ch7/featureMatch /home/yynn/SLAM_14/ch7/featureMatch/build /home/yynn/SLAM_14/ch7/featureMatch/build /home/yynn/SLAM_14/ch7/featureMatch/build/CMakeFiles/orbCV.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/orbCV.dir/depend

