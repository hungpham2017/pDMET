# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /burg/home/hqp2000/.conda/envs/lion/bin/cmake

# The command to remove a file.
RM = /burg/home/hqp2000/.conda/envs/lion/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /burg/home/hqp2000/pDMET/pdmet/lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /burg/home/hqp2000/pDMET/pdmet/lib/build

# Include any dependencies generated for this target.
include CMakeFiles/libdmet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libdmet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libdmet.dir/flags.make

CMakeFiles/libdmet.dir/libdmet.cpp.o: CMakeFiles/libdmet.dir/flags.make
CMakeFiles/libdmet.dir/libdmet.cpp.o: ../libdmet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/burg/home/hqp2000/pDMET/pdmet/lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libdmet.dir/libdmet.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libdmet.dir/libdmet.cpp.o -c /burg/home/hqp2000/pDMET/pdmet/lib/libdmet.cpp

CMakeFiles/libdmet.dir/libdmet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libdmet.dir/libdmet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /burg/home/hqp2000/pDMET/pdmet/lib/libdmet.cpp > CMakeFiles/libdmet.dir/libdmet.cpp.i

CMakeFiles/libdmet.dir/libdmet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libdmet.dir/libdmet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /burg/home/hqp2000/pDMET/pdmet/lib/libdmet.cpp -o CMakeFiles/libdmet.dir/libdmet.cpp.s

# Object files for target libdmet
libdmet_OBJECTS = \
"CMakeFiles/libdmet.dir/libdmet.cpp.o"

# External object files for target libdmet
libdmet_EXTERNAL_OBJECTS =

libdmet.cpython-39-x86_64-linux-gnu.so: CMakeFiles/libdmet.dir/libdmet.cpp.o
libdmet.cpython-39-x86_64-linux-gnu.so: CMakeFiles/libdmet.dir/build.make
libdmet.cpython-39-x86_64-linux-gnu.so: CMakeFiles/libdmet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/burg/home/hqp2000/pDMET/pdmet/lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libdmet.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libdmet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libdmet.dir/build: libdmet.cpython-39-x86_64-linux-gnu.so

.PHONY : CMakeFiles/libdmet.dir/build

CMakeFiles/libdmet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libdmet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libdmet.dir/clean

CMakeFiles/libdmet.dir/depend:
	cd /burg/home/hqp2000/pDMET/pdmet/lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /burg/home/hqp2000/pDMET/pdmet/lib /burg/home/hqp2000/pDMET/pdmet/lib /burg/home/hqp2000/pDMET/pdmet/lib/build /burg/home/hqp2000/pDMET/pdmet/lib/build /burg/home/hqp2000/pDMET/pdmet/lib/build/CMakeFiles/libdmet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libdmet.dir/depend

