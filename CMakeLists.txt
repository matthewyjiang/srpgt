cmake_minimum_required(VERSION 3.12)
project(SRPGT VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Include directories
include_directories(include)

# Find required packages
find_package(Threads REQUIRED)

# Source files
set(SOURCES
    src/robot.cpp
    src/polygeom.cpp
    src/reactive_planner.cpp
    src/disjoint_set.cpp
    src/config.cpp
)

# Header files
set(HEADERS
    include/robot.h
    include/polygeom.h
    include/reactive_planner.h
    include/disjoint_set.h
    include/config.h
    include/common.h
)

# Create library
add_library(srpgt_lib ${SOURCES} ${HEADERS})

# Main executable
add_executable(srpgt src/main.cpp)
target_link_libraries(srpgt srpgt_lib Threads::Threads)

# Test executable
add_executable(srpgt_test src/test.cpp)
target_link_libraries(srpgt_test srpgt_lib Threads::Threads)

# Install targets
install(TARGETS srpgt DESTINATION bin)
install(TARGETS srpgt_lib DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/srpgt)