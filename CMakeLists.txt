cmake_minimum_required(VERSION 3.15)
project(SplitNewton)

# Use vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "/Users/apple/Desktop/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the header-only library target
add_library(SplitNewton INTERFACE)
target_include_directories(SplitNewton INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find external dependencies
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Link external dependencies to the header-only library
target_link_libraries(SplitNewton INTERFACE fmt::fmt spdlog::spdlog_header_only)

# Add installation rules for headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

# Export the library for downstream projects
install(TARGETS SplitNewton EXPORT SplitNewtonTargets)

# Install the CMake configuration files
install(EXPORT SplitNewtonTargets
    FILE SplitNewtonConfig.cmake
    NAMESPACE SplitNewton::
    DESTINATION lib/cmake/SplitNewton
)

# Add subdirectory for tests (if tests exist, optional)
add_subdirectory(tests)