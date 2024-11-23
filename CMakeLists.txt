cmake_minimum_required(VERSION 3.15)
project(SplitNewton)

# Use vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "/Users/apple/Desktop/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Find packages
find_package(fmt CONFIG REQUIRED)

# Add executable and link dependencies
add_executable(main main.cpp)
target_link_libraries(main PRIVATE fmt::fmt)

