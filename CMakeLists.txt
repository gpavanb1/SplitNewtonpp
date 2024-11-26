cmake_minimum_required(VERSION 3.15)
project(SplitNewton)

# Use vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "/Users/apple/Desktop/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find packages
find_package(fmt CONFIG REQUIRED)

# Add executable
add_executable(main main.cpp)

# Include the src folder for the header-only library
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link dependencies
target_link_libraries(main PRIVATE fmt::fmt)

# Add subdirectories
add_subdirectory(tests)