#!/bin/bash

# Exit script on error
set -e

# Define the vcpkg toolchain file path
VCPKG_TOOLCHAIN_FILE="/Users/apple/Desktop/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Set the build type (default to Debug if not provided)
BUILD_TYPE=${1:-"Debug"}

echo "Using build type: $BUILD_TYPE"
echo "Using vcpkg toolchain file: $VCPKG_TOOLCHAIN_FILE"

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/

# Configure the project with CMake
echo "Configuring the project with CMake..."
cmake -B build -DCMAKE_TOOLCHAIN_FILE=${VCPKG_TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DVCPKG_MANIFEST_MODE=ON

# Build the project
echo "Building the project..."
cmake --build build

echo "Build completed successfully."

