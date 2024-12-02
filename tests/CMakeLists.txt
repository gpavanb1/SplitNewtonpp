# Collect all test files
file(GLOB TEST_SOURCES *.cpp)

# Set flags for coverage
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-instr-generate -fcoverage-mapping")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fprofile-instr-generate -fcoverage-mapping")

# Find GTest
find_package(GTest CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Add test executable
add_executable(test_main ${TEST_SOURCES})

# Link test executable with GTest and main library
target_link_libraries(test_main PRIVATE GTest::gtest GTest::gtest_main)
target_link_libraries(test_main PRIVATE fmt::fmt spdlog::spdlog)

# Specify include directories
target_include_directories(test_main PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Discover and enable tests
include(GoogleTest)
gtest_discover_tests(test_main)