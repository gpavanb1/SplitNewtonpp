# Collect all test files
file(GLOB TEST_SOURCES *.cpp)

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Add test executable
add_executable(test_main ${TEST_SOURCES})

# Link test executable with GTest and main library
target_link_libraries(test_main PRIVATE GTest::gtest GTest::gtest_main)

# Specify include directories
target_include_directories(test_main PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Discover and enable tests
include(GoogleTest)
gtest_discover_tests(test_main)