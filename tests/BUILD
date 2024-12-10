# Test suite combining both test files
cc_test(
    name = "splitnewton_tests",
    srcs = [
        "test_newton.cpp",
        "test_split_newton.cpp",
    ],
    deps = [
        "//:splitnewton_lib",  # Your library
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@spdlog//:spdlog",    # Add spdlog dependency here as well
        "@eigen"
    ],
    includes = ["include"],  # Add this to include headers directly
    copts = [
    "-std=c++17",                   # Enable C++17
    "-g"
    ],
    timeout = "moderate"
)
