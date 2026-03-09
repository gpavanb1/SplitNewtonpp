# Standalone binary to visually verify split vs no-split solver paths
cc_binary(
    name = "test_split_debug",
    srcs = ["test_split_debug.cpp"],
    deps = [
        "//:splitnewton_lib",
        "@spdlog//:spdlog",
        "@eigen"
    ],
    includes = ["include"],
    copts = ["-std=c++17"],
)

# Test suite combining both test files
cc_test(
    name = "splitnewton_tests",
    srcs = [
        "test_newton.cpp",
        "test_split_newton.cpp",
        "test_block_newton.cpp",
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
    "-g",
    "-fsanitize=address", "-fno-omit-frame-pointer",
    ],
    linkopts = ["-fsanitize=address"],
    linkstatic = 1,
    timeout = "moderate"
)

# Coverage-only test target (no sanitizers - they conflict with coverage instrumentation)
cc_test(
    name = "splitnewton_coverage_tests",
    srcs = [
        "test_newton.cpp",
        "test_split_newton.cpp",
        "test_block_newton.cpp",
    ],
    deps = [
        "//:splitnewton_lib",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@spdlog//:spdlog",
        "@eigen"
    ],
    includes = ["include"],
    copts = ["-std=c++17", "-g"],
    linkstatic = 1,
    timeout = "moderate"
)
