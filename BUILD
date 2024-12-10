# Main executable
cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        ":splitnewton_lib",
        "@spdlog//:spdlog",  # Add spdlog dependency here as well
        "@eigen"
    ],
    includes = ["include"],  # Add this to include headers directly
    copts = ["-std=c++17"],  # Enable C++17
)

# Library definition
cc_library(
    name = "splitnewton_lib",
    hdrs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog//:spdlog",  # Add spdlog dependency here as well
        "@eigen"
    ],
    includes = ["include"],  # Add this to include headers directly
    copts = ["-std=c++17"],  # Enable C++17
)
