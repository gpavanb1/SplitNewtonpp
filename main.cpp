#include <iostream>
#include <fmt/core.h>
#include <Eigen/Dense>

int main() {
    fmt::print("Hello, SplitNewton with vcpkg!\n");

    Eigen::Matrix2d mat;
    mat << 1, 2, 3, 4;
    std::cout << "Matrix:\n" << mat << std::endl;

    return 0;
}

