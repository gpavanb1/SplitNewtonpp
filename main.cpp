#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <stdexcept>
#include "typedefs.h"
#include "newton.hpp"
#include "split_newton.hpp"
#include "examples/test_functions.hpp"

int main(int argc, char *argv[])
{
    // Logging level (simple argument parsing)
    std::string loglevel = "WARNING";
    if (argc > 1)
    {
        loglevel = argv[1];
    }
    std::cout << "Log level set to: " << loglevel << "\n";

    // Seed
    Vector x0 = Eigen::VectorXd::LinSpaced(5000, 21.2, 31.2);

    // Mode selection
    std::string mode = "TEST"; // or "ROSENBROCK"
    auto [func, der, hess] = set_functions(mode);

    // Parameters
    double dt0 = 0.0;
    double dtmax = 0.1;

    // // Split Newton
    // auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "Starting Split-Newton...\n";

    // auto [xf_split, step_split, iter_split] = split_newton(
    //     der, hess, x0, x0.size() / 2, std::numeric_limits<size_t>::max(), false, dt0, dtmax, false, nullptr, 0.8);

    // auto end = std::chrono::high_resolution_clock::now();
    // double elapsed = std::chrono::duration<double>(end - start).count();

    // std::cout << "Final root (Split-Newton): ";
    // for (size_t i = 0; i < std::min(xf_split.size(), size_t(10)); ++i)
    //     std::cout << xf_split[i] << " ";
    // std::cout << "...\n";
    // std::cout << "Final Residual (Split-Newton): ";
    // for (const auto &res : func(xf_split))
    //     std::cout << res << " ";
    // std::cout << "\nElapsed time: " << elapsed << " seconds\n";
    // std::cout << "Total iterations: " << iter_split << "\n";

    // std::cout << std::string(20, '-') << "\n";

    // Newton
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Newton...\n";

    auto [xf_newton, step_newton, iter_newton] = newton(
        der, hess, x0, std::numeric_limits<size_t>::max(), false, dt0, dtmax, false, std::nullopt, 0.8);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Final root (Newton): ";
    for (size_t i = 0; i < std::min(static_cast<size_t>(xf_newton.size()), size_t(10)); ++i)
        std::cout << xf_newton[i] << " ";
    std::cout << "...\n";
    std::cout << "Final Residual (Newton): ";
    for (const auto &res : func(xf_newton))
        std::cout << res << " ";
    std::cout << "\nElapsed time: " << elapsed << " seconds\n";
    std::cout << "Total iterations: " << iter_newton << "\n";

    return 0;
}
