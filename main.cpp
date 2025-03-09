#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>
#include <spdlog/spdlog.h>
#include "splitnewton/typedefs.h"
#include "splitnewton/newton.hpp"
#include "splitnewton/split_newton.hpp"
#include "examples/test_functions.hpp"

// Helper function for formatting numbers in scientific notation
std::string to_scientific_string(double num)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << num;
    return oss.str();
}

int main(int argc, char *argv[])
{
    // Logging level (simple argument parsing)
    spdlog::set_level(spdlog::level::trace);
    if (argc > 1)
    {
        spdlog::set_level(spdlog::level::from_str(argv[1]));
        std::cout << "Log level set to: " << argv[1] << "\n";
    }

    // Seed
    Vector x0 = Eigen::VectorXd::LinSpaced(5000, 21.2, 31.2);

    // Mode selection
    std::string mode = "TEST"; // or "TEST"
    auto [func, der, hess] = set_functions(mode);

    // Parameters
    double dt0 = 0.0;
    double dtmax = 0.1;

    // Split Newton
    auto start = std::chrono::high_resolution_clock::now();
    spdlog::info("Starting Split-Newton...\n");

    std::vector<int> locs = {int(x0.size() / 3), int(2 * x0.size() / 3)};

    auto [xf_split, step_split, iter_split, status_split] = split_newton(
        der, hess, x0, locs, std::numeric_limits<int>::max(), true, dt0, dtmax, false, std::nullopt, 0.8, 2);
    assert(status_split == 1);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::string log_str;
    spdlog::info("Final root (Split-Newton): ");
    for (size_t i = 0; i < std::min(static_cast<size_t>(xf_split.size()), size_t(10)); ++i)
        log_str += to_scientific_string(xf_split[i]) + ", ";
    spdlog::info("{}", log_str);
    spdlog::info("...\n");
    spdlog::info("Final Residual (Split-Newton): ");
    log_str = "";
    for (const auto &res : func(xf_split))
        log_str += to_scientific_string(res) + ", ";
    spdlog::info("{}", log_str);
    spdlog::info("\nElapsed time: " + std::to_string(elapsed) + " seconds\n");
    spdlog::info("Total iterations: " + std::to_string(iter_split) + "\n");

    std::cout << std::string(20, '-') << "\n";

    // Newton
    start = std::chrono::high_resolution_clock::now();
    spdlog::info("Starting Newton...\n");

    auto [xf_newton, step_newton, iter_newton, status_newton] = newton(
        der, hess, x0, std::numeric_limits<int>::max(), true, dt0, dtmax, false, std::nullopt, 0.8, false, 2);
    assert(status_newton == 1);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(end - start).count();

    spdlog::info("Final root (Newton): ");
    log_str = "";
    for (size_t i = 0; i < std::min(static_cast<size_t>(xf_newton.size()), size_t(10)); ++i)
        log_str += to_scientific_string(xf_newton[i]) + ", ";
    spdlog::info("{}", log_str);
    spdlog::info("...\n");
    spdlog::info("Final Residual (Newton): ");
    log_str = "";
    for (const auto &res : func(xf_newton))
        log_str += to_scientific_string(res) + ", ";
    spdlog::info("{}", log_str);
    spdlog::info("\nElapsed time: " + std::to_string(elapsed) + " seconds\n");
    spdlog::info("Total iterations: " + std::to_string(iter_newton) + "\n");

    // Check if xf_split and xf_newton are close
    spdlog::info("Maximum difference between solutions: " + std::to_string((xf_split - xf_newton).cwiseAbs().maxCoeff()));

    return 0;
}
