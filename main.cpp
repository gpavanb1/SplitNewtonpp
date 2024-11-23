#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <stdexcept>
#include "newton.hpp"
#include "split_newton.hpp"

// Test functions (equivalent to `demo_func.py`)
std::vector<double> test_func(const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = std::sin(x[i]) - 0.5 * x[i]; // Example test function
    }
    return result;
}

std::vector<double> test_der(const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = std::cos(x[i]) - 0.5; // Derivative of the test function
    }
    return result;
}

std::vector<std::vector<double>> test_hess(const std::vector<double> &x)
{
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(x.size(), 0.0));
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i][i] = -std::sin(x[i]); // Diagonal Hessian
    }
    return result;
}

// Function to set functions based on mode
void set_functions(
    const std::string &mode,
    std::function<std::vector<double>(const std::vector<double> &)> &func,
    std::function<std::vector<double>(const std::vector<double> &)> &der,
    std::function<std::vector<std::vector<double>>(const std::vector<double> &)> &hess)
{
    if (mode == "ROSENBROCK")
    {
        func = [](const std::vector<double> &x) -> std::vector<double>
        {
            double result = 100 * std::pow((x[1] - x[0] * x[0]), 2) + std::pow((1 - x[0]), 2);
            return std::vector<double>{result};
        };
        der = [](const std::vector<double> &x) -> std::vector<double>
        {
            return std::vector<double>{
                -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
                200 * (x[1] - x[0] * x[0])};
        };
        hess = [](const std::vector<double> &x) -> std::vector<std::vector<double>>
        {
            return std::vector<std::vector<double>>{
                {-400 * (x[1] - 3 * x[0] * x[0]) + 2, -400 * x[0]},
                {-400 * x[0], 200}};
        };
    }
    else
    {
        func = test_func;
        der = test_der;
        hess = test_hess;
    }
}

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
    std::vector<double> x0(5000);
    for (size_t i = 0; i < x0.size(); ++i)
    {
        x0[i] = 21.2 + i * (31.2 - 21.2) / (x0.size() - 1);
    }

    // Mode selection
    std::string mode = "TEST"; // or "ROSENBROCK"
    std::function<std::vector<double>(const std::vector<double> &)> func;
    std::function<std::vector<double>(const std::vector<double> &)> der;
    std::function<std::vector<std::vector<double>>(const std::vector<double> &)> hess;
    set_functions(mode, func, der, hess);

    // Parameters
    double dt0 = 0.0;
    double dtmax = 0.1;

    // Split Newton
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Split-Newton...\n";

    auto [xf_split, step_split, iter_split] = split_newton(
        der, hess, x0, x0.size() / 2, std::numeric_limits<size_t>::max(), false, dt0, dtmax, false, nullptr, 0.8);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Final root (Split-Newton): ";
    for (size_t i = 0; i < std::min(xf_split.size(), size_t(10)); ++i)
        std::cout << xf_split[i] << " ";
    std::cout << "...\n";
    std::cout << "Final Residual (Split-Newton): ";
    for (const auto &res : func(xf_split))
        std::cout << res << " ";
    std::cout << "\nElapsed time: " << elapsed << " seconds\n";
    std::cout << "Total iterations: " << iter_split << "\n";

    std::cout << std::string(20, '-') << "\n";

    // Newton
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Newton...\n";

    auto [xf_newton, step_newton, iter_newton] = newton(
        der, hess, x0, std::numeric_limits<size_t>::max(), false, dt0, dtmax, false, nullptr, 0.8);

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Final root (Newton): ";
    for (size_t i = 0; i < std::min(xf_newton.size(), size_t(10)); ++i)
        std::cout << xf_newton[i] << " ";
    std::cout << "...\n";
    std::cout << "Final Residual (Newton): ";
    for (const auto &res : func(xf_newton))
        std::cout << res << " ";
    std::cout << "\nElapsed time: " << elapsed << " seconds\n";
    std::cout << "Total iterations: " << iter_newton << "\n";

    return 0;
}
