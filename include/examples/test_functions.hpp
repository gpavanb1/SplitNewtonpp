#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include "typedefs.h"
// Constants
constexpr double lambda_a = 6.0;
constexpr double lambda_b = 2.0;
constexpr double lambda_c = -1.0;
constexpr double lambda_d = -4.0;

// Helper function for creating log-spaced vectors
inline Vector logspace(double start, double end, int num)
{
    Vector result(num);
    for (int i = 0; i < num; ++i)
    {
        result[i] = std::pow(10, start + (end - start) * i / (num - 1));
    }
    return result;
}

// Helper function for common initialization
inline std::pair<Vector, Vector> common_init(const Vector &x0)
{
    int la = x0.size() / 2;
    int lb = x0.size() - la;

    // Generate the `a` and `b` coefficients using logspace
    Vector a = logspace(lambda_a, lambda_b, la);
    Vector b = logspace(lambda_c, lambda_d, lb);

    return {a, b};
}

// Test function: computes f(x0)
inline Vector test_func(const Vector &x0)
{
    auto [a, b] = common_init(x0);
    Vector coeff(a.size() + b.size());
    coeff << a, b; // Concatenate `a` and `b`

    return 0.25 * (coeff.array() * x0.array().pow(4));
}

// Gradient of the test function: computes f'(x0)
inline Vector test_der(const Vector &x0)
{
    auto [a, b] = common_init(x0);
    Vector coeff(a.size() + b.size());
    coeff << a, b; // Concatenate `a` and `b`

    return coeff.array() * x0.array().pow(3);
}

// Hessian of the test function: computes f''(x0)
inline Matrix test_hess(const Vector &x0)
{
    auto [a, b] = common_init(x0);
    Vector coeff(a.size() + b.size());
    coeff << a, b; // Concatenate `a` and `b`

    Vector diagonal = 3.0 * coeff.array() * x0.array().pow(2);

    return diagonal.asDiagonal();
}

inline Vector rosen_func(const Vector &x)
{
    size_t n = x.size();
    Vector result(n - 1);
    for (size_t i = 0; i < n - 1; ++i)
    {
        result[i] = 100 * std::pow((x[i + 1] - x[i] * x[i]), 2) + std::pow((1 - x[i]), 2);
    }
    return result;
};

inline Vector rosen_der(const Vector &x)
{
    size_t n = x.size();
    Vector result(n);
    result.setZero();
    for (size_t i = 0; i < n - 1; ++i)
    {
        result[i] += -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]);
        result[i + 1] += 200 * (x[i + 1] - x[i] * x[i]);
    }
    return result;
};

inline Matrix rosen_hess(const Vector &x)
{
    size_t n = x.size();
    Matrix result = Matrix::Zero(n, n);
    for (size_t i = 0; i < n - 1; ++i)
    {
        result(i, i) += -400 * (x[i + 1] - 3 * x[i] * x[i]) + 2;
        result(i, i + 1) += -400 * x[i];
        result(i + 1, i) += -400 * x[i];
        result(i + 1, i + 1) += 200;
    }
    return result;
};

// Function to set functions based on mode
inline FunctionSet set_functions(
    const std::string &mode)
{
    Func func;
    Gradient der;
    Jacobian hess;

    if (mode == "ROSENBROCK")
    {
        func = rosen_func;
        der = rosen_der;
        hess = rosen_hess;
    }
    else if (mode == "TEST")
    {
        func = test_func;
        der = test_der;
        hess = test_hess;
    }
    else
    {
        throw std::invalid_argument("Invalid mode specified");
    }
    return {func, der, hess};
}

#endif // TEST_FUNCTIONS_HPP