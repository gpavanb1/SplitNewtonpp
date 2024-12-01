#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include "typedefs.h"

// Test functions (equivalent to `demo_func.py`)
inline Vector test_func(const Vector &x)
{
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result[i] = std::sin(x[i]) - 0.5 * x[i]; // Example test function
    }
    return result;
}

inline Vector test_der(const Vector &x)
{
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result[i] = std::cos(x[i]) - 0.5; // Derivative of the test function
    }
    return result;
}

inline Matrix test_hess(const Vector &x)
{
    Matrix result = Eigen::MatrixXd::Zero(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result(i, i) = -std::sin(x[i]); // Diagonal Hessian
    }
    return result;
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