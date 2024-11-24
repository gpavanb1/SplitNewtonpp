#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include "typedefs.h"

// Test functions (equivalent to `demo_func.py`)
Vector test_func(const Vector &x)
{
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result[i] = std::sin(x[i]) - 0.5 * x[i]; // Example test function
    }
    return result;
}

Vector test_der(const Vector &x)
{
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result[i] = std::cos(x[i]) - 0.5; // Derivative of the test function
    }
    return result;
}

Matrix test_hess(const Vector &x)
{
    Matrix result = Eigen::MatrixXd::Zero(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        result(i, i) = -std::sin(x[i]); // Diagonal Hessian
    }
    return result;
}

// Function to set functions based on mode
FunctionSet set_functions(
    const std::string &mode)
{
    Func func;
    Gradient der;
    Jacobian hess;

    if (mode == "ROSENBROCK")
    {
        Func func = [](const Vector &x) -> Vector
        {
            Vector result(1);
            result[0] = 100 * std::pow((x[1] - x[0] * x[0]), 2) + std::pow((1 - x[0]), 2);
            return result;
        };

        auto der = [](const Vector &x) -> Vector
        {
            Vector result(2);
            result[0] = -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]);
            result[1] = 200 * (x[1] - x[0] * x[0]);
            return result;
        };

        auto hess = [](const Vector &x) -> Matrix
        {
            Matrix result(2, 2);
            result(0, 0) = -400 * (x[1] - 3 * x[0] * x[0]) + 2;
            result(0, 1) = -400 * x[0];
            result(1, 0) = -400 * x[0];
            result(1, 1) = 200;
            return result;
        };
        return {func, der, hess};
    }
    else
    {
        auto func = test_func;
        auto der = test_der;
        auto hess = test_hess;
        return {func, der, hess};
    }
}
