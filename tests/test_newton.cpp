#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <optional>
#include "splitnewton/newton.hpp"
#include "examples/test_functions.hpp"

// Criterion test
TEST(CriterionTest, CalculatesCorrectly)
{
    Vector x(2), s(2);
    x << 1.0, 2.0;
    s << 0.1, 0.2;
    double fn = 0.01;
    double result = criterion(x, s, fn);
    double expected = (s.array() / (x.array() * 1e-6 + 1e-5)).matrix().norm();
    ASSERT_NEAR(result, expected, 1e-9);
}

// Check bounds tests
TEST(CheckBoundsTest, ValidBounds)
{
    Vector x(3);
    x << 1.0, 2.0, 3.0;
    Vector lower(3), upper(3);
    lower << 0.0, 1.5, 2.5;
    upper << 2.0, 3.0, 4.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_TRUE(check_within_bounds(x, bounds));
}

TEST(CheckBoundsTest, InvalidBounds)
{
    Vector x(3);
    x << 1.0, 2.0, 5.0;
    Vector lower(3), upper(3);
    lower << 0.0, 1.5, 2.5;
    upper << 2.0, 3.0, 4.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_FALSE(check_within_bounds(x, bounds));
}

TEST(CheckBoundsTest, NoBounds)
{
    Vector x(2);
    x << 1.0, 2.0;
    ASSERT_TRUE(check_within_bounds(x, std::nullopt));
}

TEST(CheckBoundsTest, BoundsSizeMismatch)
{
    Vector x(2);
    x << 1.0, 2.0;
    Vector lower(1), upper(1);
    lower << 0.0;
    upper << 1.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_THROW(check_within_bounds(x, bounds), std::invalid_argument);
}

// Newton tests
TEST(NewtonTest, InvalidDt0)
{
    Vector x0(2);
    x0 << 0.1, 0.2;
    ASSERT_THROW(newton([](const Vector &)
                        { return Vector(2); },
                        [](const Vector &)
                        { return Matrix(2, 2); },
                        x0, 100, false, -1.0),
                 std::invalid_argument);
}

TEST(NewtonTest, SimpleConvergence)
{
    Vector x0(2);
    x0 << 4.0, 4.0;
    Vector expected_step(2);
    expected_step << -2.0, -2.0;

    auto df = [](const Vector &x)
    { return x - Vector::Ones(x.size()); };
    auto J = [](const Vector &x)
    { return Matrix::Identity(x.size(), x.size()); };

    auto [x, step, iter] = newton(df, J, x0, 1, false, 2.0, 2.0, false, std::nullopt, 0.8, true);
    ASSERT_TRUE(step.isApprox(expected_step, 1e-5));
}

TEST(NewtonTest, OutOfBoundsSeed)
{
    Vector x0(2);
    x0 << 2.0, 2.0;
    Vector lower(2), upper(2);
    lower << 0.0, 0.0;
    upper << 1.0, 1.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_THROW(newton([](const Vector &)
                        { return Vector(2); },
                        [](const Vector &)
                        { return Matrix(2, 2); },
                        x0, 100, false, 0.1, 1.0, false, bounds),
                 std::invalid_argument);
}

TEST(NewtonTest, RosenbrockConvergence)
{
    Vector x0(2);
    x0 << 0.1, 0.2;

    auto df = [](const Vector &x)
    { return rosen_der(x); };
    auto J = [](const Vector &x)
    { return rosen_hess(x); };

    auto [x, s, iter] = newton(df, J, x0, 100, false, 0.0, 1.0, false, std::nullopt, 0.8, false, 1);
    ASSERT_TRUE(x.isApprox(Vector::Ones(x.size()), 1e-5));
}

TEST(NewtonTest, ExactBounds)
{
    Vector x0(2);
    x0 << 0.0, 0.0;
    Vector lower(2), upper(2);
    lower << 0.0, 0.0;
    upper << 0.0, 0.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));

    auto df = [](const Vector &)
    { return Vector::Zero(2); };
    auto J = [](const Vector &)
    { return Matrix::Identity(2, 2); };

    auto [x, s, iter] = newton(df, J, x0, 1, false, 0.1, 1.0, false, bounds);
    ASSERT_TRUE(s.isApprox(Vector::Zero(2), 1e-5));
}

TEST(NewtonTest, ArmijoRule)
{
    Vector x0(1);
    x0 << 0.0;

    auto df = [](const Vector &x)
    { return x - Vector::Ones(x.size()); };
    auto J = [](const Vector &x)
    { return Matrix::Identity(x.size(), x.size()); };

    auto [x, step, iter] = newton(df, J, x0, 1, false, 0.1, 1.0, true, std::nullopt, 0.8, true);
    ASSERT_LT(step.norm(), 1.0); // Armijo scaling should reduce the step size
}
