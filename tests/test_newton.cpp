#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <optional>
#include "splitnewton/newton.hpp"      // Ensure this header is the one with the newton code
#include "examples/test_functions.hpp" // Contains definitions for e.g. Rosenbrock functions

// ------------------
// Helper Function Tests
// ------------------

// Test weighted norm calculation for a single point and multiple points.
TEST(Norm2Test, SinglePoint)
{
    Vector x(2), s(2);
    x << 1.0, 2.0;
    s << 0.1, 0.2;
    // Manually compute expected norm:
    // For each component: weight = rel*(|x|) + abs. Here rel=1e-5, abs=1e-5.
    double expected = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        double weight = 1e-5 * std::fabs(x(i)) + 1e-5;
        expected += std::pow(s(i) / weight, 2);
    }
    expected = std::sqrt(expected);
    double result = norm2(x, s); // uses default npts, abs, rel
    ASSERT_NEAR(result, expected, 1e-9);
}

TEST(Norm2Test, MultiplePoints)
{
    // Create a vector representing 2 variables at 3 points (total size 6)
    Vector x(6), s(6);
    double abs = 0.0;
    double rel = 1e-5;
    x << 1, 2, 3, 4, 5, 6;
    s << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    // npts = 3, so each variable appears 3 times.
    double result = norm2(x, s, 3, abs, rel);
    double expected = 27182.51;
    // We perform a rough check: result should be positive and finite.
    ASSERT_GT(result, 0.0);
    ASSERT_NEAR(result, expected, 1.0);
}

// ------------------
// Bounds Checking Tests
// ------------------

TEST(BoundsTest, ValidBounds)
{
    Vector x(3);
    x << 1.0, 2.0, 3.0;
    Vector lower(3), upper(3);
    lower << 0.0, 1.5, 2.5;
    upper << 2.0, 3.0, 4.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_TRUE(check_within_bounds(x, bounds));
}

TEST(BoundsTest, InvalidBounds)
{
    Vector x(3);
    x << 1.0, 2.0, 5.0;
    Vector lower(3), upper(3);
    lower << 0.0, 1.5, 2.5;
    upper << 2.0, 3.0, 4.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_FALSE(check_within_bounds(x, bounds));
}

TEST(BoundsTest, NoBoundsProvided)
{
    Vector x(2);
    x << 1.0, 2.0;
    ASSERT_TRUE(check_within_bounds(x, std::nullopt));
}

TEST(BoundsTest, MismatchedBoundsSize)
{
    Vector x(2);
    x << 1.0, 2.0;
    Vector lower(1), upper(1);
    lower << 0.0;
    upper << 1.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    ASSERT_THROW(check_within_bounds(x, bounds), std::invalid_argument);
}

// ------------------
// Scaling Factor Test
// ------------------

TEST(ScalingTest, ComputeBoundsScaling)
{
    Vector x(3), s(3);
    x << 1.8, 2.0, 3.0;
    s << 0.5, -0.5, 0.25;
    // Set bounds such that one component is at its limit.
    Vector lower(3), upper(3);
    lower << 0.0, 1.0, 2.5;
    upper << 2.0, 2.5, 3.5;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));

    // For each component compute legal delta and scaling:
    double scale0 = ((s(0) > 0) ? (upper(0) - x(0)) : (lower(0) - x(0))) / s(0);
    double scale1 = ((s(1) > 0) ? (upper(1) - x(1)) : (lower(1) - x(1))) / s(1);
    double scale2 = ((s(2) > 0) ? (upper(2) - x(2)) : (lower(2) - x(2))) / s(2);
    double expected = std::min({scale0, scale1, scale2});
    double computed = compute_bounds_scaling(x, s, bounds);
    ASSERT_NEAR(computed, expected, 1e-9);
}

TEST(ScalingTest, NoBoundsScaling)
{
    Vector x(2), s(2);
    x << 1.0, 2.0;
    s << 0.3, -0.4;
    double computed = compute_bounds_scaling(x, s, std::nullopt);
    ASSERT_NEAR(computed, 1.0, 1e-9);
}

// ------------------
// Linear Solver Tests
// ------------------

TEST(LinearSolveTest, DenseLinearSolveIdentity)
{
    Matrix jac = 2.0 * Matrix::Identity(3, 3);
    Vector dfx(3);
    dfx << 1, -2, 3;
    // The expected solution is -dfx since solve returns s and then newton negates it.
    auto [s, status] = dense_linear_solve(jac, dfx);
    ASSERT_EQ(status, 1);
    Vector expected = -dfx / 2.0;
    ASSERT_TRUE(s.isApprox(expected, 1e-9));
}

TEST(LinearSolveTest, SparseLinearSolveIdentity)
{
    Matrix dense = 2.0 * Matrix::Identity(4, 4);
    Eigen::SparseMatrix<double> jac = dense.sparseView();
    Vector dfx(4);
    dfx << 2, -1, 0, 4;
    auto [s, status] = sparse_linear_solve(jac, dfx);
    ASSERT_EQ(status, 1);
    Vector expected = -dfx / 2.0;
    ASSERT_TRUE(s.isApprox(expected, 1e-9));
}

// ------------------
// Damped Step Tests
// ------------------

// A simple test where the function is linear so that no damping is actually required.
TEST(DampStepTest, NoDampingNeeded)
{
    Vector x(2);
    x << 1.0, 1.0;
    Vector s(2);
    s << -0.5, -0.5;
    // Define a simple gradient function: f(x) = x - 0.5 (zero at 0.5)
    auto df = [](const Vector &x_val) -> Vector
    {
        return x_val - Vector::Constant(x_val.size(), 0.5);
    };
    Matrix jac = Matrix::Identity(2, 2);
    Bounds bounds = std::nullopt;
    auto [x1, step1, status] = damp_step(jac, df, x, s, bounds);
    // Since the update should reduce the error, status should be 1 (converged) or 0 (acceptable step)
    ASSERT_TRUE(status == 0 || status == 1);
    // Ensure the new iterate x1 is closer to 0.5
    ASSERT_LT((x1 - Vector::Constant(2, 0.5)).norm(), (x - Vector::Constant(2, 0.5)).norm());
}

// Test that if the bounds force a very small step the function returns status -2.
TEST(DampStepTest, DampingTooSmallDueToBounds)
{
    Vector x(2);
    x << 0.0, 0.0;
    Vector s(2);
    s << 1.0, 1.0;
    // Set bounds that exactly equal the seed point.
    Vector lower = x;
    Vector upper = x;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    auto df = [](const Vector &x_val) -> Vector
    {
        return x_val + Vector::Ones(x_val.size());
    };
    Matrix jac = Matrix::Identity(2, 2);
    auto [x1, step1, status] = damp_step(jac, df, x, s, bounds);
    ASSERT_EQ(status, -2);
}

// ------------------
// Newton Method Tests
// ------------------

// Test a simple linear system f(x) = x - target using dense linear solve.
TEST(NewtonMethodTest, SimpleLinearConvergence)
{
    Vector target(2);
    target << 1.0, 1.0;
    Vector x0(2);
    x0 << 4.0, 4.0;
    auto df = [target](const Vector &x) -> Vector
    {
        return x - target;
    };
    auto J = [](const Vector &x) -> Matrix
    {
        return Matrix::Identity(x.size(), x.size());
    };
    // Use dt0 > 0 so that timestep updates occur.
    auto [x, step, iter, status] = newton(df, J, x0, 50, 1, false, 0.1, 1.0, std::nullopt, 5);
    // Expect convergence with a solution near target.
    ASSERT_EQ(status, 1);
    ASSERT_TRUE(x.isApprox(target, 1e-4));
}

// Test Newton on the Rosenbrock function (a classic nonlinear test problem).
TEST(NewtonMethodTest, RosenbrockConvergence)
{
    // Rosenbrock function: minimum at (1,1)
    Vector x0(2);
    x0 << 0.1, 0.2;
    auto df = [](const Vector &x) -> Vector
    {
        return rosen_der(x); // Provided by test_functions.hpp
    };
    auto J = [](const Vector &x) -> Matrix
    {
        return rosen_hess(x); // Provided by test_functions.hpp
    };
    auto [x, step, iter, status] = newton(df, J, x0, 100, 1, false, 0.0, 1.0, std::nullopt, 1);
    ASSERT_EQ(status, 1);
    ASSERT_TRUE(x.isApprox(Vector::Ones(x.size()), 1e-4));
}

// Test that an out-of-bounds seed results in an exception.
TEST(NewtonMethodTest, OutOfBoundsSeedThrows)
{
    Vector x0(2);
    x0 << 2.0, 2.0;
    Vector lower(2), upper(2);
    lower << 0.0, 0.0;
    upper << 1.0, 1.0;
    Bounds bounds = std::make_optional(std::make_pair(lower, upper));
    auto df = [](const Vector &x) -> Vector
    {
        return x; // dummy function
    };
    auto J = [](const Vector &x) -> Matrix
    {
        return Matrix::Identity(x.size(), x.size());
    };
    ASSERT_THROW(newton(df, J, x0, 100, 1, false, 0.1, 1.0, bounds), std::invalid_argument);
}

// Test invalid timestep parameters throw exceptions.
TEST(NewtonMethodTest, InvalidTimeStepThrows)
{
    Vector x0(2);
    x0 << 1.0, 1.0;
    auto df = [](const Vector &x) -> Vector
    {
        return x - Vector::Ones(x.size());
    };
    auto J = [](const Vector &x) -> Matrix
    {
        return Matrix::Identity(x.size(), x.size());
    };
    // dt0 is negative.
    ASSERT_THROW(newton(df, J, x0, 50, 1, false, -0.1, 1.0, std::nullopt), std::invalid_argument);
}

// Test that damping reduces the step size.
TEST(NewtonMethodTest, DampingReducesStep)
{
    // Use a function that will require damping
    Vector x0(1);
    x0 << 0.0;
    auto df = [](const Vector &x) -> Vector
    {
        return x - Vector::Ones(x.size());
    };
    auto J = [](const Vector &x) -> Matrix
    {
        return Matrix::Identity(x.size(), x.size());
    };
    auto [x, step, iter, status] = newton(df, J, x0, 10, 1, false, 0.1, 1.0, std::nullopt);
    // Check that the computed step (after damping) is less than the undamped value of 1.
    ASSERT_LT(step.norm(), 1.0);
}
