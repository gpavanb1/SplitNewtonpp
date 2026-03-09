#include <vector>
#include <limits>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include "examples/test_functions.hpp"
#include "splitnewton/newton.hpp"
#include "splitnewton/split_newton.hpp"
#include "splitnewton/options.hpp"

class BlockNewtonTest : public ::testing::Test
{
protected:
    const int num_elements = 500;
    Vector x0 = Eigen::VectorXd::LinSpaced(num_elements, 21.2, 31.2);
    Bounds bounds = std::make_optional(std::make_pair(
        Vector::Constant(num_elements, -50.0),
        Vector::Constant(num_elements, 50.0)));
    std::vector<int> loc = {int(num_elements / 2)};
    Func func;
    Gradient der;
    Jacobian hess;

    void SetUp() override
    {
        std::tie(func, der, hess) = set_functions("TEST");
        char* argv[] = {(char*)"test"};
        splitnewton::initialize(1, argv);
    }
};

TEST_F(BlockNewtonTest, JacobiNewtonFlag)
{
    char* argv[] = {(char*)"test", (char*)"-use_jacobi"};
    splitnewton::initialize(2, argv);
    bool sparse = true;
    auto [x_opt_newton, step_newton, iterations_newton, status_newton] = newton(
        der, hess, x0, std::numeric_limits<int>::max(), 1,
        sparse, 0.0, 0.1, bounds, 1);
    auto [x_opt, step, iterations, status] = split_newton(
        der, hess, x0, loc, std::numeric_limits<int>::max(), 1,
        sparse, 0.0, 0.1, bounds, 1);
    EXPECT_LE((x_opt_newton - x_opt).cwiseAbs().maxCoeff(), 2e-4);
    ASSERT_EQ(status_newton, 1);
    ASSERT_EQ(status, 1);
}

TEST_F(BlockNewtonTest, GaussSeidelNewtonFlag)
{
    char* argv[] = {(char*)"test", (char*)"-use_gauss_seidel"};
    splitnewton::initialize(2, argv);
    bool sparse = true;
    auto [x_opt_newton, step_newton, iterations_newton, status_newton] = newton(
        der, hess, x0, std::numeric_limits<int>::max(), 1,
        sparse, 0.0, 0.1, bounds, 1);
    auto [x_opt, step, iterations, status] = split_newton(
        der, hess, x0, loc, std::numeric_limits<int>::max(), 1,
        sparse, 0.0, 0.1, bounds, 1);
    EXPECT_LE((x_opt_newton - x_opt).cwiseAbs().maxCoeff(), 2e-4);
    ASSERT_EQ(status_newton, 1);
    ASSERT_EQ(status, 1);
}
