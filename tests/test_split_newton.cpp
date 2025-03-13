#include <vector>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include "examples/test_functions.hpp"
#include "splitnewton/split_newton.hpp"

class SplitNewtonTest : public ::testing::Test
{
protected:
    // Initial x0
    const int num_elements = 500;
    Vector x0 = Eigen::VectorXd::LinSpaced(num_elements, 21.2, 31.2);

    // Bounds for the test
    Bounds bounds = std::make_optional(std::make_pair(
        Vector::Constant(num_elements, -50.0),  // Lower bounds
        Vector::Constant(num_elements, 50.0))); // Upper bounds

    std::vector<int> loc = {int(num_elements / 2)}; // Split location

    // Mode selection
    std::string mode = "TEST"; // or "TEST"

    // Function pointers for the test
    Func func;
    Gradient der;
    Jacobian hess;

    // Set up the test fixture
    void SetUp() override
    {
        // Initialize the functions based on the selected mode
        std::tie(func, der, hess) = set_functions(mode);
    }
};

// Test comparing newton and split_newton for sparse case
TEST_F(SplitNewtonTest, NewtonVsSplitNewtonSparse)
{
    bool sparse = true;

    spdlog::set_level(spdlog::level::info);

    // Run newton solver
    auto [x_opt_newton, step_newton, iterations_newton, status_newton] = newton(
        der, hess, x0, std::numeric_limits<int>::max(),
        sparse, 0.0, 0.1, false, bounds, 0.8, 1);

    // Run split_newton solver
    auto [x_opt_split_newton, step_split_newton, iterations_split_newton, status_split_newton] = split_newton(
        der, hess, x0, loc, std::numeric_limits<int>::max(),
        sparse, 0.0, 0.1, false, bounds, 0.8, 1);

    // Compare results
    EXPECT_LE((x_opt_newton - x_opt_split_newton).cwiseAbs().maxCoeff(), 2e-4);
    ASSERT_TRUE(status_newton == 1);
    ASSERT_TRUE(status_split_newton == 1);
}

// Test comparing newton and split_newton for dense case
TEST_F(SplitNewtonTest, NewtonVsSplitNewtonDense)
{
    bool sparse = false;

    // Run newton solver
    auto [x_opt_newton, step_newton, iterations_newton, status_newton] = newton(
        der, hess, x0, std::numeric_limits<int>::max(),
        sparse, 0.0, 0.1, false, bounds, 0.8, 1);

    // Run split_newton solver
    auto [x_opt_split_newton, step_split_newton, iterations_split_newton, status_split_newton] = split_newton(
        der, hess, x0, loc, std::numeric_limits<int>::max(),
        sparse, 0.0, 0.1, false, bounds, 0.8, 1);

    // Compare results
    EXPECT_LE((x_opt_newton - x_opt_split_newton).cwiseAbs().maxCoeff(), 2e-4);
    ASSERT_TRUE(status_newton == 1);
    ASSERT_TRUE(status_split_newton == 1);
}

// Test negative dt0 and dtmax exception
TEST_F(SplitNewtonTest, NegativeDtException)
{

    // Test negative dt0
    EXPECT_THROW({
        try {
            split_newton(der, hess, x0, loc, 
                std::numeric_limits<int>::max(), false, -0.1, 1.0);
        } catch (const std::invalid_argument& e) {
            EXPECT_STREQ("Must specify positive dt0 and dtmax", e.what());
            throw;
        } }, std::invalid_argument);

    // Test negative dtmax
    EXPECT_THROW({
        try {
            split_newton(der, hess, x0, loc, 
                std::numeric_limits<int>::max(), false, 0.1, -1.0);
        } catch (const std::invalid_argument& e) {
            EXPECT_STREQ("Must specify positive dt0 and dtmax", e.what());
            throw;
        } }, std::invalid_argument);
}

// Test incorrect split location exception
TEST_F(SplitNewtonTest, IncorrectSplitLocationException)
{
    std::vector<int> loc = {num_elements + 1}; // Incorrect location, greater than length of x0

    EXPECT_THROW({
        try {
            split_newton(der, hess, x0, loc);
        } catch (const std::invalid_argument& e) {
            EXPECT_STREQ("Incorrect split location", e.what());
            throw;
        } }, std::invalid_argument);
}

// Test attach function
TEST(AttachTest, VectorAttachment)
{
    Vector x1 = Vector::Constant(2, 1.0);
    Vector x2 = Vector::Constant(3, 2.0);

    Vector result = attach(x1, x2);

    EXPECT_EQ(result.size(), 5);
    EXPECT_DOUBLE_EQ(result(0), 1.0);
    EXPECT_DOUBLE_EQ(result(1), 1.0);
    EXPECT_DOUBLE_EQ(result(2), 2.0);
    EXPECT_DOUBLE_EQ(result(3), 2.0);
    EXPECT_DOUBLE_EQ(result(4), 2.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}