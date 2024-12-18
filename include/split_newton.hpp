#ifndef SPLIT_NEWTON_HPP
#define SPLIT_NEWTON_HPP

#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include "typedefs.h"
#include "newton.hpp"

// Function to attach two vectors
Vector attach(const Vector &x, const Vector &y)
{
    Vector result(x.size() + y.size());
    result << x, y;
    return result;
}

// Split Newton Method
std::tuple<Vector, Vector, int> split_newton(
    Gradient df, Jacobian J, const Vector &x0, int loc, int maxiter = std::numeric_limits<int>::max(),
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0, bool armijo = false,
    const Bounds &bounds = std::nullopt, double bound_fac = 0.8, int jacobian_age = 5)
{

    /**
     * @brief Applies the Split Newton method to solve for a root of a function.
     *
     * This function uses the Split Newton method to find the root of a nonlinear equation
     * given the gradient and Jacobian. It provides several options to control the iteration
     * process, such as step size, maximum iterations, and boundary conditions.
     *
     * @param df A function representing the gradient of the objective function.
     * @param J A function representing the Jacobian of the objective function.
     * @param x0 The initial guess for the root.
     * @param loc The location or index where the vector/system is being split.
     * @param maxiter The maximum number of iterations (default is
     *        `std::numeric_limits<int>::max()`).
     * @param sparse Whether to use sparse matrices for the Jacobian (default is `false`).
     * @param dt0 Initial step size (default is `0.0`).
     * @param dtmax Maximum allowable step size (default is `1.0`).
     * @param armijo Whether to use the Armijo rule for step size selection (default
     *        is `false`).
     * @param bounds Optional bounds for the solution, represented by a `Bounds`
     *        object - `std::pair` of `Vector` (default is `std::nullopt`).
     * @param bound_fac Factor for damping the step size on hitting bounds during the
     *        iterations (default is `0.8`).
     * @param jacobian_age Number of iterations after which the Jacobian is updated
     *        (default is `5`).
     *
     * @return A tuple containing:
     *         - The computed solution vector.
     *         - The computed gradient at the solution.
     *         - The number of iterations performed.
     */

    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax");
    }
    if (loc > x0.size())
    {
        throw std::invalid_argument("Incorrect split location");
    }

    double dt = dt0;

    // Initial split of x0 into xa and xb
    Vector xa = x0.segment(0, loc);
    Vector xb = x0.segment(loc, x0.size() - loc);
    Vector x = x0;

    Vector s = Vector::Constant(x0.size(), std::numeric_limits<double>::infinity());
    double crit = std::numeric_limits<double>::infinity();

    int iter = 0;
    while (crit >= 1.0 && iter < maxiter)
    {
        // B Cycle
        auto dfb = [&](const Vector &xb_local)
        {
            return df(attach(xa, xb_local)).segment(loc, x0.size() - loc);
        };

        auto Jb = [&](const Vector &xb_local)
        {
            Matrix Jb_matrix = J(attach(xa, xb_local)).block(loc, loc, x0.size() - loc, x0.size() - loc);
            return Jb_matrix;
        };

        Bounds local_bounds = bounds ? std::make_optional(std::make_pair(
                                           bounds->first.segment(loc, x0.size() - loc),
                                           bounds->second.segment(loc, x0.size() - loc)))
                                     : std::nullopt;

        auto [new_xb, sb, local_iter_b] = newton(
            dfb, Jb, xb, maxiter, sparse, dt, dtmax, armijo, local_bounds, bound_fac, false, jacobian_age);
        xb = new_xb;
        spdlog::debug("B iterations: {}", local_iter_b);

        // A Cycle
        auto dfa = [&](const Vector &xa_local)
        {
            return df(attach(xa_local, xb)).segment(0, loc);
        };

        auto Ja = [&](const Vector &xa_local)
        {
            Matrix Ja_matrix = J(attach(xa_local, xb)).block(0, 0, loc, loc);
            return Ja_matrix;
        };

        local_bounds = bounds ? std::make_optional(std::make_pair(
                                    bounds->first.segment(0, loc),
                                    bounds->second.segment(0, loc)))
                              : std::nullopt;

        auto [new_xa, sa, local_iter_a] = newton(
            dfa, Ja, xa, 1, sparse, dt, dtmax, armijo, local_bounds, bound_fac, true, jacobian_age);
        xa = new_xa;
        spdlog::debug("A iterations: {}", local_iter_a);

        // Construct new x and step
        Vector xnew = attach(xa, xb);
        s = xnew - x;

        // Check convergence
        double fn = df(xnew).norm();
        crit = criterion(x, s, fn);
        spdlog::trace("Iteration " + std::to_string(iter) + ": Criterion = " + std::to_string(crit));

        // Update x
        x = xnew;
        iter++;
    }

    return {x, s, iter};
}

#endif // SPLITNEWTON_HPP
