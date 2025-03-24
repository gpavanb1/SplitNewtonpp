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
#include <vector>
#include "typedefs.h"
#include "helper.hpp"
#include "newton.hpp"

// Function to attach two vectors
inline Vector attach(const Vector &x, const Vector &y)
{
    Vector result(x.size() + y.size());
    result << x, y;
    return result;
}

// Recursive Split Newton Method
inline std::tuple<Vector, Vector, int, int> split_newton(
    Gradient df, Jacobian J, const Vector &x0, const std::vector<int> &locs, int maxiter = std::numeric_limits<int>::max(),
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0, bool armijo = false,
    const Bounds &bounds = std::nullopt, double bound_fac = 0.8, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
{
    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax");
    }

    // Base case: If no more splits, solve directly with Newton
    if (locs.empty())
    {
        return newton(df, J, x0, maxiter, sparse, dt0, dtmax, armijo, bounds, bound_fac, jacobian_age, abs, rel);
    }

    // Get current split location
    int loc = locs[0];
    if (loc > x0.size())
    {
        throw std::invalid_argument("Incorrect split location");
    }

    // Split x0 into xa and xb
    Vector xa = x0.segment(0, loc);
    Vector xb = x0.segment(loc, x0.size() - loc);

    // Define residual and Jacobian for xa (keeping xb fixed)
    auto dfa = [&](const Vector &xa_local)
    {
        return df(attach(xa_local, xb)).segment(0, loc).eval();
    };

    auto Ja = [&](const Vector &xa_local)
    {
        Matrix Ja_matrix = J(attach(xa_local, xb)).block(0, 0, loc, loc);
        return Ja_matrix;
    };

    // Define residual and Jacobian for xb (keeping xa fixed)
    auto dfb = [&](const Vector &xb_local)
    {
        return df(attach(xa, xb_local)).segment(loc, x0.size() - loc).eval();
    };

    auto Jb = [&](const Vector &xb_local)
    {
        Matrix Jb_matrix = J(attach(xa, xb_local)).block(loc, loc, x0.size() - loc, x0.size() - loc);
        return Jb_matrix;
    };

    // Adjust locs for recursion (relative to xb)
    std::vector<int> new_locs(locs.begin() + 1, locs.end());
    for (int &l : new_locs)
        l -= loc;

    // Adjust bounds for recursion
    Bounds bounds_a = bounds ? std::make_optional(std::make_pair(
                                   bounds->first.segment(0, loc),
                                   bounds->second.segment(0, loc)))
                             : std::nullopt;

    Bounds bounds_b = bounds ? std::make_optional(std::make_pair(
                                   bounds->first.segment(loc, x0.size() - loc),
                                   bounds->second.segment(loc, x0.size() - loc)))
                             : std::nullopt;

    Vector x = x0;
    Vector s = Vector::Constant(x0.size(), std::numeric_limits<double>::infinity());
    double crit = std::numeric_limits<double>::infinity();
    int iter = 1;
    int status = 0;

    while (1)
    {
        // Solve the rightmost subsystem recursively
        auto [new_xb, sb, iter_b, status_b] = split_newton(dfb, Jb, xb, new_locs, maxiter, sparse, dt0, dtmax, armijo, bounds_b, bound_fac, jacobian_age, abs, rel);
        xb = new_xb;

        // One Newton step for left subsystem
        auto [new_xa, sa, iter_a, status_a] = newton(dfa, Ja, xa, 1, sparse, dt0, dtmax, armijo, bounds_a, bound_fac, jacobian_age, abs, rel);
        xa = new_xa;
        // If Newton failed miserably, return
        if (status_a < -1)
        {
            status = status_a;
            break;
        }

        // Construct full x and check convergence
        Vector xnew = attach(xa, xb);
        s = xnew - x;
        double fn = df(xnew).norm();
        crit = criterion(x, s, fn);

        spdlog::trace("Split-Newton: Iteration {}: Criterion = {}", iter, to_scientific(crit, 3));

        // Check if converged
        if (crit < 1.0)
        {
            status = 1;
            break;
        }

        // Reached maximum iterations
        if (iter >= maxiter)
        {
            spdlog::warn("Maximum Split-Newton iterations reached");
            status = -1;
            break;
        }

        x = xnew;
        iter++;
    }

    return {x, s, iter, status};
}

#endif // SPLIT_NEWTON_HPP