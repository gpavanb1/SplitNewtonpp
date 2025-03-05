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
#include "newton.hpp"

// Function to attach two vectors
inline Vector attach(const Vector &x, const Vector &y)
{
    Vector result(x.size() + y.size());
    result << x, y;
    return result;
}

// Recursive Split Newton Method
inline std::tuple<Vector, Vector, int> split_newton(
    Gradient df, Jacobian J, const Vector &x0, const std::vector<int> &locs, int maxiter = std::numeric_limits<int>::max(),
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0, bool armijo = false,
    const Bounds &bounds = std::nullopt, double bound_fac = 0.8, int jacobian_age = 5)
{
    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax");
    }

    // Base case: If no more splits, solve directly with Newton
    if (locs.empty())
    {
        return newton(df, J, x0, maxiter, sparse, dt0, dtmax, armijo, bounds, bound_fac, false, jacobian_age);
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
    int iter = 0;

    while (crit >= 1.0 && iter < maxiter)
    {
        // Solve the rightmost subsystem recursively
        auto [new_xb, sb, iter_b] = split_newton(dfb, Jb, xb, new_locs, maxiter, sparse, dt0, dtmax, armijo, bounds_b, bound_fac, jacobian_age);
        xb = new_xb;

        // One Newton step for left subsystem
        auto [new_xa, sa, iter_a] = newton(dfa, Ja, xa, 1, sparse, dt0, dtmax, armijo, bounds_a, bound_fac, true, jacobian_age);
        xa = new_xa;

        // Construct full x and check convergence
        Vector xnew = attach(xa, xb);
        s = xnew - x;
        double fn = df(xnew).norm();
        crit = criterion(x, s, fn);

        spdlog::trace("Iteration {}: Criterion = {}", iter, crit);

        x = xnew;
        iter++;
    }

    return {x, s, iter};
}

#endif // SPLIT_NEWTON_HPP