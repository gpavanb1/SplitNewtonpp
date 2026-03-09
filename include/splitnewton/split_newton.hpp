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
#include "shuffler.hpp"
#include "block/jacobi.hpp"
#include "block/gauss_seidel.hpp"
#include "options.hpp"

// Function to attach two vectors
inline Vector attach(const Vector& x, const Vector& y)
{
    Vector result(x.size() + y.size());
    result << x, y;
    return result;
}

/**
 * @brief Internal recursive implementation of the split Newton solver.
 * 
 * This function decomposes the problem into smaller subsystems based on the provided 
 * split locations and solves them iteratively using a block Gauss-Seidel approach.
 * 
 * @param df Gradient function.
 * @param J Jacobian function.
 * @param x0 Initial guess (already in split layout).
 * @param locs Split locations in terms of global vector indices.
 * @param maxiter Maximum number of iterations.
 * @param npts Number of grid points.
 * @param sparse Whether to use sparse solvers.
 * @param dt0 Initial pseudo-timestep.
 * @param dtmax Maximum pseudo-timestep.
 * @param bounds Variable bounds (already in split layout).
 * @param jacobian_age Number of iterations between Jacobian updates.
 * @param abs Absolute tolerance.
 * @param rel Relative tolerance.
 * @return A tuple containing the final state, step taken, number of iterations, and status.
 */
inline std::tuple<Vector, Vector, int, int> split_newton_recursive(
    Gradient df, Jacobian J, const Vector& x0, const std::vector<int>& locs, int maxiter = std::numeric_limits<int>::max(), int npts = 1,
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0,
    const Bounds& bounds = std::nullopt, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
{
    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax");
    }

    // Base case: If no more splits, solve directly with Newton
    if (locs.empty())
    {
        return newton(df, J, x0, maxiter, npts, sparse, dt0, dtmax, bounds, jacobian_age, abs, rel);
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
    auto dfa = [&](const Vector& xa_local)
    {
        return df(attach(xa_local, xb)).segment(0, loc).eval();
    };

    auto Ja = [&](const Vector& xa_local)
    {
        Matrix Ja_matrix = J(attach(xa_local, xb)).block(0, 0, loc, loc);
        return Ja_matrix;
    };

    // Define residual and Jacobian for xb (keeping xa fixed)
    auto dfb = [&](const Vector& xb_local)
    {
        return df(attach(xa, xb_local)).segment(loc, x0.size() - loc).eval();
    };

    auto Jb = [&](const Vector& xb_local)
    {
        Matrix Jb_matrix = J(attach(xa, xb_local)).block(loc, loc, x0.size() - loc, x0.size() - loc);
        return Jb_matrix;
    };

    // Adjust locs for recursion (relative to xb)
    std::vector<int> new_locs(locs.begin() + 1, locs.end());
    for (int& l : new_locs)
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
        auto [new_xb, sb, iter_b, status_b] = split_newton_recursive(dfb, Jb, xb, new_locs, maxiter, npts, sparse, dt0, dtmax, bounds_b, jacobian_age, abs, rel);
        xb = new_xb;

        // One Newton step for left subsystem
        auto [new_xa, sa, iter_a, status_a] = newton(dfa, Ja, xa, 1, npts, sparse, dt0, dtmax, bounds_a, jacobian_age, abs, rel);
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
        crit = norm2(x, s, npts, abs, rel);

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

/**
 * @brief Top-level split Newton solver that handles shuffling and problem decomposition.
 * 
 * This function accepts operators in the natural interleaved layout, performs necessary 
 * shuffling to a blocked (split) layout, and calls the recursive solver to handle 
 * the block-based solution process.
 * 
 * @param df_interleaved Gradient function in interleaved layout.
 * @param J_interleaved Jacobian function in interleaved layout.
 * @param x0_interleaved Initial guess in interleaved layout.
 * @param split_locs_components Component indices where the system should be split.
 * @param maxiter Maximum number of iterations.
 * @param npts Number of grid points.
 * @param sparse Whether to use sparse solvers.
 * @param dt0 Initial pseudo-timestep.
 * @param dtmax Maximum pseudo-timestep.
 * @param bounds_interleaved Variable bounds in interleaved layout.
 * @param jacobian_age Number of iterations between Jacobian updates.
 * @param abs Absolute tolerance.
 * @param rel Relative tolerance.
 * @return A tuple containing the final state, step taken, number of iterations, and status.
 */
inline std::tuple<Vector, Vector, int, int> split_newton(
    Gradient df_interleaved, Jacobian J_interleaved, const Vector& x0_interleaved,
    const std::vector<int>& split_locs_components, int maxiter = std::numeric_limits<int>::max(), int npts = 1,
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0,
    const Bounds& bounds_interleaved = std::nullopt, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
{
    if (split_locs_components.empty())
    {
        return newton(df_interleaved, J_interleaved, x0_interleaved, maxiter, npts, sparse, dt0, dtmax, bounds_interleaved, jacobian_age, abs, rel);
    }

    // 1. Setup Shuffler
    int nv = x0_interleaved.size() / npts;
    splitnewton::Shuffler sh(npts, nv, split_locs_components);

    // 2. Shuffle initial guess
    Vector x0 = sh.shuffle(x0_interleaved);

    // 3. Wrap operators to handle shuffling internally
    Gradient df = [&](const Vector& u_split) -> Vector
    {
        return sh.shuffle(df_interleaved(sh.unshuffle(u_split)));
    };

    Jacobian J = [&](const Vector& u_split) -> Matrix
    {
        // Matrix shuffling is expensive, but necessary for block iteration if starting from interleaved
        // Note: Simulation layer now builds interleaved Jacobians faster.
        return Matrix(sh.shuffle_matrix(J_interleaved(sh.unshuffle(u_split)).sparseView()));
    };

    // 4. Shuffle bounds
    Bounds bounds = std::nullopt;
    if (bounds_interleaved)
    {
        bounds = std::make_pair(sh.shuffle(bounds_interleaved->first), sh.shuffle(bounds_interleaved->second));
    }

    // 5. Convert component split locations to global vector indices for the blocked solver
    std::vector<int> locs;
    std::vector<int> sorted = split_locs_components;
    std::sort(sorted.begin(), sorted.end());
    for (int l : sorted)
    {
        locs.push_back(l * npts);
    }

    // 6. Call appropriate blocked solver
    Vector xf_split, step_split;
    int iter, status;

    if (splitnewton::Options::getInstance().hasFlag("-use_jacobi"))
    {
        spdlog::info("Using Block-Jacobi Newton solver");
        std::tie(xf_split, step_split, iter, status) = jacobi_block_newton(
            df, J, x0, locs, maxiter, npts, sparse, dt0, dtmax, bounds, jacobian_age, abs, rel);
    }
    else if (splitnewton::Options::getInstance().hasFlag("-use_gauss_seidel"))
    {
        spdlog::info("Using Block-Gauss-Seidel Newton solver");
        std::tie(xf_split, step_split, iter, status) = gauss_seidel_block_newton(
            df, J, x0, locs, maxiter, npts, sparse, dt0, dtmax, bounds, jacobian_age, abs, rel);
    }
    else
    {
        spdlog::info("Using recursive Split-Newton solver");
        std::tie(xf_split, step_split, iter, status) = split_newton_recursive(
            df, J, x0, locs, maxiter, npts, sparse, dt0, dtmax, bounds, jacobian_age, abs, rel);
    }

    // 7. Unshuffle result back to interleaved
    return {sh.unshuffle(xf_split), sh.unshuffle(step_split), iter, status};
}

#endif  // SPLIT_NEWTON_HPP