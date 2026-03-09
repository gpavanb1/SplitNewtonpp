#ifndef JACOBI_HPP
#define JACOBI_HPP

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <limits>
#include <spdlog/spdlog.h>
#include "../newton.hpp"
#include "../typedefs.h"

namespace splitnewton {
} // namespace splitnewton

/**
 * @brief Block Jacobi Newton solver.
 * 
 * This solver decomposes the problem into subsystems and updates all subsystems 
 * simultaneously in each iteration using the values from the previous iteration.
 */
inline std::tuple<Vector, Vector, int, int> jacobi_block_newton(
    Gradient df, Jacobian J, const Vector& x0, const std::vector<int>& locs, 
    int maxiter = std::numeric_limits<int>::max(), int npts = 1,
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0,
    const Bounds& bounds = std::nullopt, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
{
    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax");
    }

    int n = x0.size();
    std::vector<int> boundaries;
    boundaries.push_back(0);
    for (int l : locs) boundaries.push_back(l);
    boundaries.push_back(n);

    int num_blocks = boundaries.size() - 1;

    Vector x = x0;
    Vector s = Vector::Zero(n);
    int iter = 1;
    int status = 0;

    while (iter <= maxiter)
    {
        Vector x_new = x;
        Vector full_step = Vector::Zero(n);
        
        // Take one Newton step for each block independently
        for (int i = 0; i < num_blocks; ++i)
        {
            int start = boundaries[i];
            int size = boundaries[i+1] - start;

            auto dfa = [&](const Vector& xa_local)
            {
                Vector x_temp = x;
                x_temp.segment(start, size) = xa_local;
                return df(x_temp).segment(start, size).eval();
            };

            auto Ja = [&](const Vector& xa_local)
            {
                Vector x_temp = x;
                x_temp.segment(start, size) = xa_local;
                return J(x_temp).block(start, start, size, size).eval();
            };

            Vector xa = x.segment(start, size);
            
            Bounds bounds_a = bounds ? std::make_optional(std::make_pair(
                                           bounds->first.segment(start, size),
                                           bounds->second.segment(start, size)))
                                     : std::nullopt;

            auto [new_xa, sa, iter_a, status_a] = newton(dfa, Ja, xa, 1, npts, sparse, dt0, dtmax, bounds_a, jacobian_age, abs, rel);
            
            x_new.segment(start, size) = new_xa;
            full_step.segment(start, size) = sa;
        }

        s = x_new - x;
        double crit = norm2(x, s, npts, abs, rel);

        spdlog::trace("Jacobi-Newton: Iteration {}: Criterion = {}", iter, to_scientific(crit, 3));

        if (crit < 1.0)
        {
            status = 1;
            x = x_new;
            break;
        }

        x = x_new;
        iter++;
    }

    if (status == 0 && iter > maxiter)
    {
        spdlog::warn("Maximum Jacobi-Newton iterations reached");
        status = -1;
    }

    return {x, s, iter, status};
}

#endif // JACOBI_HPP
