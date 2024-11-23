#ifndef SPLIT_NEWTON_HPP
#define SPLIT_NEWTON_HPP

#include <vector>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <functional>
#include "newton.hpp"

namespace split_newton
{

    // Utility to attach two vectors
    template <typename Vector>
    Vector attach(const Vector &x, const Vector &y)
    {
        Vector result(x);
        result.insert(result.end(), y.begin(), y.end());
        return result;
    }

    // Split Newton function
    template <
        typename GradientFunc,
        typename JacobianFunc,
        typename Vector,
        typename Bounds = std::pair<Vector, Vector>>
    std::tuple<Vector, Vector, int>
    split_newton(
        const GradientFunc &df,
        const JacobianFunc &J,
        const Vector &x0,
        size_t loc,
        size_t maxiter = std::numeric_limits<size_t>::max(),
        bool sparse = false,
        double dt0 = 0.0,
        double dtmax = 1.0,
        bool armijo = false,
        const Bounds *bounds = nullptr,
        double bound_fac = 0.8)
    {
        // Validate inputs
        if (dt0 < 0 || dtmax < 0)
        {
            throw std::invalid_argument("Must specify positive dt0 and dtmax");
        }
        if (loc > x0.size())
        {
            throw std::invalid_argument("Incorrect split location");
        }

        // Initial conditions
        double dt = dt0;
        Vector xa(x0.begin(), x0.begin() + loc);
        Vector xb(x0.begin() + loc, x0.end());
        Vector x = x0;
        Vector s(x.size(), std::numeric_limits<double>::infinity());
        double crit = std::numeric_limits<double>::infinity();
        size_t iter = 0;

        while (crit >= 1.0 && iter < maxiter)
        {
            // B Cycle
            auto dfb = [&](const Vector &x_b) -> Vector
            {
                return Vector(df(attach(xa, x_b)).begin() + loc, df(attach(xa, x_b)).end());
            };
            auto Jb = [&](const Vector &x_b) -> std::vector<std::vector<double>>
            {
                auto Jb_matrix = J(attach(xa, x_b));
                std::vector<std::vector<double>> result(Jb_matrix.begin() + loc, Jb_matrix.end());
                for (auto &row : result)
                {
                    row = Vector(row.begin() + loc, row.end());
                }
                return result;
            };
            auto local_bounds_b = bounds ? std::make_pair(
                                               Vector(bounds->first.begin() + loc, bounds->first.end()),
                                               Vector(bounds->second.begin() + loc, bounds->second.end()))
                                         : std::nullopt;
            auto [xb_new, sb, local_iter_b] = newton(dfb, Jb, xb, maxiter, sparse, dt, dtmax, armijo, local_bounds_b ? &(*local_bounds_b) : nullptr, bound_fac);
            xb = xb_new;

            // A Cycle
            auto dfa = [&](const Vector &x_a) -> Vector
            {
                return Vector(df(attach(x_a, xb)).begin(), df(attach(x_a, xb)).begin() + loc);
            };
            auto Ja = [&](const Vector &x_a) -> std::vector<std::vector<double>>
            {
                auto Ja_matrix = J(attach(x_a, xb));
                std::vector<std::vector<double>> result(Ja_matrix.begin(), Ja_matrix.begin() + loc);
                for (auto &row : result)
                {
                    row = Vector(row.begin(), row.begin() + loc);
                }
                return result;
            };
            auto local_bounds_a = bounds ? std::make_pair(
                                               Vector(bounds->first.begin(), bounds->first.begin() + loc),
                                               Vector(bounds->second.begin(), bounds->second.begin() + loc))
                                         : std::nullopt;
            auto [xa_new, sa, local_iter_a] = newton(dfa, Ja, xa, 1, sparse, dt, dtmax, armijo, local_bounds_a ? &(*local_bounds_a) : nullptr, bound_fac);
            xa = xa_new;

            // Construct new x and step
            auto x_new = attach(xa, xb);
            for (size_t i = 0; i < s.size(); ++i)
            {
                s[i] = x_new[i] - x[i];
            }

            // Check convergence
            crit = criterion(x, s);
            x = x_new;
            ++iter;
        }

        return {x, s, static_cast<int>(iter)};
    }

} // namespace split_newton

#endif // SPLIT_NEWTON_HPP
