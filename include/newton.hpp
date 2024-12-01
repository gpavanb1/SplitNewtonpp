#ifndef NEWTON_HPP
#define NEWTON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <optional>
#include "typedefs.h"

// Constants
constexpr double EPS = std::numeric_limits<double>::epsilon();

// Criterion Function
inline double criterion(const Vector &x, const Vector &s, double fn, double abs = 1e-5, double rel = 1e-6)
{
    // Criterion based on scaled step size and gradient norm
    double step_criterion = (s.array() / (x.array() * rel + abs)).matrix().norm();
    double gradient_criterion = fn / (x.norm() * rel + abs);
    return std::max(step_criterion, gradient_criterion);
}

// Check if within bounds
inline bool check_within_bounds(const Vector &x0, const Bounds &bounds = std::nullopt)
{
    if (!bounds)
    {
        return true; // No bounds to check
    }

    const auto &[lower, upper] = *bounds;

    if (lower.size() != x0.size() || upper.size() != x0.size())
    {
        throw std::invalid_argument("Bounds must match the size of the solution vector.");
    }

    for (int i = 0; i < x0.size(); ++i)
    {
        if (!(lower[i] <= x0[i] && x0[i] <= upper[i]))
        {
            return false;
        }
    }
    return true;
}

// Newton Method
inline std::tuple<Vector, Vector, int> newton(
    Gradient df, Jacobian J, Vector x0, int maxiter = std::numeric_limits<int>::max(),
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0, bool armijo = false,
    const Bounds &bounds = std::nullopt, double bound_fac = 0.8,
    bool suppress_gradient_check = false, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
{

    /**
     * @brief Applies the Newton method to solve for a root of a function.
     *
     * This function iterates the Newton method to find the root of a nonlinear equation
     * given the gradient and Jacobian. The method includes various options to control
     * the iteration process.
     *
     * @param df A function representing the gradient of the objective function.
     * @param J A function representing the Jacobian of the objective function.
     * @param x0 The initial guess for the root.
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
     * @param suppress_gradient_check Whether to suppress the gradient norm check -
     *        used in Split Newton to avoid checking as only 1 iteration is done in
     *        A-Cycle (default is `false`).
     * @param jacobian_age Number of iterations after which the Jacobian is updated
     *        (default is `5`).
     * @param abs Absolute tolerance for convergence (default is `1e-5`).
     * @param rel Relative tolerance for convergence (default is `1e-6`).
     *
     * @return A tuple containing:
     *         - The computed solution vector.
     *         - The computed gradient at the solution.
     *         - The number of iterations performed.
     */

    if (dt0 < 0 || dtmax < 0)
    {
        throw std::invalid_argument("Must specify positive dt0 and dtmax.");
    }

    double dt = dt0;

    // Check if initial point is within bounds
    if (!check_within_bounds(x0, bounds))
    {
        throw std::invalid_argument("Seed must be within the provided bounds.");
    }

    Vector x = x0;
    double f0 = df(x0).norm();
    Vector s = Eigen::VectorXd::Constant(x.size(), std::numeric_limits<double>::infinity());
    double crit = std::numeric_limits<double>::infinity();

    int iter = 0;
    int jacobian_iter = 0; // Counter to track Jacobian age
    Vector dfx;
    Matrix jac; // Store the Jacobian matrix
    double fn;

    // Define sparse solver outside the loop to reuse it
    Eigen::SparseLU<Eigen::SparseMatrix<double>> sparse_solver;

    while (crit >= 1 && iter < maxiter)
    {
        // Update Jacobian and sparse solver only as needed
        if (jacobian_iter == 0 || jacobian_iter >= jacobian_age)
        {
            spdlog::debug("Recomputing Jacobian at iteration {}", iter);
            jac = J(x);
            if (dt != 0)
            {
                jac += (1.0 / dt) * Eigen::MatrixXd::Identity(x.size(), x.size());
            }

            if (sparse)
            {
                Eigen::SparseMatrix<double> sp_jac = jac.sparseView();
                sparse_solver.compute(sp_jac);
                if (sparse_solver.info() != Eigen::Success)
                {
                    throw std::runtime_error("Sparse solver factorization failed");
                }
            }

            jacobian_iter = 0; // Reset counter
        }

        dfx = df(x);
        fn = dfx.norm();

        // Solve for step direction
        if (sparse)
        {
            s = sparse_solver.solve(dfx);
            if (sparse_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Sparse solver failed to solve");
            }
        }
        else
        {
            s = jac.colPivHouseholderQr().solve(dfx);
        }
        s = -s;

        // Apply Armijo rule
        if (armijo)
        {
            double alpha = 1e-4;
            double fac = 1.0;
            for (int i = 1; i <= 10; ++i)
            {
                fac = std::pow(2, -i);
                Vector new_step = df(x + fac * s);
                const double new_step_norm = new_step.norm();
                spdlog::debug("Armijo iteration: {} {} {}", fac, new_step_norm, fn);
                if (new_step_norm <= (1 - alpha * fac) * fn)
                {
                    break;
                }
            }
            s *= fac;
            spdlog::debug("Armijo scaling: {}", fac);
        }

        // Apply bounds
        if (bounds.has_value())
        {
            const auto &[lower, upper] = *bounds;
            double best_scaling = 1.0;

            for (int i = 0; i < x.size(); ++i)
            {
                if (s[i] != 0.0)
                {
                    double legal_delta = (s[i] < 0 ? lower[i] - x[i] : upper[i] - x[i]) * bound_fac;
                    double scaling = legal_delta / s[i];
                    best_scaling = std::min(best_scaling, scaling);
                }
            }
            s *= best_scaling;

            if (best_scaling != 1.0)
                spdlog::info("Bounds hit. Damping factor: {}", best_scaling);
        }

        // Check convergence
        crit = criterion(x, s, fn);
        spdlog::trace("Iteration " + std::to_string(iter) + ": Criterion = " + std::to_string(crit));
        if (dt != 0)
        {
            spdlog::info("Timestep: {}", dt);
        }

        // Update x
        x += s;
        ++iter;

        // Update Jacobian age counter
        ++jacobian_iter;

        // Update timestep
        dt = std::min(dt0 * f0 / (fn + EPS), dtmax);
    }

    // Warn if fn is large but converged
    double gradient_criterion = fn / (x.norm() * rel + abs);
    if (gradient_criterion > 1.0 && !suppress_gradient_check)
        spdlog::warn("Gradient value is large but converged");

    return {x, s, iter};
}

#endif // NEWTON_HPP
