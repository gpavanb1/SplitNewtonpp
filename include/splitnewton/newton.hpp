#ifndef NEWTON_HPP
#define NEWTON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <optional>
#include "helper.hpp"
#include "typedefs.h"

// Constants
constexpr double EPS = std::numeric_limits<double>::epsilon();

// Weighted norm
inline double norm2(const Vector &x, const Vector &s, int npts = 1, double abs = 1e-5, double rel = 1e-5)
{
    // This is assumed to be divisible here
    int nv = x.size() / npts;

    double sum = 0.0;
    for (int n = 0; n < nv; ++n)
    {
        double esum = 0.0;
        // Sum absolute values for component n over all points.
        for (int j = 0; j < npts; ++j)
            esum += std::abs(x(j * nv + n));
        double ewt = rel * esum / npts + abs;
        for (int j = 0; j < npts; ++j)
        {
            double f = s(j * nv + n) / ewt;
            sum += f * f;
        }
    }
    return std::sqrt(sum);
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

// Helper function: Solve the linear system using either a dense or sparse approach.
// The function returns a tuple with the Newton step (already negated) and the norm of the residual.
inline std::tuple<Vector, int> sparse_linear_solve(const Eigen::SparseMatrix<double> &jac, const Vector &dfx)
{
    double status = 0;
    Vector s(dfx.size());

    // Compute diagonal scaling (Jacobi preconditioning)
    Eigen::VectorXd scale(jac.rows());
    for (int i = 0; i < jac.rows(); ++i)
    {
        double diag = jac.coeff(i, i);
        scale(i) = (std::abs(diag) < 1e-12) ? 1.0 : 1.0 / diag;
    }
    // Scale the Jacobian and the residual vector
    Eigen::SparseMatrix<double> sp_jac_scaled = scale.asDiagonal() * jac;
    Eigen::VectorXd dfx_scaled = dfx.cwiseProduct(scale);
    // Solve the system using SparseLU
    Eigen::SparseLU<Eigen::SparseMatrix<double>> sparse_solver;
    sparse_solver.compute(sp_jac_scaled);
    if (sparse_solver.info() != Eigen::Success)
    {
        spdlog::error("SparseLU failed to factorize");
        status = -3;
        return {-s, status};
    }
    Eigen::VectorXd s_scaled = sparse_solver.solve(dfx_scaled);
    if (sparse_solver.info() != Eigen::Success)
    {
        spdlog::error("SparseLU failed to solve");
        status = -4;
        return {-s, status};
    }
    s = s_scaled;
    status = 1;
    return {-s, status};
}

inline std::tuple<Vector, int> dense_linear_solve(const Matrix &jac, const Vector &dfx)
{
    double status = 1;
    Vector s = jac.colPivHouseholderQr().solve(dfx);
    return {-s, status};
}

// Compute scaling coefficients given the vector, step and bounds
inline double compute_bounds_scaling(const Vector &x, const Vector &s, const Bounds &bounds)
{
    double best_scaling = 1.0;
    if (!bounds.has_value())
        return best_scaling;

    // Else, extract bounds and find limits
    const auto &[lower, upper] = *bounds;

    for (int i = 0; i < x.size(); ++i)
    {
        if (s(i) != 0.0)
        {
            double legal_delta = (s(i) < 0 ? lower[i] - x[i] : upper[i] - x[i]);
            double scaling = legal_delta / s(i);
            best_scaling = std::min(best_scaling, scaling);
        }
    }
    return best_scaling;
}

inline std::tuple<Vector, Vector, int> damp_step(const Matrix &jac, Gradient df, const Vector &x, const Vector &s, const Bounds &bounds, int npts = 1, bool sparse = true, double abs = 1e-5, double rel = 1e-5, int NDAMP = 7, double damp_fac = std::sqrt(2.0))
{
    Vector x1, step1;
    int status = 0;

    // Compute initial norms
    double s0, s1;
    s0 = norm2(x, s, npts, abs, rel);

    // Compute the scaling factor
    double f_bound = compute_bounds_scaling(x, s, bounds);
    if (f_bound != 1.0)
        spdlog::debug("Bounds hit. Damping factor: {}", to_scientific(f_bound, 3));
    if (f_bound < 1e-10)
    {
        spdlog::debug("Damping factor is too small. Ending Newton prematurely...");
        status = -2;
        return {x, s, status};
    }

    // Proceed to damped iteration
    double f_damp = 1.0;
    int m;
    for (m = 0; m < NDAMP; m++)
    {
        double ff = f_bound * f_damp;
        x1 = x + ff * s;

        // Solve damped step
        if (sparse)
        {
            Eigen::SparseMatrix<double> sp_jac = jac.sparseView();
            sp_jac.makeCompressed();
            std::tie(step1, status) = sparse_linear_solve(sp_jac, df(x1));
        }
        else
        {
            std::tie(step1, status) = dense_linear_solve(jac, df(x1));
        }
        // Exit if the linear solve failed
        if (status != 1)
            return {x1, step1, status};

        // Check convergence
        s1 = norm2(x1, step1, npts, abs, rel);

        spdlog::trace("F_bound: {}, F_damp: {}, log10(s0): {}, log10(s1): {}", to_scientific(f_bound, 3), to_scientific(f_damp, 3), to_scientific(std::log10(s0), 3), to_scientific(std::log10(s1), 3));

        // If the new undamped step is small or if the undamped step is reducing in size, converge!
        if (s1 < 1.0 || s1 < s0)
            break;

        // Update damping factor
        f_damp /= damp_fac;
    }

    // If a damping coefficient was found, return 1 if the solution after
    // stepping by the damped step would represent a converged solution, and
    // return 0 otherwise. If no damping coefficient could be found, return -2.
    if (m < NDAMP)
    {
        if (s1 > 1.0)
        {
            spdlog::debug("Damping factor found: {}, but solution not converged", to_scientific(f_damp, 3));
            status = 0;
        }
        else
        {
            spdlog::debug("Damping factor found: {}, solution converged", to_scientific(f_damp, 3));
            status = 1;
        }
    }
    else
    {
        spdlog::error("Damping factor not found");
        status = -2;
    }

    return {x1, step1, status};
}

// Newton Method
inline std::tuple<Vector, Vector, int, int> newton(
    Gradient df, Jacobian J, Vector x0, int maxiter = std::numeric_limits<int>::max(), int npts = 1,
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0,
    const Bounds &bounds = std::nullopt, int jacobian_age = 5, double abs = 1e-5, double rel = 1e-6)
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
     * @param npts Number of points in the vector (used for multi-variate problems - default is `1`).
     * @param sparse Whether to use sparse matrices for the Jacobian (default is `false`).
     * @param dt0 Initial step size (default is `0.0`).
     * @param dtmax Maximum allowable step size (default is `1.0`).
     * @param bounds Optional bounds for the solution, represented by a `Bounds`
     *        object - `std::pair` of `Vector` (default is `std::nullopt`).
     * @param jacobian_age Number of iterations after which the Jacobian is updated
     *        (default is `5`).
     * @param abs Absolute tolerance for convergence (default is `1e-5`).
     * @param rel Relative tolerance for convergence (default is `1e-6`).

     *
     * @return A tuple containing:
     *         - The computed solution vector.
     *         - The computed gradient at the solution.
     *         - The number of iterations performed.
     *         - The status of the iteration:
     *            - `1` if converged.
     *            - `-1` if maximum iterations reached.
     *            - `-2` if damping factor is too small.
     *            - `-3` if SparseLU failed to factorize.
     *            - `-4` if SparseLU failed to solve.
     *            - `0` if no status is set.
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

    // Current step and future step for damped Newton
    Vector step0, step1, step;

    Vector x1;

    double crit = std::numeric_limits<double>::infinity();
    int status = 0;
    int iter = 1;

    Vector x = x0;

    // Evaluate f0
    double f0 = df(x0).cwiseAbs().maxCoeff();

    Matrix jac; // Store the Jacobian matrix
    double fn;

    while (1)
    {
        // Update Jacobian and sparse solver only as needed
        if (jacobian_age == 1 || iter % jacobian_age == 1)
        {
            spdlog::debug("Recomputing Jacobian at iteration {}", iter);
            jac = J(x);
            if (dt != 0)
            {
                jac += (1.0 / dt) * Eigen::MatrixXd::Identity(x.size(), x.size());
            }
        }

        // Compute the initial step
        if (sparse)
        {
            Eigen::SparseMatrix<double> sp_jac = jac.sparseView();
            sp_jac.makeCompressed();
            std::tie(step0, status) = sparse_linear_solve(sp_jac, df(x));
        }
        else
            std::tie(step0, status) = dense_linear_solve(jac, df(x));
        if (status != 1)
            break;

        // Damped Newton step
        std::tie(x1, step1, status) = damp_step(jac, df, x, step0, bounds, npts, sparse);
        if (status < 0)
            break;
        // Continue if status is 0

        // Update x
        x = x1;
        step = step1;

        // Check convergence
        crit = norm2(x1, step1, npts, abs, rel);

        // Get steady state norm
        double fn = df(x1).cwiseAbs().maxCoeff();

        // Output trace
        spdlog::trace("Iteration: {}, Criterion: {}, log10(ss): {}", iter, to_scientific(crit, 3), to_scientific(std::log10(fn), 3));

        if (dt != 0)
        {
            spdlog::info("Timestep: {}", to_scientific(dt, 3));
        }

        if (crit < 1.0)
        {
            status = 1;
            spdlog::debug("Converged in {} iterations", iter);
            break;
        }

        // Respond based on status
        if (iter >= maxiter)
        {
            if (iter != 1)
                spdlog::warn("Maximum iterations reached");
            status = -1;
            break;
        }

        // Update timestep
        dt = std::min(dt0 * f0 / (fn + EPS), dtmax);

        // Update iteration counts
        ++iter;
    }

    return {x, step, iter, status};
}

#endif // NEWTON_HPP
