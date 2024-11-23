#ifndef SPLITNEWTON_HPP
#define SPLITNEWTON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <iostream>

// Constants
constexpr double EPS = std::numeric_limits<double>::epsilon();

// Criterion Function
template <typename Vector>
double criterion(const Vector& x, const Vector& s, double abs = 1e-5, double rel = 1e-6) {
    return (s.array() / (x.array() * rel + abs)).matrix().norm();
}

// Check if within bounds
template <typename Vector>
bool check_within_bounds(const Vector& x0, const std::vector<Vector>& bounds) {
    if (bounds.empty()) {
        return true;
    }

    if (bounds.size() != 2) {
        throw std::invalid_argument("Bounds must be 2 lists: lower and upper bounds of each dimension.");
    }

    const Vector& lower = bounds[0];
    const Vector& upper = bounds[1];

    if (lower.size() != x0.size() || upper.size() != x0.size()) {
        throw std::invalid_argument("Bounds lists must be as long as the solution vector.");
    }

    for (int i = 0; i < x0.size(); ++i) {
        if (!(lower[i] <= x0[i] && x0[i] <= upper[i])) {
            return false;
        }
    }
    return true;
}

// Newton Method
template <typename GradientFunc, typename JacobianFunc, typename Vector>
std::tuple<Vector, Vector, int> newton(
    GradientFunc df, JacobianFunc J, Vector x0, int maxiter = std::numeric_limits<int>::max(),
    bool sparse = false, double dt0 = 0.0, double dtmax = 1.0, bool armijo = false,
    const std::vector<Vector>& bounds = {}, double bound_fac = 0.8) {
    
    // Validate timestep
    if (dt0 < 0 || dtmax < 0) {
        throw std::invalid_argument("Must specify positive dt0 and dtmax.");
    }

    double dt = dt0;

    // Check if initial point is within bounds
    if (!check_within_bounds(x0, bounds)) {
        throw std::invalid_argument("Seed must be within the provided bounds.");
    }

    Vector x = x0;
    double f0 = df(x0).norm();
    Vector s = Vector::Constant(x.size(), std::numeric_limits<double>::infinity());
    double crit = std::numeric_limits<double>::infinity();

    int iter = 0;
    while (crit >= 1 && iter < maxiter) {
        // Update Jacobian and gradient
        Eigen::MatrixXd jac = J(x);
        if (dt != 0) {
            jac += (1 / dt) * Eigen::MatrixXd::Identity(x.size(), x.size());
        }

        Vector dfx = df(x);
        double fn = dfx.norm();

        // Solve for step direction
        if (sparse) {
            Eigen::SparseMatrix<double> sp_jac = jac.sparseView();
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(sp_jac);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Sparse solver failed to decompose the matrix.");
            }
            s = solver.solve(dfx);
        } else {
            s = jac.colPivHouseholderQr().solve(dfx);
        }
        s = -s;

        // Apply Armijo rule
        if (armijo) {
            double alpha = 1e-4;
            double fac = 1.0;
            for (int i = 1; i <= 10; ++i) {
                fac = std::pow(2, -i);
                Vector new_step = df(x + fac * s);
                if (new_step.norm() <= (1 - alpha * fac) * fn) {
                    break;
                }
            }
            s *= fac;
        }

        // Apply bounds
        if (!bounds.empty()) {
            const Vector& lower = bounds[0];
            const Vector& upper = bounds[1];
            double best_scaling = 1.0;

            for (int i = 0; i < x.size(); ++i) {
                if (s[i] != 0.0) {
                    double legal_delta = (s[i] < 0 ? lower[i] - x[i] : upper[i] - x[i]) * bound_fac;
                    double scaling = legal_delta / s[i];
                    best_scaling = std::min(best_scaling, scaling);
                }
            }
            s *= best_scaling;
        }

        // Check convergence
        crit = criterion(x, s);
        if (dt != 0) {
            std::cout << "Timestep: " << dt << std::endl;
        }

        // Update x
        x += s;
        ++iter;

        // Update timestep
        dt = std::min(dt0 * f0 / (fn + EPS), dtmax);
    }

    return {x, s, iter};
}

#endif // SPLITNEWTON_HPP
