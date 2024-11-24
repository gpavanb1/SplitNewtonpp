#ifndef TYPDEFS_H
#define TYPDEFS_H

#include <Eigen/Dense>
#include <functional>
#include <optional>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

using Func = std::function<Vector(const Vector &)>;
using Gradient = std::function<Vector(const Vector &)>;
using Jacobian = std::function<Matrix(const Vector &)>;

using FunctionSet = std::tuple<Func, Gradient, Jacobian>;

using Bounds = std::optional<std::pair<Vector, Vector>>;

#endif