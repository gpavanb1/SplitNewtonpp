# SplitNewton++

![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)

Fast, Bounded, Split [Newton](https://en.wikipedia.org/wiki/Newton%27s_method) with [pseudo-transient continuation
](https://ctk.math.ncsu.edu/TALKS/Purdue.pdf) and [backtracking](https://en.wikipedia.org/wiki/Backtracking_line_search)

Check out its Python cousin - [SplitNewton](https://github.com/gpavanb1/SplitNewton)

## Where is this used?

Good for ill-conditioned problems where there are two different sets of systems

Particular applications include
* [Fast-Slow Reaction-Diffusion systems](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)
* [CFD](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) - Pressure-Velocity coupling

## What does 'split' mean?

The system is divided into two and for ease of communication, let's refer to first set of variables as "outer" and the second as "inner".

* Holding the outer variables fixed, Newton iteration is performed till convergence using the sub-Jacobian

* One Newton step is performed for the outer variables with inner held fixed (using its sub-Jacobian)

* This process is repeated till convergence criterion is met for the full system (same as in Newton)

## How to install and execute?

This is a header-only library and can be most easily used with VSCode. The `tasks.json` contains the build and run commands that can be invoked directly from the IDE.

There is an example in `main.cpp` which can be compiled and executed

## How much faster is this?

For the bounded Rosenbrock problem with N=5000, the C++ version is almost 2.5x faster (even though the sparse linear solver is in C for the Python version). The comparison is as follows

| Method    | Time       | Iterations    | Time/Iteration |
|-----------|------------|---------------| -------------- |
C++ |  ~14 seconds  | 21  | ~0.6 seconds |
Python | ~23 seconds | 15  | ~1.5 seconds |

## How to test?
You can run tests with the `gtest` framework. There is a `CMake: Run Tests` task in `.vscode/tasks.json` which can be used to run the tests.

The coverage reports can be generated with `llvm-cov` or `gcov` depending on the OS being used

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.
