# SplitNewton++

![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14783839.svg)](https://doi.org/10.5281/zenodo.14783839)

Fast, Bounded, Split [Newton](https://en.wikipedia.org/wiki/Newton%27s_method) with [pseudo-transient continuation
](https://ctk.math.ncsu.edu/TALKS/Purdue.pdf) and [backtracking](https://en.wikipedia.org/wiki/Backtracking_line_search)

Check out its Python cousin - [SplitNewton](https://github.com/gpavanb1/SplitNewton)

## Where is this used?

Good for ill-conditioned problems where there are two different sets of systems

Particular applications include
* [Fast-Slow Reaction-Diffusion systems](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)
* [CFD](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) - Pressure-Velocity coupling

## What does 'split' mean?

The system is divided into multiple segments, and for ease of communication, let’s refer to the first segment of variables as "outer" and the remaining as "inner".

* Holding the outer variables fixed, Newton iteration is performed recursively for the inner variables, using the sub-Jacobian associated with them, until convergence is reached.

* One Newton step is then performed for the outer variables, while the inner variables are kept fixed, using the sub-Jacobian for the outer subsystem.

* This process is repeated, alternating between solving the inner and outer subsystems, until the convergence criterion for the entire system (similar to standard Newton) is met.

### Example:

Consider a system of 5 variables, with the split locations at indices [1, 4]. This results in the following segments:

  * `a1` (variables from 0 to 1)
  * `a2 a3 a4` (variables from 1 to 4)
  * `a5` (variable at index 4)

1. First, the innermost segment `a5` is solved recursively using Newton's method while holding the variables `a1` and `a2 a3 a4`) fixed. This step is repeated until the convergence criterion for `a5` is met.

2. Next, one Newton step is taken for the segment `a2 a3 a4`, with `a5` held fixed. This step is followed by solving `a5` again till convergence.

3. This alternating process repeats: solving for `a5` until convergence, then one step for `a2 a3 a4`, and so on, until all subsystems converge.

Finally, one Newton step is performed for `a1`, with the other segments fixed. This completes one cycle of the split Newton process.

## How to install and execute?

This is a header-only library and can be most easily used with VSCode. The `tasks.json` contains the build and run commands that can be invoked directly from the IDE.

There is an example in `main.cpp` which can be compiled and executed

## How much faster is this?

For the bounded test problem with N=5000 and two split locations (at 1/3rd and 2/3rd), the C++ version is faster (even though the sparse linear solver is in C for the Python version). The comparison is as follows

| Method    | Time       | Iterations    | Time/Iteration |
|-----------|------------|---------------| -------------- |
C++ |  ~15 seconds  | 16  | <1 seconds |
Python | ~37 seconds | 33  | ~1.2 seconds |

## How to test?
You can run tests with the `gtest` framework. There is a `Bazel: Test` task in `.vscode/tasks.json` which can be used to run the tests.

The coverage reports can be generated with `llvm-cov` or `gcov` depending on the OS being used

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.

## Citing

If you are using SplitNewton++ in any scientific work, please make sure to cite as follows
```
@software{pavan_b_govindaraju_2025_14783839,
  author       = {Pavan B Govindaraju},
  title        = {gpavanb1/SplitNewtonpp: v0.2.0},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.14783839},
  url          = {https://doi.org/10.5281/zenodo.14783839}
}
```
