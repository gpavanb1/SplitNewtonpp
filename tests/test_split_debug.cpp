/**
 * @file test_split_debug.cpp
 * @brief Demonstrates that split_newton solves different linear systems depending on split_locs.
 *
 * When split_locs is EMPTY: Direct newton path - solves full NxN system (no shuffler).
 * When split_locs is SET: Shuffler + block solver - solves smaller block systems (e.g. loc x loc, (N-loc) x (N-loc)).
 *
 * The same Jacobian J is passed in both cases, but:
 * - Without split: newton(J) solves J*s = -df for full system
 * - With split: shuffler permutes to blocked layout, then split_newton_recursive solves
 *   block A (Ja*sa = -dfa) and block B (Jb*sb = -dfb) separately
 */

#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <spdlog/spdlog.h>
#include "examples/test_functions.hpp"
#include "splitnewton/shuffler.hpp"
#include "splitnewton/split_newton.hpp"

static void verifyShuffler()
{
    const int npts = 2;
    const int nv = 2;
    const int n = npts * nv;
    std::vector<int> split_locs = {1};

    splitnewton::Shuffler sh(npts, nv, split_locs);

    // Build interleaved Jacobian with distinct entries: J(i,j) = 10*(i+1) + (j+1)
    Eigen::MatrixXd J_interleaved(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            J_interleaved(i, j) = 10 * (i + 1) + (j + 1);

    std::cout << "\n--- SHUFFLER VERIFICATION ---\n";
    std::cout << "Interleaved J (before shuffle), layout [u0,v0,u1,v1]:\n";
    std::cout << std::fixed << std::setprecision(0) << J_interleaved << "\n";

    Eigen::SparseMatrix<double> J_sparse = J_interleaved.sparseView();
    Eigen::MatrixXd J_shuffled = Eigen::MatrixXd(sh.shuffle_matrix(J_sparse));

    std::cout << "Shuffled J (after shuffle), layout [u0,u1,v0,v1] (blocked):\n";
    std::cout << J_shuffled << "\n";

    // Expected: interleaved_to_split perm [0,2,1,3] for npts=2,nv=2,split_locs={1}
    // So shuffled[r,c] = interleaved[split_to_interleaved[r], split_to_interleaved[c]]
    // Block A (0:2,0:2) = u block = rows/cols 0,2 from interleaved -> positions 0,1 in split
    // Block B (2:4,2:4) = v block = rows/cols 1,3 from interleaved -> positions 2,3 in split
    // Verify block structure: top-left 2x2 should be J(0,0), J(0,2), J(2,0), J(2,2)
    const double tol = 1e-10;
    assert(std::abs(J_shuffled(0, 0) - J_interleaved(0, 0)) < tol && "shuffle preserves (0,0)");
    assert(std::abs(J_shuffled(0, 1) - J_interleaved(0, 2)) < tol && "shuffle: (0,1)<-(0,2)");
    assert(std::abs(J_shuffled(1, 0) - J_interleaved(2, 0)) < tol && "shuffle: (1,0)<-(2,0)");
    assert(std::abs(J_shuffled(1, 1) - J_interleaved(2, 2)) < tol && "shuffle: (1,1)<-(2,2)");
    assert(std::abs(J_shuffled(2, 2) - J_interleaved(1, 1)) < tol && "shuffle: (2,2)<-(1,1)");
    assert(std::abs(J_shuffled(2, 3) - J_interleaved(1, 3)) < tol && "shuffle: (2,3)<-(1,3)");
    assert(std::abs(J_shuffled(3, 2) - J_interleaved(3, 1)) < tol && "shuffle: (3,2)<-(3,1)");
    assert(std::abs(J_shuffled(3, 3) - J_interleaved(3, 3)) < tol && "shuffle preserves (3,3)");

    Eigen::VectorXd v_interleaved(n);
    v_interleaved << 1.0, 2.0, 3.0, 4.0;
    Eigen::VectorXd v_shuffled = sh.shuffle(v_interleaved);
    Eigen::VectorXd v_roundtrip = sh.unshuffle(v_shuffled);
    assert((v_interleaved - v_roundtrip).norm() < tol && "shuffle(unshuffle(v)) == v");

    std::cout << "Vector: interleaved [1,2,3,4] -> shuffled " << v_shuffled.transpose()
              << " -> unshuffled " << v_roundtrip.transpose() << "\n";

    // Assert block structure: split_newton_recursive extracts J.block(0,0,loc,loc) and J.block(loc,n,loc,n)
    // from the SHUFFLED matrix. Block A (0:2,0:2) = u-block, Block B (2:4,2:4) = v-block.
    const int loc = 2;  // npts * 1 (split at component 1)
    Eigen::MatrixXd blockA = J_shuffled.block(0, 0, loc, loc);
    Eigen::MatrixXd blockB = J_shuffled.block(loc, loc, n - loc, n - loc);
    assert(std::abs(blockA(0, 0) - 11) < tol && std::abs(blockA(1, 1) - 33) < tol && "Block A = u-block diagonal");
    assert(std::abs(blockB(0, 0) - 22) < tol && std::abs(blockB(1, 1) - 44) < tol && "Block B = v-block diagonal");
    std::cout << "Block A (solver uses this):\n" << blockA << "\n";
    std::cout << "Block B (solver uses this):\n" << blockB << "\n";
    std::cout << "All shuffler assertions passed. System is solved in shuffled (blocked) layout.\n---\n";
}

int main()
{
    spdlog::set_level(spdlog::level::debug);  // Show system sizes (100x100 vs 50x50)

    verifyShuffler();

    const int num_elements = 100;
    Vector x0 = Eigen::VectorXd::LinSpaced(num_elements, 21.2, 31.2);
    std::vector<int> split_loc = {num_elements / 2};

    auto [func, der, hess] = set_functions("TEST");

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASE 1: split_newton with EMPTY split_locs (no shuffler)\n";
    std::cout << "        -> Should solve ONE full " << num_elements << "x" << num_elements << " system\n";
    std::cout << std::string(70, '=') << "\n\n";

    auto [x_no_split, step1, iter1, status1] = split_newton(
        der, hess, x0, {}, std::numeric_limits<int>::max(), 1,
        true, 0.0, 0.1, std::nullopt, 1);

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASE 2: split_newton with split_locs = {" << split_loc[0] << "} (shuffler + blocks)\n";
    std::cout << "        -> Should solve BLOCK A: " << split_loc[0] << "x" << split_loc[0]
              << " and BLOCK B: " << (num_elements - split_loc[0]) << "x" << (num_elements - split_loc[0]) << "\n";
    std::cout << std::string(70, '=') << "\n\n";

    auto [x_with_split, step2, iter2, status2] = split_newton(
        der, hess, x0, split_loc, std::numeric_limits<int>::max(), 1,
        true, 0.0, 0.1, std::nullopt, 1);

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Without split: status=" << status1 << ", iterations=" << iter1 << "\n";
    std::cout << "With split:    status=" << status2 << ", iterations=" << iter2 << "\n";
    std::cout << "Max diff between solutions: " << (x_no_split - x_with_split).cwiseAbs().maxCoeff() << "\n";
    std::cout << "\nBoth paths use the SAME Jacobian (hess), but solve DIFFERENT linear systems.\n";
    std::cout << std::string(70, '=') << "\n";

    return (status1 == 1 && status2 == 1) ? 0 : 1;
}
