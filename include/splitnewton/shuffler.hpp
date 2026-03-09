#ifndef SHUFFLER_HPP
#define SHUFFLER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace splitnewton {

/**
 * @brief Helper class to handle shuffling between interleaved and split layouts.
 * 
 * Interleaved (Natural): [u0, v0, u1, v1, ...]
 * Split (Blocked): [u0, u1, ..., v0, v1, ...]
 */
class Shuffler
{
public:
    Shuffler() : num_points_(0), nv_(0) {}

    Shuffler(int num_points, int nv, const std::vector<int>& split_locs)
        : num_points_(num_points), nv_(nv), split_locs_(split_locs)
    {
        if (num_points == 0 || nv == 0) return;

        // Grouping: indices in nv
        std::vector<int> group_sizes;
        std::vector<int> group_offsets;
        
        if (split_locs.empty())
        {
            group_sizes.push_back(nv);
            group_offsets.push_back(0);
        }
        else
        {
            std::vector<int> sorted = split_locs;
            std::sort(sorted.begin(), sorted.end());
            
            group_sizes.push_back(sorted[0]);
            for (size_t k = 1; k < sorted.size(); ++k)
            {
                group_sizes.push_back(sorted[k] - sorted[k - 1]);
            }
            group_sizes.push_back(nv - sorted.back());

            group_offsets.push_back(0);
            for (size_t k = 1; k < group_sizes.size(); ++k)
            {
                group_offsets.push_back(group_offsets[k - 1] + group_sizes[k - 1]);
            }
        }

        // Mapping: interleaved -> split
        interleaved_to_split_.resize(num_points * nv);
        for (int i = 0; i < num_points; ++i)
        {
            for (int j = 0; j < nv; ++j)
            {
                int g = -1;
                for (size_t k = 0; k < group_sizes.size(); ++k)
                {
                    if (j >= group_offsets[k] && j < group_offsets[k] + group_sizes[k])
                    {
                        g = k;
                        break;
                    }
                }
                int group_start = group_offsets[g] * num_points;
                int group_var_size = group_sizes[g];
                int offset_within_group = j - group_offsets[g];
                interleaved_to_split_[i * nv + j] = group_start + i * group_var_size + offset_within_group;
            }
        }

        split_to_interleaved_.resize(num_points * nv);
        for (int i = 0; i < (int)interleaved_to_split_.size(); ++i)
        {
            split_to_interleaved_[interleaved_to_split_[i]] = i;
        }
    }

    /**
     * @brief Shuffles an interleaved vector into split layout.
     * @param interleaved The vector in interleaved layout [u0, v0, u1, v1, ...].
     * @return The vector in split layout [u0, u1, ..., v0, v1, ...].
     */
    Eigen::VectorXd shuffle(const Eigen::VectorXd& interleaved) const
    {
        if (split_locs_.empty() || interleaved_to_split_.empty()) return interleaved;
        Eigen::VectorXd split(interleaved.size());
        for (int i = 0; i < (int)interleaved.size(); ++i)
        {
            split[interleaved_to_split_[i]] = interleaved[i];
        }
        return split;
    }

    /**
     * @brief Unshuffles a split vector back into interleaved layout.
     * @param split The vector in split layout [u0, u1, ..., v0, v1, ...].
     * @return The vector in interleaved layout [u0, v0, u1, v1, ...].
     */
    Eigen::VectorXd unshuffle(const Eigen::VectorXd& split) const
    {
        if (split_locs_.empty() || split_to_interleaved_.empty()) return split;
        Eigen::VectorXd interleaved(split.size());
        for (int i = 0; i < (int)split.size(); ++i)
        {
            interleaved[split_to_interleaved_[i]] = split[i];
        }
        return interleaved;
    }

    /**
     * @brief Shuffles an interleaved sparse matrix into split layout.
     * @param interleaved_mat The sparse matrix in interleaved layout.
     * @return The sparse matrix in split layout.
     */
    Eigen::SparseMatrix<double> shuffle_matrix(const Eigen::SparseMatrix<double>& interleaved_mat) const
    {
        if (split_locs_.empty() || interleaved_to_split_.empty()) return interleaved_mat;
        
        int n = interleaved_mat.rows();
        Eigen::SparseMatrix<double> split_mat(n, n);
        
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplets;
        triplets.reserve(interleaved_mat.nonZeros());

        for (int k = 0; k < interleaved_mat.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(interleaved_mat, k); it; ++it)
            {
                triplets.emplace_back(interleaved_to_split_[it.row()], 
                                     interleaved_to_split_[it.col()], 
                                     it.value());
            }
        }
        
        split_mat.setFromTriplets(triplets.begin(), triplets.end());
        return split_mat;
    }

private:
    int num_points_;
    int nv_;
    std::vector<int> split_locs_;
    std::vector<int> interleaved_to_split_;
    std::vector<int> split_to_interleaved_;
};

} // namespace splitnewton

#endif // SHUFFLER_HPP
