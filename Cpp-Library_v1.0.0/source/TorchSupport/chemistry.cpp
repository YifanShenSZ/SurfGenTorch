// Chemistry for libtorch

#include <torch/torch.h>

namespace CL { namespace TS { namespace chemistry {
    bool check_degeneracy(const double & threshold, const at::Tensor & energy) {
        bool deg = false;
        for (size_t i = 0; i < energy.numel() - 1; i++) {
            if (energy[i+1].item<double>() - energy[i].item<double>() < threshold) {
                deg = true;
                break;
            }
        }
        return deg;
    }

    // Matrix off-diagonal elements do not have determinate phase, because
    // the eigenvectors defining a representation have indeterminate phase difference

    // This module will fix the phase of NStates order symmetric matrix
    size_t NStates;
    // There are 2^(NStates-1) possibilities in total,
    // where the user input matrix indicates the base case and is excluded from trying,
    // so possible_phases.size() = 2^(NStates-1) - 1
    // possible_phases[i] contains one of the phases of NStates states
    // where true means -, false means +,
    // the phase of the last state is always arbitrarily assigned to +,
    // so possible_phases[i].size() == NStates-1
    std::vector<std::vector<bool>> possible_phases;

    void initialize_phase_fixing(const size_t & NStates_) {
        // NStates
        NStates = NStates_;
        // possible_phases
        possible_phases.clear();
        // Unchanged case is exculded
        possible_phases.resize(1 << (NStates-1) - 1);
        for (auto & phase : possible_phases) phase.resize(NStates-1);
        possible_phases[0][0] = true;
        for (size_t i = 1; i < NStates-1; i++)
        possible_phases[0][i] = false;
        for (size_t i = 1; i < possible_phases.size(); i++) {
            for (size_t j = 0; j < NStates-1; j++)
            possible_phases[i][j] = possible_phases[i-1][j];
            size_t count = 0;
            while(possible_phases[i][count]) {
                possible_phases[i][count] = false;
                count++;
            }
            possible_phases[i][count] = true;
        }
    }

    // Fix M by minimizing || M - ref ||_F^2
    void fix(at::Tensor & M, const at::Tensor & ref) {
        double change_min = 0.0;
        int     phase_min = -1;
        if (M.sizes().size() > 2) {
            std::vector<int64_t> dim_vec(M.sizes().size()-2);
            for (size_t i = 0; i < dim_vec.size(); i++) dim_vec[i] = i+2;
            c10::IntArrayRef sum_dim(dim_vec.data(), dim_vec.size());
            at::Tensor diff = (M - ref).pow_(2).sum(sum_dim);
            if (sum_dim.size() < 3) sum_dim = {};
            else {
                dim_vec.resize(M.sizes().size()-4);
                for (size_t i = 0; i < dim_vec.size(); i++) dim_vec[i] = i;
                sum_dim = c10::IntArrayRef(dim_vec.data(), dim_vec.size());
            }
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += (-M[i][j] - ref[i][j]).pow_(2).sum(sum_dim) - diff[i][j];
                    if (possible_phases[phase][i])
                    change += (-M[i][NStates-1] - ref[i][NStates-1]).pow_(2).sum(sum_dim) - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        else {
            at::Tensor diff = (M - ref).pow_(2);
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += (-M[i][j] - ref[i][j]).pow_(2) - diff[i][j];
                    if (possible_phases[phase][i])
                    change += (-M[i][NStates-1] - ref[i][NStates-1]).pow_(2) - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        // Modify M if the best phase is different from the input
        if (phase_min > -1) {
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possible_phases[phase_min][i] != possible_phases[phase_min][j])
                M[i][j] = -M[i][j];
                if (possible_phases[phase_min][i])
                M[i][NStates-1] = -M[i][NStates-1];
            }
        }
    }
    // Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    void fix(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) {
        double change_min = 0.0;
        int     phase_min = -1;
        if (M1.sizes().size() > 2 && M2.sizes().size() > 2) {
            std::vector<int64_t> dim_vec1(M1.sizes().size()-2);
            for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i+2;
            c10::IntArrayRef sum_dim1(dim_vec1.data(), dim_vec1.size());
            std::vector<int64_t> dim_vec2(M2.sizes().size()-2);
            for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i+2;
            c10::IntArrayRef sum_dim2(dim_vec2.data(), dim_vec2.size());
            at::Tensor diff = weight * (M1 - ref1).pow_(2).sum(sum_dim1) + (M2 - ref2).pow_(2).sum(sum_dim2);
            if (sum_dim1.size() < 3) sum_dim1 = {};
            else {
                dim_vec1.resize(M1.sizes().size()-4);
                for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i;
                sum_dim1 = c10::IntArrayRef(dim_vec1.data(), dim_vec1.size());
            }
            if (sum_dim2.size() < 3) sum_dim2 = {};
            else {
                dim_vec2.resize(M2.sizes().size()-4);
                for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i;
                sum_dim2 = c10::IntArrayRef(dim_vec2.data(), dim_vec2.size());
            }
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M1.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum(sum_dim1)
                              + (-M2[i][j] - ref2[i][j]).pow_(2).sum(sum_dim2)
                              - diff[i][j];
                    if (possible_phases[phase][i])
                    change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2).sum(sum_dim1)
                              + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2).sum(sum_dim2)
                              - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        else if (M1.sizes().size() > 2) {
            std::vector<int64_t> dim_vec1(M1.sizes().size()-2);
            for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i+2;
            c10::IntArrayRef sum_dim1(dim_vec1.data(), dim_vec1.size());
            at::Tensor diff = weight * (M1 - ref1).pow_(2).sum(sum_dim1) + (M2 - ref2).pow_(2);
            if (sum_dim1.size() < 3) sum_dim1 = {};
            else {
                dim_vec1.resize(M1.sizes().size()-4);
                for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i;
                sum_dim1 = c10::IntArrayRef(dim_vec1.data(), dim_vec1.size());
            }
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M1.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum(sum_dim1)
                              + (-M2[i][j] - ref2[i][j]).pow_(2)
                              - diff[i][j];
                    if (possible_phases[phase][i])
                    change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2).sum(sum_dim1)
                              + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2)
                              - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        else if (M2.sizes().size() > 2) {
            std::vector<int64_t> dim_vec2(M2.sizes().size()-2);
            for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i+2;
            c10::IntArrayRef sum_dim2(dim_vec2.data(), dim_vec2.size());
            at::Tensor diff = weight * (M1 - ref1).pow_(2) + (M2 - ref2).pow_(2).sum(sum_dim2);
            if (sum_dim2.size() < 3) sum_dim2 = {};
            else {
                dim_vec2.resize(M2.sizes().size()-4);
                for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i;
                sum_dim2 = c10::IntArrayRef(dim_vec2.data(), dim_vec2.size());
            }
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M1.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += weight * (-M1[i][j] - ref1[i][j]).pow_(2)
                              + (-M2[i][j] - ref2[i][j]).pow_(2).sum(sum_dim2)
                              - diff[i][j];
                    if (possible_phases[phase][i])
                    change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2)
                              + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2).sum(sum_dim2)
                              - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        else {
            at::Tensor diff = weight * (M1 - ref1).pow_(2) + (M2 - ref2).pow_(2);
            // Try out phase possibilities
            for (int phase = 0; phase < possible_phases.size(); phase++) {
                at::Tensor change = M1.new_zeros({});
                for (size_t i = 0; i < NStates; i++) {
                    for (size_t j = i+1; j < NStates-1; j++)
                    if (possible_phases[phase][i] != possible_phases[phase][j])
                    change += weight * (-M1[i][j] - ref1[i][j]).pow_(2)
                              + (-M2[i][j] - ref2[i][j]).pow_(2)
                              - diff[i][j];
                    if (possible_phases[phase][i])
                    change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2)
                              + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2)
                              - diff[i][NStates-1];
                }
                if (change.item<double>() < change_min) {
                    change_min = change.item<double>();
                    phase_min  = phase;
                }
            }
        }
        // Modify M1 and M2 if the best phase is different from the input
        if (phase_min > -1) {
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possible_phases[phase_min][i] != possible_phases[phase_min][j]) {
                    M1[i][j] = -M1[i][j];
                    M2[i][j] = -M2[i][j];
                }
                if (possible_phases[phase_min][i]) {
                    M1[i][NStates-1] = -M1[i][NStates-1];
                    M2[i][NStates-1] = -M2[i][NStates-1];
                }
            }
        }
    }
} // namespace chemistry
} // namespace TS
} // namespace CL