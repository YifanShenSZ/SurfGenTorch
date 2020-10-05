/*
Train Hd network
*/

#include <omp.h>
#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>
#include <FortranLibrary.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "Hd.hpp"
#include "AbInitio.hpp"

namespace train_Hd {

// The "unit" of energy, accounting for the unit difference bewteen energy and gradient
double unit, unit_square;
void set_unit(const std::vector<AbInitio::RegHam *> & RegSet) {
    double MaxEnergy = 0.0, MaxGrad = 0.0;
    for (auto & data : RegSet) {
        double temp = std::abs(data->energy[0].item<double>());
        MaxEnergy = temp > MaxEnergy ? temp : MaxEnergy;
        temp = data->dH[0][0].norm().item<double>();
        MaxGrad = temp > MaxGrad ? temp : MaxGrad;
    }
    unit = MaxGrad / MaxEnergy;
    unit_square = unit * unit;
    std::cout << "The typical work length of this system = " << 1.0 / unit << '\n';
}

// Compute adiabatic energy (Ha) and gradient (dHa) from
// observable net input layer (x) and its Jacobian^T w.r.t. Cartesian coordinate
inline std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(
std::vector<at::Tensor> & x, const std::vector<at::Tensor> & J_IL_r_T) {
    // Enable gradient w.r.t. x to compute dH
    for (auto & irred : x) irred.set_requires_grad(true);
    // Compute diabatic quantity
    at::Tensor  H = Hd::compute_Hd(x);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, J_IL_r_T[0].size(0)});
    for (int i = 0; i < Hd::NStates; i++) {
        torch::autograd::variable_list g = torch::autograd::grad({H[i][i]}, {x[0]}, {}, true, true);
        dH[i][i] = J_IL_r_T[0].mv(g[0]);
        for (int j = i + 1; j < Hd::NStates; j++) {
            auto & irred = Hd::symmetry[i][j];
            torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {x[0], x[irred]}, {}, true, true);
            dH[i][j] = J_IL_r_T[0].mv(g[0]) + J_IL_r_T[irred].mv(g[1]);
        }
    }
    // Disable gradient w.r.t. x to save CPU during loss.backward
    for (auto & irred : x) irred.set_requires_grad(false);
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true);
    dH = CL::TS::LA::UT_A3_U(dH, state);
    return std::make_tuple(energy, dH);
}

inline void slice(const int & NStates, at::Tensor & energy, at::Tensor & dH) {
    energy = energy.slice(0, 0, NStates);
    dH = dH.slice(0, 0, NStates);
    dH = dH.slice(1, 0, NStates);
}

inline at::Tensor dH_loss(const at::Tensor & dH, const at::Tensor & target) {
    at::Tensor loss = dH.new_zeros({});
    for (int i = 0; i < dH.size(0); i++)
    for (int j = i+1; j < dH.size(0); j++)
    loss += (dH[i][j] - target[i][j]).pow(2).sum();
    loss *= 2.0;
    for (int i = 0; i < dH.size(0); i++)
    loss += (dH[i][i] - target[i][i]).pow(2).sum();
    return loss;
}

inline at::Tensor H_dH_loss(const at::Tensor & H, const at::Tensor & dH,
const at::Tensor & H_target, const at::Tensor & dH_target) {
    at::Tensor loss = H.new_zeros({});
    for (int i = 0; i < H.size(0); i++)
    for (int j = i+1; j < H.size(0); j++)
    loss += unit_square * (H[i][j] - H_target[i][j]).pow(2)
          + (dH[i][j] - dH_target[i][j]).pow(2).sum();
    loss *= 2.0;
    for (int i = 0; i < H.size(0); i++)
    loss += unit_square * (H[i][i] - H_target[i][i]).pow(2)
          + (dH[i][i] - dH_target[i][i]).pow(2).sum();
    return loss;
}

// The procedure to compute loss is to:
// 1. compute Ha and dHa
// 2. slice to the number of states in data
// 3. if degenerate, transform to composite representation
// 4. fix the phase of off-diagonals
// 5. evaluate loss
at::Tensor loss_reg(AbInitio::RegHam * data) {
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T);
    slice(data->energy.size(0), H, dH);
    CL::TS::chemistry::fix(dH, data->dH);
    at::Tensor loss = unit_square * torch::mse_loss(H, data->energy, at::Reduction::Sum)
                    + dH_loss(dH, data->dH);
    return data->weight * loss;
}
at::Tensor loss_deg(AbInitio::DegHam * data) {
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T);
    slice(data->H.size(0), H, dH);
    CL::TS::chemistry::composite_representation(H, dH);
    CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
    at::Tensor loss = H_dH_loss(H, dH, data->H, data->dH);
    return loss;
}

std::tuple<double, double> RMSD_reg(const std::vector<AbInitio::RegHam *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    if (! DataSet.empty()) {
        for (auto & data : DataSet) {
            at::Tensor H, dH;
            std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T);
            slice(data->energy.size(0), H, dH);
            CL::TS::chemistry::fix(dH, data->dH);
            e_H += torch::mse_loss(H, data->energy, at::Reduction::Mean).item<double>();
            e_dH += dH_loss(dH, data->dH).item<double>() / dH.numel();
        }
        e_H /= DataSet.size(); e_dH /= DataSet.size();
        e_H  = std::sqrt(e_H); e_dH  = std::sqrt(e_dH);
    }
    return std::make_tuple(e_H, e_dH);
}
std::tuple<double, double> RMSD_deg(const std::vector<AbInitio::DegHam *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    if (! DataSet.empty()) {
        for (auto & data : DataSet) {
            at::Tensor H, dH;
            std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T);
            slice(data->H.size(0), H, dH);
            CL::TS::chemistry::composite_representation(H, dH);
            CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
            at::Tensor H_MSD = H.new_zeros({}), dH_MSD = dH.new_zeros({});
            for (int i = 0; i < H.size(0); i++)
            for (int j = i+1; j < H.size(0); j++) {
                 H_MSD += ( H[i][j] - data-> H[i][j]).pow(2);
                dH_MSD += (dH[i][j] - data->dH[i][j]).pow(2).sum();
            }
            H_MSD *= 2.0; dH_MSD *= 2.0;
            for (int i = 0; i < H.size(0); i++) {
                 H_MSD += ( H[i][i] - data-> H[i][i]).pow(2);
                dH_MSD += (dH[i][i] - data->dH[i][i]).pow(2).sum();
            }
            e_H  +=  H_MSD.item<double>() /  H.numel();
            e_dH += dH_MSD.item<double>() / dH.numel();
        }
        e_H /= DataSet.size(); e_dH /= DataSet.size();
        e_H  = std::sqrt(e_H); e_dH  = std::sqrt(e_dH);
    }
    return std::make_tuple(e_H, e_dH);
}

// When simultaneously train the dimensionality reduction network
namespace DimRed_trainer {
    // Compute adiabatic energy (Ha) and gradient (dHa) from
    // scaled and symmetry adapted internal coordinate (SSAq) and its Jacobian^T w.r.t. Cartesian coordinate
    inline std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(
    std::vector<at::Tensor> & SSAq, const std::vector<at::Tensor> & J_SSAq_r_T) {
        // Enable gradient w.r.t. SSAq to compute dH
        for (auto & irred : SSAq) irred.set_requires_grad(true);
        // Compute observable net input layer
        std::vector<at::Tensor> Redq = DimRed::reduce(SSAq);
        std::vector<at::Tensor> input_layer = ON::input_layer(Redq);
        // Compute diabatic quantity
        at::Tensor  H = Hd::compute_Hd(input_layer);
        at::Tensor dH = H.new_zeros({Hd::NStates, Hd::NStates, J_SSAq_r_T[0].size(0)});
        for (int i = 0; i < Hd::NStates; i++) 
        for (int j = i; j < Hd::NStates; j++) {
            torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, SSAq, {}, true, true);
            for (size_t irred = 0; irred < SSAq.size(); irred++) dH[i][j] += J_SSAq_r_T[irred].mv(g[irred]);
        }
        // Disable gradient w.r.t. SSAq to save CPU during loss.backward
        for (auto & irred : SSAq) irred.set_requires_grad(false);
        // Transform to adiabatic representation
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        dH = CL::TS::LA::UT_A3_U(dH, state);
        return std::make_tuple(energy, dH);
    }

    std::tuple<double, double> RMSD_reg(const std::vector<AbInitio::RegHam *> DataSet) {
        double e_H = 0.0, e_dH = 0.0;
        if (! DataSet.empty()) {
            for (auto & data : DataSet) {
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                e_H += torch::mse_loss(H, data->energy, at::Reduction::Mean).item<double>();
                e_dH += dH_loss(dH, data->dH).item<double>() / dH.numel();
            }
            e_H /= DataSet.size(); e_dH /= DataSet.size();
            e_H  = std::sqrt(e_H); e_dH  = std::sqrt(e_dH);
        }
        return std::make_tuple(e_H, e_dH);
    }
    std::tuple<double, double> RMSD_deg(const std::vector<AbInitio::DegHam *> DataSet) {
        double e_H = 0.0, e_dH = 0.0;
        if (! DataSet.empty()) {
            for (auto & data : DataSet) {
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                at::Tensor H_MSD = H.new_zeros({}), dH_MSD = dH.new_zeros({});
                for (int i = 0; i < H.size(0); i++)
                for (int j = i+1; j < H.size(0); j++) {
                     H_MSD += ( H[i][j] - data-> H[i][j]).pow(2);
                    dH_MSD += (dH[i][j] - data->dH[i][j]).pow(2).sum();
                }
                H_MSD *= 2.0; dH_MSD *= 2.0;
                for (int i = 0; i < H.size(0); i++) {
                     H_MSD += ( H[i][i] - data-> H[i][i]).pow(2);
                    dH_MSD += (dH[i][i] - data->dH[i][i]).pow(2).sum();
                }
                e_H  +=  H_MSD.item<double>() /  H.numel();
                e_dH += dH_MSD.item<double>() / dH.numel();
            }
            e_H /= DataSet.size(); e_dH /= DataSet.size();
            e_H  = std::sqrt(e_H); e_dH  = std::sqrt(e_dH);
        }
        return std::make_tuple(e_H, e_dH);
    }
} // namespace DimRed_trainer

// To make use of Fortran-Library nonlinear optimizers:
//     1. Map the network parameters to vector c
//     2. Compute residue and Jacobian
namespace FLopt {
    const double & Sqrt2 = 1.4142135623730951;

    int OMP_NUM_THREADS;

    // Hd element networks (each thread owns a copy)
    // The 0th thread shares Hd::nets
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<ON::Net>>>>> netss;

    // Data set
    std::vector<AbInitio::RegHam *> RegSet;
    std::vector<AbInitio::DegHam *> DegSet;
    // Divide data set for parallelism
    std::vector<size_t> RegChunk, DegChunk, start;

    // Compute adiabatic energy (Ha) and gradient (dHa) from input layer, J^T, specific network
    inline std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(
    std::vector<at::Tensor> & x, const std::vector<at::Tensor> & J_IL_r_T,
    const int & thread) {
        auto & nets = netss[thread];
        // Enable gradient w.r.t. input layer to compute dH
        for (auto & irred : x) irred.set_requires_grad(true);
        // Compute diabatic quantity
        at::Tensor H = x[0].new_empty({Hd::NStates, Hd::NStates});
        for (int i = 0; i < Hd::NStates; i++) {
            H[i][i] = nets[i][i][0]->forward(x[0]);
            for (int j = i + 1; j < Hd::NStates; j++) {
                auto & irred = x[Hd::symmetry[i][j]];
                at::Tensor net_outputs = irred.new_empty(irred.sizes());
                for (int k = 0; k < net_outputs.size(0); k++) net_outputs[k] = nets[i][j][k]->forward(x[0]);
                H[i][j] = net_outputs.dot(irred);
            }
        }
        at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, J_IL_r_T[0].size(0)});
        for (int i = 0; i < Hd::NStates; i++) {
            torch::autograd::variable_list g = torch::autograd::grad({H[i][i]}, {x[0]}, {}, true, true);
            dH[i][i] = J_IL_r_T[0].mv(g[0]);
            for (int j = i + 1; j < Hd::NStates; j++) {
                auto & irred = Hd::symmetry[i][j];
                torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {x[0], x[irred]}, {}, true, true);
                dH[i][j] = J_IL_r_T[0].mv(g[0]) + J_IL_r_T[irred].mv(g[1]);
            }
        }
        // Disable gradient w.r.t. input layer to save CPU during loss.backward
        for (auto & irred : x) irred.set_requires_grad(false);
        // Transform to adiabatic representation
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        dH = CL::TS::LA::UT_A3_U(dH, state);
        return std::make_tuple(energy, dH);
    }

    // Push network parameters to parameter vector c
    inline void p2c(const int & thread, double * c) {
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[thread][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(c[count]), p.data_ptr<double>(), p.numel() * sizeof(double));
            count += p.numel();
        }
    }
    // The other way round
    inline void c2p(const double * c, const int & thread) {
        torch::NoGradGuard no_grad;
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[thread][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(p.data_ptr<double>(), &(c[count]), p.numel() * sizeof(double));
            count += p.numel();
        }
    }

    inline void net_zero_grad(const int & thread) {
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[thread][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad() && p.grad().defined()) {
            p.grad().detach_();
            p.grad().zero_();
        };
    }

    void loss(double & l, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                loss[thread] += data->weight * (
                                unit_square * torch::mse_loss(H, data->energy, at::Reduction::Sum)
                                + dH_loss(dH, data->dH) );
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                loss[thread] += H_dH_loss(H, dH, data->H, data->dH);
            }
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
    }
    void grad(double * g, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                loss[thread] += data->weight * (
                                unit_square * torch::mse_loss(H, data->energy, at::Reduction::Sum)
                                + dH_loss(dH, data->dH) );
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                loss[thread] += H_dH_loss(H, dH, data->H, data->dH);
            }
            net_zero_grad(thread);
            loss[thread].backward();
        }
        // Push network gradients to g
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[0][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    g[count] += pg[i];
                    count++;
                }
            }
        }
    }
    int loss_grad(double & l, double * g, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                loss[thread] += data->weight * (
                                unit_square * torch::mse_loss(H, data->energy, at::Reduction::Sum)
                                + dH_loss(dH, data->dH) );
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                loss[thread] += H_dH_loss(H, dH, data->H, data->dH);
            }
            net_zero_grad(thread);
            loss[thread].backward();
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
        // Push network gradients to g
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[0][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    g[count] += pg[i];
                    count++;
                }
            }
        }
        return 0;
    }

    void residue(double * r, const double * c, const int & NEq, const int & Nc) {
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            size_t count = start[thread];
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                // energy residue
                at::Tensor r_E = data->weight * unit * (H - data->energy);
                std::memcpy(&(r[count]), r_E.data_ptr<double>(), r_E.numel() * sizeof(double));
                count += r_E.numel();
                // dH residue
                for (int i = 0; i < H.size(0); i++) {
                    at::Tensor r_dH = data->weight * (dH[i][i] - data->dH[i][i]);
                    std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                    count += r_dH.numel();
                    for (int j = i+1; j < H.size(0); j++) {
                        at::Tensor r_dH = data->weight * Sqrt2 * (dH[i][j] - data->dH[i][j]);
                        std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                        count += r_dH.numel();
                    }
                }
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                // H and dH residue
                for (int i = 0; i < H.size(0); i++) {
                    r[count] = unit * (H[i][i] - data->H[i][i]).item<double>();
                    count++;
                    at::Tensor r_dH = dH[i][i] - data->dH[i][i];
                    std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                    count += r_dH.numel();
                    for (int j = i+1; j < H.size(0); j++) {
                        r[count] = Sqrt2 * unit * (H[i][j] - data->H[i][j]).item<double>();
                        count++;
                        at::Tensor r_dH = Sqrt2 * (dH[i][j] - data->dH[i][j]);
                        std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                        count += r_dH.numel();
                    }
                }
            }
        }
    }
    // Push the network parameter gradient to Jacobian^T 
    inline void dp2JT(const int & thread, double * JT, const int & NEq, const size_t & column) {
        size_t row = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & net : netss[thread][i][j])
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            double * pg = p.grad().data_ptr<double>();
            for (size_t i = 0; i < p.grad().numel(); i++) {
                JT[row * NEq + column] = pg[i];
                row++;
            }
        }
    }
    void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            size_t column = start[thread];
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->energy.size(0), H, dH);
                CL::TS::chemistry::fix(dH, data->dH);
                // energy Jacobian
                at::Tensor r_E = data->weight * unit * H;
                for (size_t el = 0; el < r_E.numel(); el++) {
                    net_zero_grad(thread);
                    r_E[el].backward({}, true);
                    dp2JT(thread, JT, NEq, column);
                    column++;
                }
                // dH Jacobian
                for (int i = 0; i < H.size(0); i++) {
                    at::Tensor r_dH = data->weight * dH[i][i];
                    for (size_t el = 0; el < r_dH.numel(); el++) {
                        net_zero_grad(thread);
                        r_dH[el].backward({}, true);
                        dp2JT(thread, JT, NEq, column);
                        column++;
                    }
                    for (int j = i+1; j < H.size(0); j++) {
                        at::Tensor r_dH = data->weight * Sqrt2 * dH[i][j];
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(thread);
                            r_dH[el].backward({}, true);
                            dp2JT(thread, JT, NEq, column);
                            column++;
                        }
                    }
                }
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                at::Tensor H, dH;
                std::tie(H, dH) = compute_Ha_dHa(data->input_layer, data->J_IL_r_T, thread);
                slice(data->H.size(0), H, dH);
                CL::TS::chemistry::composite_representation(H, dH);
                CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                // H and dH Jacobian
                for (int i = 0; i < H.size(0); i++) {
                    at::Tensor r_H = H[i][i];
                    net_zero_grad(thread);
                    r_H.backward({}, true);
                    dp2JT(thread, JT, NEq, column);
                    column++;
                    at::Tensor r_dH = dH[i][i];
                    for (size_t el = 0; el < r_dH.numel(); el++) {
                        net_zero_grad(thread);
                        r_dH[el].backward({}, true);
                        dp2JT(thread, JT, NEq, column);
                        column++;
                    }
                    for (int j = i+1; j < H.size(0); j++) {
                        at::Tensor r_H = Sqrt2 * H[i][j];
                        net_zero_grad(thread);
                        r_H.backward({}, true);
                        dp2JT(thread, JT, NEq, column);
                        column++;
                        at::Tensor r_dH = Sqrt2 * dH[i][j];
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(thread);
                            r_dH[el].backward({}, true);
                            dp2JT(thread, JT, NEq, column);
                            column++;
                        }
                    }
                }
            }
        }
    }

    // When simultaneously train the dimensionality reduction network
    namespace DimRed_trainer {
        // The dimensionality reduction network (each thread owns a copy)
        // The 0th thread shares DimRed::nets
        std::vector<std::vector<std::shared_ptr<DimRed::Net>>> DimRedNetss;

        // Compute adiabatic energy (Ha) and gradient (dHa) from
        // scaled and symmetry adapted internal coordinate (SSAq) and its Jacobian^T w.r.t. Cartesian coordinate
        inline std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(
        std::vector<at::Tensor> & SSAq, const std::vector<at::Tensor> & J_SSAq_r_T,
        const int & thread) {
            auto & nets = netss[thread];
            // Enable gradient w.r.t. SSAq to compute dH
            for (auto & irred : SSAq) irred.set_requires_grad(true);
            // Compute observable net input layer
            std::vector<at::Tensor> Redq(SSAq.size());
            for (size_t i = 0; i < SSAq.size(); i++) Redq[i] = DimRedNetss[thread][i]->reduce(SSAq[i]);
            std::vector<at::Tensor> input_layer = ON::input_layer(Redq);
            // Compute diabatic quantity
            at::Tensor H = input_layer[0].new_empty({Hd::NStates, Hd::NStates});
            for (int i = 0; i < Hd::NStates; i++) {
                H[i][i] = nets[i][i][0]->forward(input_layer[0]);
                for (int j = i + 1; j < Hd::NStates; j++) {
                    auto & irred = input_layer[Hd::symmetry[i][j]];
                    at::Tensor net_outputs = irred.new_empty(irred.sizes());
                    for (int k = 0; k < net_outputs.size(0); k++) net_outputs[k] = nets[i][j][k]->forward(input_layer[0]);
                    H[i][j] = net_outputs.dot(irred);
                }
            }
            at::Tensor dH = H.new_zeros({Hd::NStates, Hd::NStates, J_SSAq_r_T[0].size(0)});
            for (int i = 0; i < Hd::NStates; i++) 
            for (int j = i; j < Hd::NStates; j++) {
                torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, SSAq, {}, true, true);
                for (size_t irred = 0; irred < SSAq.size(); irred++) dH[i][j] += J_SSAq_r_T[irred].mv(g[irred]);
            }
            // Disable gradient w.r.t. SSAq to save CPU during loss.backward
            for (auto & irred : SSAq) irred.set_requires_grad(false);
            // Transform to adiabatic representation
            at::Tensor energy, state;
            std::tie(energy, state) = H.symeig(true);
            dH = CL::TS::LA::UT_A3_U(dH, state);
            return std::make_tuple(energy, dH);
        }

        // Push network parameters to parameter vector c
        inline void p2c(const int & thread, double * c) {
            size_t count = 0;
            // Hd networks
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                std::memcpy(&(c[count]), p.data_ptr<double>(), p.numel() * sizeof(double));
                count += p.numel();
            }
            // Dimensionality reduction networks
            for (auto & net : DimRedNetss[thread])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                std::memcpy(&(c[count]), p.data_ptr<double>(), p.numel() * sizeof(double));
                count += p.numel();
            }
        }
        // The other way round
        inline void c2p(const double * c, const int & thread) {
            torch::NoGradGuard no_grad;
            size_t count = 0;
            // Hd networks
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                std::memcpy(p.data_ptr<double>(), &(c[count]), p.numel() * sizeof(double));
                count += p.numel();
            }
            // Dimensionality reduction networks
            for (auto & net : DimRedNetss[thread])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                std::memcpy(p.data_ptr<double>(), &(c[count]), p.numel() * sizeof(double));
                count += p.numel();
            }
        }

        inline void net_zero_grad(const int & thread) {
            // Hd networks
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad() && p.grad().defined()) {
                p.grad().detach_();
                p.grad().zero_();
            };
            // Dimensionality reduction networks
            for (auto & net : DimRedNetss[thread])
            for (auto & p : net->parameters())
            if (p.requires_grad() && p.grad().defined()) {
                p.grad().detach_();
                p.grad().zero_();
            };
        }

        void residue(double * r, const double * c, const int & NEq, const int & Nc) {
            #pragma omp parallel for
            for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
                c2p(c, thread);
                size_t count = start[thread];
                for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                    auto & data = RegSet[idata];
                    at::Tensor H, dH;
                    std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T, thread);
                    slice(data->energy.size(0), H, dH);
                    CL::TS::chemistry::fix(dH, data->dH);
                    // energy residue
                    at::Tensor r_E = data->weight * unit * (H - data->energy);
                    std::memcpy(&(r[count]), r_E.data_ptr<double>(), r_E.numel() * sizeof(double));
                    count += r_E.numel();
                    // dH residue
                    for (int i = 0; i < H.size(0); i++) {
                        at::Tensor r_dH = data->weight * (dH[i][i] - data->dH[i][i]);
                        std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                        count += r_dH.numel();
                        for (int j = i+1; j < H.size(0); j++) {
                            at::Tensor r_dH = data->weight * Sqrt2 * (dH[i][j] - data->dH[i][j]);
                            std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                            count += r_dH.numel();
                        }
                    }
                }
                for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                    auto & data = DegSet[idata];
                    at::Tensor H, dH;
                    std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T, thread);
                    slice(data->H.size(0), H, dH);
                    CL::TS::chemistry::composite_representation(H, dH);
                    CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                    // H and dH residue
                    for (int i = 0; i < H.size(0); i++) {
                        r[count] = unit * (H[i][i] - data->H[i][i]).item<double>();
                        count++;
                        at::Tensor r_dH = dH[i][i] - data->dH[i][i];
                        std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                        count += r_dH.numel();
                        for (int j = i+1; j < H.size(0); j++) {
                            r[count] = Sqrt2 * unit * (H[i][j] - data->H[i][j]).item<double>();
                            count++;
                            at::Tensor r_dH = Sqrt2 * (dH[i][j] - data->dH[i][j]);
                            std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                            count += r_dH.numel();
                        }
                    }
                }
            }
        }
        // Push the network parameter gradient to Jacobian^T 
        inline void dp2JT(const int & thread, double * JT, const int & NEq, const size_t & column) {
            size_t row = 0;
            // Hd networks
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[thread][i][j])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    JT[row * NEq + column] = pg[i];
                    row++;
                }
            }
            // Dimensionality reduction networks
            for (auto & net : DimRedNetss[thread])
            for (auto & p : net->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    JT[row * NEq + column] = pg[i];
                    row++;
                }
            }
        }
        void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
            #pragma omp parallel for
            for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
                c2p(c, thread);
                size_t column = start[thread];
                for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                    auto & data = RegSet[idata];
                    at::Tensor H, dH;
                    std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T, thread);
                    slice(data->energy.size(0), H, dH);
                    CL::TS::chemistry::fix(dH, data->dH);
                    // energy Jacobian
                    at::Tensor r_E = data->weight * unit * H;
                    for (size_t el = 0; el < r_E.numel(); el++) {
                        net_zero_grad(thread);
                        r_E[el].backward({}, true);
                        dp2JT(thread, JT, NEq, column);
                        column++;
                    }
                    // dH Jacobian
                    for (int i = 0; i < H.size(0); i++) {
                        at::Tensor r_dH = data->weight * dH[i][i];
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(thread);
                            r_dH[el].backward({}, true);
                            dp2JT(thread, JT, NEq, column);
                            column++;
                        }
                        for (int j = i+1; j < H.size(0); j++) {
                            at::Tensor r_dH = data->weight * Sqrt2 * dH[i][j];
                            for (size_t el = 0; el < r_dH.numel(); el++) {
                                net_zero_grad(thread);
                                r_dH[el].backward({}, true);
                                dp2JT(thread, JT, NEq, column);
                                column++;
                            }
                        }
                    }
                }
                for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                    auto & data = DegSet[idata];
                    at::Tensor H, dH;
                    std::tie(H, dH) = compute_Ha_dHa(data->SSAq, data->J_SSAq_r_T, thread);
                    slice(data->H.size(0), H, dH);
                    CL::TS::chemistry::composite_representation(H, dH);
                    CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
                    // H and dH Jacobian
                    for (int i = 0; i < H.size(0); i++) {
                        at::Tensor r_H = H[i][i];
                        net_zero_grad(thread);
                        r_H.backward({}, true);
                        dp2JT(thread, JT, NEq, column);
                        column++;
                        at::Tensor r_dH = dH[i][i];
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(thread);
                            r_dH[el].backward({}, true);
                            dp2JT(thread, JT, NEq, column);
                            column++;
                        }
                        for (int j = i + 1; j < H.size(0); j++) {
                            at::Tensor r_H = Sqrt2 * H[i][j];
                            net_zero_grad(thread);
                            r_H.backward({}, true);
                            dp2JT(thread, JT, NEq, column);
                            column++;
                            at::Tensor r_dH = Sqrt2 * dH[i][j];
                            for (size_t el = 0; el < r_dH.numel(); el++) {
                                net_zero_grad(thread);
                                r_dH[el].backward({}, true);
                                dp2JT(thread, JT, NEq, column);
                                column++;
                            }
                        }
                    }
                }
            }
        }
    } // namespace DimRed_trainer

    void initialize(const size_t & freeze_, const bool & train_DimRed_,
    const std::vector<AbInitio::RegHam *> & RegSet_,
    const std::vector<AbInitio::DegHam *> & DegSet_) {
        OMP_NUM_THREADS = omp_get_max_threads();
        std::cout << "The number of threads = " << OMP_NUM_THREADS << '\n';

        netss.resize(OMP_NUM_THREADS);
        netss[0] = Hd::nets;
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            auto & nets = netss[thread];
            nets.resize(Hd::NStates);
            for (int i = 0; i < Hd::NStates; i++) {
                nets[i].resize(Hd::NStates);
                for (int j = i; j < Hd::NStates; j++) {
                    nets[i][j].resize(Hd::nets[i][j].size());
                    for (int k = 0; k < nets[i][j].size(); k++) {
                        nets[i][j][k] = std::make_shared<ON::Net>(Hd::nets[i][j][k]);
                        nets[i][j][k]->to(torch::kFloat64);
                        nets[i][j][k]->copy(Hd::nets[i][j][k]);
                        nets[i][j][k]->freeze(freeze_);
                        nets[i][j][k]->train();
                    }
                }
            }
        }

        RegSet = RegSet_;
        DegSet = DegSet_;
        std::cout << "For parallelism, the number of  regular   data in use = "
                  << OMP_NUM_THREADS * (RegSet.size() / OMP_NUM_THREADS) << '\n';
        std::cout << "For parallelism, the number of degenerate data in use = "
                  << OMP_NUM_THREADS * (DegSet.size() / OMP_NUM_THREADS) << '\n';
        RegChunk.resize(OMP_NUM_THREADS);
        DegChunk.resize(OMP_NUM_THREADS);
        start.resize(OMP_NUM_THREADS);
        RegChunk[0] = RegSet.size() / OMP_NUM_THREADS;
        DegChunk[0] = DegSet.size() / OMP_NUM_THREADS;
        start[0] = 0;
        for (int i = 1; i < OMP_NUM_THREADS; i++) {
            RegChunk[i] = RegChunk[i-1] + RegChunk[0];
            DegChunk[i] = DegChunk[i-1] + DegChunk[0];
            start[i] = start[i-1];
            for (size_t j = RegChunk[i-1] - RegChunk[0]; j < RegChunk[i-1]; j++)
            start[i] += RegSet[j]->energy.size(0) + (RegSet[j]->dH.size(0)+1)*RegSet[j]->dH.size(0)/2 * RegSet[j]->dH.size(2);
            for (size_t j = DegChunk[i-1] - DegChunk[0]; j < DegChunk[i-1]; j++)
            start[i] += (DegSet[j]->H.size(0)+1)*DegSet[j]->H.size(0)/2 * (1 + DegSet[j]->dH.size(2));
        }

        if (train_DimRed_) {
            DimRed_trainer::DimRedNetss.resize(OMP_NUM_THREADS);
            DimRed_trainer::DimRedNetss[0] = DimRed::nets;
            for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
                auto & nets = DimRed_trainer::DimRedNetss[thread];
                nets.resize(DimRed::nets.size());
                for (size_t i = 0; i < nets.size(); i++) {
                    nets[i] = std::make_shared<DimRed::Net>(DimRed::nets[i]);
                    nets[i]->to(torch::kFloat64);
                    nets[i]->copy(DimRed::nets[i]);
                    nets[i]->freeze(freeze_);
                    nets[i]->freeze_inverse();
                    nets[i]->train();
                }
            }
        }
    }

    void optimize(const bool & train_DimRed, const std::string & opt, const size_t & epoch) {
        // Initialize
        if (train_DimRed) {
            int Nc = 0;
            // Hd networks
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[0][i][j])
            Nc += CL::TS::NParameters(net->parameters());
            // Dimensionality reduction networks
            for (auto & net : DimRed_trainer::DimRedNetss[0])
            Nc += CL::TS::NParameters(net->parameters());
            std::cout << "There are " << Nc << " parameters to train\n";
            double * c = new double[Nc];
            DimRed_trainer::p2c(0, c);
            int NEq = 0;
            for (size_t i = 0; i < OMP_NUM_THREADS * (RegSet.size() / OMP_NUM_THREADS); i++)
            NEq += RegSet[i]->energy.size(0)
                 + (RegSet[i]->dH.size(0) + 1) * RegSet[i]->dH.size(0) / 2 * RegSet[i]->dH.size(2);
            for (size_t i = 0; i < OMP_NUM_THREADS * (DegSet.size() / OMP_NUM_THREADS); i++)
            NEq += (DegSet[i]-> H.size(0) + 1) * DegSet[i]-> H.size(0) / 2
                 + (DegSet[i]->dH.size(0) + 1) * DegSet[i]->dH.size(0) / 2 * DegSet[i]->dH.size(2);
            std::cout << "The data set corresponds to " << NEq << " least square equations" << std::endl;
            // Train
            if (opt == "SD") {

            }
            else if (opt == "CG") {

            }
            else {
                for (auto & data : RegSet) data->weight = std::sqrt(data->weight);
                FL::NO::TrustRegion(DimRed_trainer::residue, DimRed_trainer::Jacobian, c, NEq, Nc, true, epoch);
                for (auto & data : RegSet) data->weight = data->weight * data->weight;
            }
            // Finish
            DimRed_trainer::c2p(c, 0);
            delete [] c;
        }
        else {
            int Nc = 0;
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & net : netss[0][i][j])
            Nc += CL::TS::NParameters(net->parameters());
            std::cout << "There are " << Nc << " parameters to train\n";
            double * c = new double[Nc];
            p2c(0, c);
            int NEq = 0;
            for (size_t i = 0; i < OMP_NUM_THREADS * (RegSet.size() / OMP_NUM_THREADS); i++)
            NEq += RegSet[i]->energy.size(0)
                 + (RegSet[i]->dH.size(0) + 1) * RegSet[i]->dH.size(0) / 2 * RegSet[i]->dH.size(2);
            for (size_t i = 0; i < OMP_NUM_THREADS * (DegSet.size() / OMP_NUM_THREADS); i++)
            NEq += (DegSet[i]-> H.size(0) + 1) * DegSet[i]-> H.size(0) / 2
                 + (DegSet[i]->dH.size(0) + 1) * DegSet[i]->dH.size(0) / 2 * DegSet[i]->dH.size(2);
            std::cout << "The data set corresponds to " << NEq << " least square equations" << std::endl;
            // Train
            if (opt == "SD") {
                FL::NO::SteepestDescent(loss, grad, loss_grad, c, Nc, false, true, epoch);
            }
            else if (opt == "CG") {
                FL::NO::ConjugateGradient(loss, grad, loss_grad, c, Nc, "DY", false, true, epoch);
            }
            else {
                for (auto & data : RegSet) data->weight = std::sqrt(data->weight);
                FL::NO::TrustRegion(residue, Jacobian, c, NEq, Nc, true, epoch);
                for (auto & data : RegSet) data->weight = data->weight * data->weight;
            }
            // Finish
            c2p(c, 0);
            delete [] c;
        }
    }
} // namespace FLopt

void train(const std::vector<double> & guess_diag, const size_t & freeze, const std::vector<std::string> & chk,
const std::vector<std::string> & data_set, const bool & train_DimRed,
const double & zero_point, const double & weight,
const std::string & opt, const size_t & epoch, const size_t & batch_size, const double & learning_rate) {
    std::cout << "Start training diabatic Hamiltonian\n";
    // Initialize network
    if (Hd::nets[0][0][0]->cold && (! guess_diag.empty())) {
        assert(("Wrong number of initial guesses for Hd diagonals", guess_diag.size() == Hd::NStates));
        for (int i = 0; i < Hd::NStates; i++) {
            torch::NoGradGuard no_grad;
            Hd::nets[i][i][0]->tail->bias.fill_(guess_diag[i]);
        }
    }
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        size_t count = 0;
        for (auto & net : Hd::nets[i][j]) {
            net->freeze(freeze);
            count += CL::TS::NParameters(net->parameters());
        }
        std::cout << "Number of trainable network parameters for Hd" << i+1 << j+1
                  << " = " << count << '\n';
    }
    if (train_DimRed)
    for (size_t i = 0; i < DimRed::nets.size(); i++) {
        auto & net = DimRed::nets[i];
        net->freeze_inverse();
        std::cout << "Number of trainable network parameters for dimensionality reduction of irreducible " << i + 1
                  << " = " << CL::TS::NParameters(net->parameters()) << '\n';
    }
    // Read data set
    AbInitio::DataSet<AbInitio::RegHam> * RegSet;
    AbInitio::DataSet<AbInitio::DegHam> * DegSet;
    std::tie(RegSet, DegSet) = AbInitio::read_HamSet(data_set, train_DimRed, zero_point, weight);
    std::cout << "Number of  regular   data = " << RegSet->size_int() << '\n';
    std::cout << "Number of degenerate data = " << DegSet->size_int() << '\n';
    // Initialize underlying modules
    CL::TS::chemistry::initialize_phase_fixing(Hd::NStates);
    set_unit(RegSet->example());
    std::cout << "The initial guess gives:\n";
    double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
    if (train_DimRed) {
        std::tie(regRMSD_H, regRMSD_dH) = DimRed_trainer::RMSD_reg(RegSet->example());
        std::tie(degRMSD_H, degRMSD_dH) = DimRed_trainer::RMSD_deg(DegSet->example());
    }
    else {
        std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
        std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
    }
    std::cout << "For  regular   data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
              << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
    std::cout << std::endl;
    if (opt == "Adam" || opt == "SGD") {
        // Not implemented
    }
    else {
        FLopt::initialize(freeze, train_DimRed, RegSet->example(), DegSet->example());
        FLopt::optimize(train_DimRed, opt, epoch);
        double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
        if (train_DimRed) {
            std::tie(regRMSD_H, regRMSD_dH) = DimRed_trainer::RMSD_reg(RegSet->example());
            std::tie(degRMSD_H, degRMSD_dH) = DimRed_trainer::RMSD_deg(DegSet->example());
        }
        else {
            std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
            std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
        }
        std::cout << "For  regular   data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
                  << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
        for (int i = 0; i < Hd::NStates; i++) {
            torch::save(Hd::nets[i][i][0], "Hd" + std::to_string(i+1) + std::to_string(i+1) + ".net");
            for (int j = i + 1; j < Hd::NStates; j++)
            for (int k = 0; k < Hd::nets[i][j].size(); k++)
            torch::save(Hd::nets[i][j][k], "Hd" + std::to_string(i+1) + std::to_string(j+1) + "_" + std::to_string(k) + ".net");
        }
        if (train_DimRed)
        for (size_t irred = 0; irred < DimRed::nets.size(); irred++)
        torch::save(DimRed::nets[irred], "DimRed" + std::to_string(irred+1) + ".net");
    }
}

} // namespace train_Hd