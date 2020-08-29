#include <regex>
#include <omp.h>
#include <torch/torch.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "Hd.hpp"
#include "AbInitio.hpp"

namespace train {

// The "unit" of energy, accounting for the unit difference bewteen energy and gradient
double unit, unit_square;

void define_Hd(const std::string & Hd_in, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & chk, const size_t & chk_depth, const std::vector<double> & guess_diag) {
    std::ifstream ifs; ifs.open(Hd_in);
        std::string line;
        std::vector<std::string> strs;
        // Number of electronic states
        std::getline(ifs, line);
        std::getline(ifs, line);
        Hd::NStates = std::stoul(line);
        // Symmetry of Hd elements
        std::getline(ifs, line);
        CL::utility::CreateArray(Hd::Hd_symm, Hd::NStates, Hd::NStates);
        for (int i = 0; i < Hd::NStates; i++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            for (int j = 0; j < Hd::NStates; j++)
            Hd::Hd_symm[i][j] = std::stoul(strs[j]) - 1;
        }
        // Input layer specification file
        std::string Hd_input_layer_in;
        std::getline(ifs, line);
        std::getline(ifs, Hd_input_layer_in);
        CL::utility::trim(Hd_input_layer_in);
    ifs.close();
    // Number of irreducible representations
    Hd::NIrred = 0;
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = 0; j < Hd::NStates; j++)
    Hd::NIrred = Hd::Hd_symm[i][j] > Hd::NIrred ? Hd::Hd_symm[i][j] : Hd::NIrred;
    Hd::NIrred++;
    // Polynomial numbering rule
    std::vector<size_t> NInput_per_irred = Hd::input::prepare_PNR(Hd_input_layer_in);
    // Initialize networks
    Hd::nets.resize(Hd::NStates);
    for (int i = 0; i < Hd::NStates; i++) {
        Hd::nets[i].resize(Hd::NStates);
        for (int j = i; j < Hd::NStates; j++) {
            Hd::nets[i][j] = std::make_shared<Hd::Net>(NInput_per_irred[Hd::Hd_symm[i][j]],
                Hd::Hd_symm[i][j] == 0, max_depth);
            Hd::nets[i][j]->to(torch::kFloat64);
        }
    }
    if (! chk.empty()) {
        assert(("Wrong number of checkpoint files", chk.size() == (Hd::NStates+1)*Hd::NStates/2));
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++) {
            Hd::nets[i][j]->warmstart(chk[count], chk_depth);
            count++;
        }
    }
    else {
        assert(("Wrong number of initial guess of Hd diagonal", guess_diag.size() == Hd::NStates));
        for (int i = 0; i < Hd::NStates; i++)
        (*(Hd::nets[i][i]->fc[Hd::nets[i][i]->fc.size()-1]))->bias.data_ptr<double>()[0] = guess_diag[i];
    }
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        Hd::nets[i][j]->freeze(freeze);
        std::cout << "Number of trainable network parameters for Hd" << i+1 << j+1 << " = "
            << CL::TS::NParameters(Hd::nets[i][j]->parameters()) << '\n';
    }
}

// Compute Hd from input layer determined in AbInitio
at::Tensor compute_Hd(const std::vector<at::Tensor> & input_layer) {
    // Compute upper triangle
    at::Tensor Hd = input_layer[0].new_empty({Hd::NStates, Hd::NStates});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++)
    Hd[i][j] = Hd::nets[i][j]->forward(input_layer[Hd::Hd_symm[i][j]]);
    return Hd;
}
at::Tensor compute_Hd(const std::vector<at::Tensor> & input_layer,
const std::vector<std::vector<std::shared_ptr<Hd::Net>>> & nets) {
    // Compute upper triangle
    at::Tensor Hd = input_layer[0].new_empty({Hd::NStates, Hd::NStates});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++)
    Hd[i][j] = nets[i][j]->forward(input_layer[Hd::Hd_symm[i][j]]);
    return Hd;
}

void set_unit(const std::vector<AbInitio::RegData *> & RegSet) {
    double MaxEnergy = 0.0, MaxGrad = 0.0;
    for (auto & data : RegSet) {
        double temp = std::abs(data->energy[0].item<double>());
        MaxEnergy = temp > MaxEnergy ? temp : MaxEnergy;
        temp = data->dH[0][0].norm().item<double>();
        MaxGrad = temp > MaxGrad ? temp : MaxGrad;
    }
    unit = MaxGrad / MaxEnergy;
    unit_square = unit * unit;
    std::cout << "The typical work length of this system = " << 1.0/unit << '\n';
}

// To make use of Fortran-Library nonlinear optimizers:
//     1. Map the network parameters to vector c
//     2. Compute residue and Jacobian
namespace FLopt {
    const double Sqrt2 = 1.4142135623730951;

    int OMP_NUM_THREADS;

    // Hd element networks (each thread owns a copy)
    std::vector<std::vector<std::vector<std::shared_ptr<Hd::Net>>>> netss;

    // Data set
    std::vector<AbInitio::RegData *> RegSet;
    std::vector<AbInitio::DegData *> DegSet;
    // Divide data set for parallelism
    std::vector<size_t> RegChunk, DegChunk, start;

    // Push network parameters to parameter vector c
    void p2c(const std::vector<std::vector<std::shared_ptr<Hd::Net>>> & nets, double * c) {
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : nets[i][j]->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(c[count]), p.data_ptr<double>(), p.numel() * sizeof(double));
            count += p.numel();
        }
    }
    // The other way round
    void c2p(const double * c, const std::vector<std::vector<std::shared_ptr<Hd::Net>>> & nets) {
        torch::NoGradGuard no_grad;
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : nets[i][j]->parameters())
        if (p.requires_grad()) {
            std::memcpy(p.data_ptr<double>(), &(c[count]), p.numel() * sizeof(double));
            count += p.numel();
        }
    }

    // Push the network parameter gradient to Jacobian^T 
    void dp2JT(const std::vector<std::vector<std::shared_ptr<Hd::Net>>> & nets,
    double * JT, const int & NEq, const size_t & column) {
        size_t row = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : nets[i][j]->parameters())
        if (p.requires_grad()) {
            double * pg = p.grad().data_ptr<double>();
            for (size_t i = 0; i < p.grad().numel(); i++) {
                JT[row * NEq + column] = pg[i];
                row++;
            }
        }
    }

    void net_zero_grad(const std::vector<std::vector<std::shared_ptr<Hd::Net>>> & nets) {
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : nets[i][j]->parameters())
        if (p.requires_grad() && p.grad().defined()) {
            p.grad().detach_();
            p.grad().zero_();
        };
    }

    void loss(double & l, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & nets = netss[thread];
            c2p(c, nets);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->energy.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += (dH[i][i] - data->dH[i][i]).pow(2).sum();
                // + energy loss
                loss_data += unit_square * torch::mse_loss(energy, data->energy, at::Reduction::Sum);
                loss[thread] += data->weight * loss_data;
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->H.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Transform to composite representation
                at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
                at::Tensor eigval, eigvec;
                std::tie(eigval, eigvec) = dHdH.symeig(true, true);
                dHdH = eigvec.transpose(0, 1);
                H = dHdH.mm(energy.diag().mm(eigvec));
                dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // H and dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += unit_square * (H[i][j] - data->H[i][j]).pow(2)
                        + (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += unit_square * (H[i][i] - data->H[i][i]).pow(2)
                        + (dH[i][i] - data->dH[i][i]).pow(2).sum();
                loss[thread] += loss_data;
            }
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
    }
    void grad(double * g, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & nets = netss[thread];
            c2p(c, nets);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->energy.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += (dH[i][i] - data->dH[i][i]).pow(2).sum();
                // + energy loss
                loss_data += unit_square * torch::mse_loss(energy, data->energy, at::Reduction::Sum);
                loss[thread] += data->weight * loss_data;
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->H.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Transform to composite representation
                at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
                at::Tensor eigval, eigvec;
                std::tie(eigval, eigvec) = dHdH.symeig(true, true);
                dHdH = eigvec.transpose(0, 1);
                H = dHdH.mm(energy.diag().mm(eigvec));
                dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // H and dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += unit_square * (H[i][j] - data->H[i][j]).pow(2)
                        + (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += unit_square * (H[i][i] - data->H[i][i]).pow(2)
                        + (dH[i][i] - data->dH[i][i]).pow(2).sum();
                loss[thread] += loss_data;
            }
            net_zero_grad(nets);
            loss[thread].backward();
        }
        // Push network gradients to g
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : netss[0][i][j]->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & p : netss[thread][i][j]->parameters())
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
            auto & nets = netss[thread];
            c2p(c, nets);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->energy.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += (dH[i][i] - data->dH[i][i]).pow(2).sum();
                // + energy loss
                loss_data += unit_square * torch::mse_loss(energy, data->energy, at::Reduction::Sum);
                loss[thread] += data->weight * loss_data;
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->H.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Transform to composite representation
                at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
                at::Tensor eigval, eigvec;
                std::tie(eigval, eigvec) = dHdH.symeig(true, true);
                dHdH = eigvec.transpose(0, 1);
                H = dHdH.mm(energy.diag().mm(eigvec));
                dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // H and dH loss
                at::Tensor loss_data = H.new_zeros({});
                for (int i = 0; i < data_NStates; i++)
                for (int j = i+1; j < data_NStates; j++)
                loss_data += unit_square * (H[i][j] - data->H[i][j]).pow(2)
                        + (dH[i][j] - data->dH[i][j]).pow(2).sum();
                loss_data *= 2.0;
                for (int i = 0; i < data_NStates; i++)
                loss_data += unit_square * (H[i][i] - data->H[i][i]).pow(2)
                        + (dH[i][i] - data->dH[i][i]).pow(2).sum();
                loss[thread] += loss_data;
            }
            net_zero_grad(nets);
            loss[thread].backward();
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
        // Push network gradients to g
        size_t count = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        for (auto & p : netss[0][i][j]->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++)
            for (auto & p : netss[thread][i][j]->parameters())
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
            auto & nets = netss[thread];
            c2p(c, nets);
            size_t count = start[thread];
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->energy.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // energy residue
                at::Tensor r_E = data->weight * unit * (energy - data->energy);
                std::memcpy(&(r[count]), r_E.data_ptr<double>(), r_E.numel() * sizeof(double));
                count += r_E.numel();
                // dH residue
                for (int i = 0; i < data_NStates; i++) {
                    at::Tensor r_dH = data->weight * (dH[i][i] - data->dH[i][i]);
                    std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                    count += r_dH.numel();
                    for (int j = i+1; j < data_NStates; j++) {
                        at::Tensor r_dH = data->weight * Sqrt2 * (dH[i][j] - data->dH[i][j]);
                        std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                        count += r_dH.numel();
                    }
                }
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->H.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Transform to composite representation
                at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
                at::Tensor eigval, eigvec;
                std::tie(eigval, eigvec) = dHdH.symeig(true, true);
                dHdH = eigvec.transpose(0, 1);
                H = dHdH.mm(energy.diag().mm(eigvec));
                dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // H and dH residue
                for (int i = 0; i < data_NStates; i++) {
                    r[count] = unit * (H[i][i] - data->H[i][i]).item<double>();
                    count++;
                    at::Tensor r_dH = dH[i][i] - data->dH[i][i];
                    std::memcpy(&(r[count]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
                    count += r_dH.numel();
                    for (int j = i+1; j < data_NStates; j++) {
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
    void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & nets = netss[thread];
            c2p(c, nets);
            size_t column = start[thread];
            for (size_t idata = RegChunk[thread] - RegChunk[0]; idata < RegChunk[thread]; idata++) {
                auto & data = RegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->energy.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // energy Jacobian
                at::Tensor r_E = data->weight * unit * (energy - data->energy);
                for (size_t el = 0; el < r_E.numel(); el++) {
                    net_zero_grad(nets);
                    r_E[el].backward({}, true);
                    dp2JT(nets, JT, NEq, column);
                    column++;
                }
                // dH Jacobian
                for (int i = 0; i < data_NStates; i++) {
                    at::Tensor r_dH = data->weight * (dH[i][i] - data->dH[i][i]);
                    for (size_t el = 0; el < r_dH.numel(); el++) {
                        net_zero_grad(nets);
                        r_dH[el].backward({}, true);
                        dp2JT(nets, JT, NEq, column);
                        column++;
                    }
                    for (int j = i+1; j < data_NStates; j++) {
                        at::Tensor r_dH = data->weight * Sqrt2 * (dH[i][j] - data->dH[i][j]);
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(nets);
                            r_dH[el].backward({}, true);
                            dp2JT(nets, JT, NEq, column);
                            column++;
                        }
                    }
                }
            }
            for (size_t idata = DegChunk[thread] - DegChunk[0]; idata < DegChunk[thread]; idata++) {
                auto & data = DegSet[idata];
                // Compute diabatic quantity
                at::Tensor H = compute_Hd(data->input_layer, nets);
                at::Tensor dH = H.new_empty(data->dH.sizes());
                for (int i = 0; i < Hd::NStates; i++)
                for (int j = i; j < Hd::NStates; j++) {
                    if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                        data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                        data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                    };
                    H[i][j].backward({}, true);
                    dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
                }
                // Transform to adiabatic representation
                at::Tensor energy, state;
                std::tie(energy, state) = H.symeig(true, true);
                dH = CL::TS::LA::UT_A3_U(dH, state);
                // Slice to the number of states in data
                int data_NStates = data->H.size(0);
                if (Hd::NStates != data_NStates) {
                    energy = energy.slice(0, 0, data_NStates+1);
                    dH = dH.slice(0, 0, data_NStates+1);
                    dH = dH.slice(1, 0, data_NStates+1);
                }
                // Transform to composite representation
                at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
                at::Tensor eigval, eigvec;
                std::tie(eigval, eigvec) = dHdH.symeig(true, true);
                dHdH = eigvec.transpose(0, 1);
                H = dHdH.mm(energy.diag().mm(eigvec));
                dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
                // Determine phase difference of eigenvectors
                CL::TS::chemistry::fix(dH, data->dH);
                // H and dH Jacobian
                for (int i = 0; i < data_NStates; i++) {
                    at::Tensor r_H = H[i][i] - data->H[i][i];
                    net_zero_grad(nets);
                    r_H.backward({}, true);
                    dp2JT(nets, JT, NEq, column);
                    column++;
                    at::Tensor r_dH = dH[i][i] - data->dH[i][i];
                    for (size_t el = 0; el < r_dH.numel(); el++) {
                        net_zero_grad(nets);
                        r_dH[el].backward({}, true);
                        dp2JT(nets, JT, NEq, column);
                        column++;
                    }
                    for (int j = i+1; j < data_NStates; j++) {
                        at::Tensor r_H = Sqrt2 * (H[i][j] - data->H[i][j]);
                        net_zero_grad(nets);
                        r_H.backward({}, true);
                        dp2JT(nets, JT, NEq, column);
                        column++;
                        at::Tensor r_dH = Sqrt2 * (dH[i][j] - data->dH[i][j]);
                        for (size_t el = 0; el < r_dH.numel(); el++) {
                            net_zero_grad(nets);
                            r_dH[el].backward({}, true);
                            dp2JT(nets, JT, NEq, column);
                            column++;
                        }
                    }
                }
            }
        }
    }

    void initialize(const size_t & freeze_,
    const std::vector<AbInitio::RegData *> & RegSet_,
    const std::vector<AbInitio::DegData *> & DegSet_) {
        OMP_NUM_THREADS = omp_get_max_threads();
        std::cout << "The number of threads = " << OMP_NUM_THREADS << '\n';

        netss.resize(OMP_NUM_THREADS);
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & nets = netss[thread];
            nets.resize(Hd::NStates);
            for (int i = 0; i < Hd::NStates; i++) {
                nets[i].resize(Hd::NStates);
                for (int j = i; j < Hd::NStates; j++) {
                    nets[i][j] = std::make_shared<Hd::Net>(
                        (*(Hd::nets[i][j]->fc[0]))->options.in_features(),
                        Hd::Hd_symm[i][j] == 0, Hd::nets[i][j]->fc.size());
                    nets[i][j]->to(torch::kFloat64);
                    nets[i][j]->copy(Hd::nets[i][j]);
                    nets[i][j]->freeze(freeze_);
                }
            }
        }

        RegSet = RegSet_;
        DegSet = DegSet_;
        std::cout << "For parallelism, the number of regular data in use = "
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
    }

    void optimize(const std::string & opt, const size_t & epoch) {
        // Initialize
        int Nc = 0;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        Nc += CL::TS::NParameters(netss[0][i][j]->parameters());
        std::cout << "There are " << Nc << " parameters to train\n";
        double * c = new double[Nc];
        p2c(Hd::nets, c);
        int NEq = 0;
        for (size_t i = 0; i < OMP_NUM_THREADS * (RegSet.size() / OMP_NUM_THREADS); i++)
        NEq += RegSet[i]->energy.size(0) + (RegSet[i]->dH.size(0)+1)*RegSet[i]->dH.size(0)/2 * RegSet[i]->dH.size(2);
        for (size_t i = 0; i < OMP_NUM_THREADS * (DegSet.size() / OMP_NUM_THREADS); i++)
        NEq += (DegSet[i]->dH.size(0)+1)*DegSet[i]->dH.size(0)/2 * (1 + DegSet[i]->dH.size(2));
        std::cout << "The data set corresponds to " << NEq << " least square equations" << std::endl;
        // Train
        if (opt == "CG") {
            FL::NO::ConjugateGradient(loss, grad, loss_grad, c, Nc, "DY", false, true, epoch);
        } else {
            FL::NO::TrustRegion(residue, Jacobian, c, NEq, Nc, true, epoch);
        }
        // Finish
        c2p(c, Hd::nets);
        delete [] c;
    }
} // namespace FLopt

at::Tensor loss_reg(AbInitio::RegData * data) {
    // Compute diabatic quantity
    at::Tensor H = compute_Hd(data->input_layer);
    at::Tensor dH = H.new_empty(data->dH.sizes());
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
            data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
            data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
        };
        H[i][j].backward({}, true);
        dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true, true);
    dH = CL::TS::LA::UT_A3_U(dH, state);
    // Slice to the number of states in data
    int data_NStates = data->energy.size(0);
    if (Hd::NStates != data_NStates) {
        energy = energy.slice(0, 0, data_NStates+1);
        dH = dH.slice(0, 0, data_NStates+1);
        dH = dH.slice(1, 0, data_NStates+1);
    }
    // Determine phase difference of eigenvectors
    CL::TS::chemistry::fix(dH, data->dH);
    // dH loss
    at::Tensor loss = H.new_zeros({});
    for (int i = 0; i < data_NStates; i++)
    for (int j = i+1; j < data_NStates; j++)
    loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
    loss *= 2.0;
    for (int i = 0; i < data_NStates; i++)
    loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
    // + energy loss
    loss += unit_square * torch::mse_loss(energy, data->energy, at::Reduction::Sum);
    return data->weight * loss;
}
at::Tensor loss_deg(AbInitio::DegData * data) {
    // Compute diabatic quantity
    at::Tensor H = compute_Hd(data->input_layer);
    at::Tensor dH = H.new_empty(data->dH.sizes());
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
            data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
            data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
        };
        H[i][j].backward({}, true);
        dH[i][j]= data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true, true);
    dH = CL::TS::LA::UT_A3_U(dH, state);
    // Slice to the number of states in data
    int data_NStates = data->H.size(0);
    if (Hd::NStates != data_NStates) {
        energy = energy.slice(0, 0, data_NStates+1);
        dH = dH.slice(0, 0, data_NStates+1);
        dH = dH.slice(1, 0, data_NStates+1);
    }
    // Transform to composite representation
    at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true, true);
    dHdH = eigvec.transpose(0, 1);
    H = dHdH.mm(energy.diag().mm(eigvec));
    dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
    // Determine phase difference of eigenvectors
    CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
    // H and dH loss
    at::Tensor loss = H.new_zeros({});
    for (int i = 0; i < data_NStates; i++)
    for (int j = i+1; j < data_NStates; j++)
    loss += unit_square * (H[i][j] - data->H[i][j]).pow(2)
            + (dH[i][j] - data->dH[i][j]).pow(2).sum();
    loss *= 2.0;
    for (int i = 0; i < data_NStates; i++)
    loss += unit_square * (H[i][i] - data->H[i][i]).pow(2)
            + (dH[i][i] - data->dH[i][i]).pow(2).sum();
    return loss;
}

std::tuple<double, double> RMSD_reg(const std::vector<AbInitio::RegData *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    if (! DataSet.empty()) {
        for (auto & data : DataSet) {
            // Compute diabatic quantity
            at::Tensor H = compute_Hd(data->input_layer);
            at::Tensor dH = H.new_empty(data->dH.sizes());
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++) {
                if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                    data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                    data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                };
                H[i][j].backward({}, true);
                dH[i][j] = data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
            }
            // Transform to adiabatic representation
            at::Tensor energy, state;
            std::tie(energy, state) = H.symeig(true, true);
            dH = CL::TS::LA::UT_A3_U(dH, state);
            // Slice to the number of states in data
            int data_NStates = data->energy.size(0);
            if (Hd::NStates != data_NStates) {
                energy = energy.slice(0, 0, data_NStates+1);
                dH = dH.slice(0, 0, data_NStates+1);
                dH = dH.slice(1, 0, data_NStates+1);
            }
            // Determine phase difference of eigenvectors
            CL::TS::chemistry::fix(dH, data->dH);
            // energy RMSD
            e_H += torch::mse_loss(energy, data->energy, at::Reduction::Sum).item<double>() / data_NStates;
            // dH RMSD
            at::Tensor loss = H.new_zeros({});
            for (int i = 0; i < data_NStates; i++)
            for (int j = i+1; j < data_NStates; j++)
            loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
            loss *= 2.0;
            for (int i = 0; i < data_NStates; i++)
            loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
            e_dH += loss.item<double>() / data_NStates / data_NStates;
        }
        e_H  /= DataSet.size();
        e_dH /= DataSet.size() * DataSet[0]->dH.size(2);
        e_H  = std::sqrt(e_H);
        e_dH = std::sqrt(e_dH);
    }
    return std::make_tuple(e_H, e_dH);
}
std::tuple<double, double> RMSD_deg(const std::vector<AbInitio::DegData *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    if (! DataSet.empty()) {
        for (auto & data : DataSet) {
            // Compute diabatic quantity
            at::Tensor H = compute_Hd(data->input_layer);
            at::Tensor dH = H.new_empty(data->dH.sizes());
            for (int i = 0; i < Hd::NStates; i++)
            for (int j = i; j < Hd::NStates; j++) {
                if (data->input_layer[Hd::Hd_symm[i][j]].grad().defined()) {
                    data->input_layer[Hd::Hd_symm[i][j]].grad().detach_();
                    data->input_layer[Hd::Hd_symm[i][j]].grad().zero_();
                };
                H[i][j].backward({}, true);
                dH[i][j]= data->JT[Hd::Hd_symm[i][j]].mv(data->input_layer[Hd::Hd_symm[i][j]].grad());
            }
            // Transform to adiabatic representation
            at::Tensor energy, state;
            std::tie(energy, state) = H.symeig(true, true);
            dH = CL::TS::LA::UT_A3_U(dH, state);
            // Slice to the number of states in data
            int data_NStates = data->H.size(0);
            if (Hd::NStates != data_NStates) {
                energy = energy.slice(0, 0, data_NStates+1);
                dH = dH.slice(0, 0, data_NStates+1);
                dH = dH.slice(1, 0, data_NStates+1);
            }
            // Transform to composite representation
            at::Tensor dHdH = CL::TS::LA::sy3matdotmul(dH, dH);
            at::Tensor eigval, eigvec;
            std::tie(eigval, eigvec) = dHdH.symeig(true, true);
            dHdH = eigvec.transpose(0, 1);
            H = dHdH.mm(energy.diag().mm(eigvec));
            dH = CL::TS::LA::UT_A3_U(dHdH, dH, eigvec);
            // Determine phase difference of eigenvectors
            CL::TS::chemistry::fix(H, dH, data->H, data->dH, unit_square);
            // H and dH loss
            at::Tensor H_loss = H.new_zeros({}), dH_loss = H.new_zeros({});
            for (int i = 0; i < data_NStates; i++)
            for (int j = i+1; j < data_NStates; j++) {
                 H_loss += ( H[i][j] - data-> H[i][j]).pow(2);
                dH_loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
            }
            H_loss *= 2.0; dH_loss *= 2.0;
            for (int i = 0; i < data_NStates; i++) {
                 H_loss += ( H[i][i] - data-> H[i][i]).pow(2);
                dH_loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
            }
            e_H  +=  H_loss.item<double>() / data_NStates / data_NStates;
            e_dH += dH_loss.item<double>() / data_NStates / data_NStates;
        }
        e_H  /= DataSet.size();
        e_dH /= DataSet.size() * DataSet[0]->dH.size(2);
        e_H  = std::sqrt(e_H);
        e_dH = std::sqrt(e_dH);
    }
    return std::make_tuple(e_H, e_dH);
}

void train(const std::string & Hd_in, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & data_set, const double & zero_point, const double & weight,
const std::vector<std::string> & chk, const size_t & chk_depth, std::vector<double> & guess_diag,
const std::string & opt, const size_t & epoch, const size_t & batch_size, const double & learning_rate) {
    std::cout << "Start training\n";
    // Initialize network
    if (guess_diag.empty()) guess_diag = std::vector<double>(Hd::NStates, 0.0);
    define_Hd(Hd_in, max_depth, freeze, chk, chk_depth, guess_diag);
    // Read data set
    AbInitio::DataSet<AbInitio::RegData> * RegSet;
    AbInitio::DataSet<AbInitio::DegData> * DegSet;
    std::tie(RegSet, DegSet) = AbInitio::read_DataSet(data_set, zero_point, weight);
    std::cout << "Number of regular data = " << RegSet->size_int() << '\n';
    std::cout << "Number of degenerate data = " << DegSet->size_int() << '\n';
    // Initialize underlying modules
    CL::TS::chemistry::initialize_phase_fixing(Hd::NStates);
    set_unit(RegSet->example());
    std::cout << "The initial guess gives:\n";
    double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
    std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
    std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
    std::cout << "For regular data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
              << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
    std::cout << std::endl;
    if (opt == "Adam" || opt == "SGD") {
        // Create data set loader
        auto reg_loader = torch::data::make_data_loader(* RegSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        auto deg_loader = torch::data::make_data_loader(* DegSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        std::cout << "batch size = " << batch_size << '\n';
        // Concatenate network parameters
        std::vector<at::Tensor> parameters;
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++) {
            std::vector<at::Tensor> par = Hd::nets[i][j]->parameters();
            parameters.insert(parameters.end(), par.begin(), par.end());
        }
        if (opt == "Adam") {
            // Create optimizer
            torch::optim::Adam optimizer(parameters, learning_rate);
            if (chk.size() > (Hd::NStates+1)*Hd::NStates/2 && max_depth == chk_depth)
            torch::load(optimizer, chk[(Hd::NStates+1)*Hd::NStates/2]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 1; iepoch <= epoch; iepoch++) {
                for (auto & batch : * reg_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) loss += loss_reg(data);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                for (auto & batch : * deg_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) loss += loss_deg(data);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == 0) {
                    double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
                    std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
                    std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
                    std::cout << "epoch = " << iepoch << '\n'
                              << "For regular data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
                              << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
                    for (int i = 0; i < Hd::NStates; i++)
                    for (int j = i; j < Hd::NStates; j++)
                    torch::save(Hd::nets[i][j], "Hd"+std::to_string(i)+std::to_string(j)+"_"+std::to_string(iepoch)+".net");
                    torch::save(optimizer, "Hd_"+std::to_string(iepoch)+".opt");
                }
            }
        }
        else {
            // Create optimizer
            torch::optim::SGD optimizer(parameters,
                torch::optim::SGDOptions(learning_rate)
                .momentum(0.9).nesterov(true));
            if (chk.size() > (Hd::NStates+1)*Hd::NStates/2 && max_depth == chk_depth)
            torch::load(optimizer, chk[(Hd::NStates+1)*Hd::NStates/2]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 1; iepoch <= epoch; iepoch++) {
                for (auto & batch : * reg_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) loss += loss_reg(data);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                for (auto & batch : * deg_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) loss += loss_deg(data);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == 0) {
                    double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
                    std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
                    std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
                    std::cout << "epoch = " << iepoch << '\n'
                              << "For regular data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
                              << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
                    for (int i = 0; i < Hd::NStates; i++)
                    for (int j = i; j < Hd::NStates; j++)
                    torch::save(Hd::nets[i][j], "Hd"+std::to_string(i)+std::to_string(j)+"_"+std::to_string(iepoch)+".net");
                    torch::save(optimizer, "Hd_"+std::to_string(iepoch)+".opt");
                }
            }
        }
    }
    else {
        FLopt::initialize(freeze, RegSet->example(), DegSet->example());
        FLopt::optimize(opt, epoch);
        double regRMSD_H, regRMSD_dH, degRMSD_H, degRMSD_dH;
        std::tie(regRMSD_H, regRMSD_dH) = RMSD_reg(RegSet->example());
        std::tie(degRMSD_H, degRMSD_dH) = RMSD_deg(DegSet->example());
        std::cout << "For regular data, RMSD(H) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
                  << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
        for (int i = 0; i < Hd::NStates; i++)
        for (int j = i; j < Hd::NStates; j++)
        torch::save(Hd::nets[i][j], "Hd"+std::to_string(i)+std::to_string(j)+".net");
    }
}

} // namespace train