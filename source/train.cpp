#include <regex>
#include <omp.h>
#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
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
        for (size_t i = 0; i < Hd::NStates; i++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            for (size_t j = 0; j < Hd::NStates; j++)
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
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = 0; j < Hd::NStates; j++)
    Hd::NIrred = Hd::Hd_symm[i][j] > Hd::NIrred ? Hd::Hd_symm[i][j] : Hd::NIrred;
    Hd::NIrred++;
    // Polynomial numbering rule
    std::vector<size_t> NInput_per_irred = Hd::input::prepare_PNR(Hd_input_layer_in);
    // Initialize networks
    Hd::nets.resize(Hd::NStates);
    for (size_t i = 0; i < Hd::NStates; i++) {
        Hd::nets[i].resize(Hd::NStates);
        for (size_t j = i; j < Hd::NStates; j++) {
            Hd::nets[i][j] = std::make_shared<Hd::Net>(NInput_per_irred[Hd::Hd_symm[i][j]],
                Hd::Hd_symm[i][j] == 0, max_depth);
            Hd::nets[i][j]->to(torch::kFloat64);
        }
    }
    if (! chk.empty()) {
        assert(("Wrong number of checkpoint files", chk.size() == (Hd::NStates+1)*Hd::NStates/2));
        size_t count = 0;
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i; j < Hd::NStates; j++) {
            Hd::nets[i][j]->warmstart(chk[count], chk_depth);
            count++;
        }
    }
    else {
        assert(("Wrong number of initial guess of Hd diagonal", guess_diag.size() == Hd::NStates));
        for (size_t i = 0; i < Hd::NStates; i++)
        (*(Hd::nets[i][i]->fc[Hd::nets[i][i]->fc.size()-1]))->bias.data_ptr<double>()[0] = guess_diag[i];
    }
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i; j < Hd::NStates; j++) {
        Hd::nets[i][j]->freeze(freeze);
        std::cout << "Number of trainable parameters for Hd" << i+1 << j+1 << " = "
            << CL::TS::NParameters(Hd::nets[i][j]->parameters()) << '\n';
    }
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

at::Tensor loss_reg(AbInitio::RegData * data) {
    // Compute diabatic quantity
    at::Tensor H = Hd::compute_Hd_from_input_layer(data->input_layer);
    at::Tensor dH = data->dH.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i; j < Hd::NStates; j++) {
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
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i+1; j < Hd::NStates; j++)
    loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
    loss *= 2.0;
    for (size_t i = 0; i < Hd::NStates; i++)
    loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
    // + energy loss
    loss += unit_square * torch::mse_loss(energy, data->energy, at::Reduction::Sum);
    return data->weight * loss;
}
at::Tensor loss_deg(AbInitio::DegData * data) {
    // Compute diabatic quantity
    at::Tensor H = Hd::compute_Hd_from_input_layer(data->input_layer);
    at::Tensor dH = data->dH.new_empty(data->dH.sizes());
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i; j < Hd::NStates; j++) {
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
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i+1; j < Hd::NStates; j++)
    loss += unit_square * (H[i][j] - data->H[i][j]).pow(2)
            + (dH[i][j] - data->dH[i][j]).pow(2).sum();
    loss *= 2.0;
    for (size_t i = 0; i < Hd::NStates; i++)
    loss += unit_square * (H[i][i] - data->H[i][i]).pow(2)
            + (dH[i][i] - data->dH[i][i]).pow(2).sum();
    return loss;
}

std::tuple<double, double> RMSD_reg(const std::vector<AbInitio::RegData *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    for (auto & data : DataSet) {
        // Compute diabatic quantity
        at::Tensor H = Hd::compute_Hd_from_input_layer(data->input_layer);
        at::Tensor dH = data->dH.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i; j < Hd::NStates; j++) {
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
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i+1; j < Hd::NStates; j++)
        loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
        loss *= 2.0;
        for (size_t i = 0; i < Hd::NStates; i++)
        loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
        e_dH += loss.item<double>() / data_NStates / data_NStates;
    }
    e_H  /= DataSet.size();
    e_dH /= DataSet.size() * SSAIC::cartdim;
    return std::make_tuple(std::sqrt(e_H), std::sqrt(e_dH));
}
std::tuple<double, double> RMSD_deg(const std::vector<AbInitio::DegData *> DataSet) {
    double e_H = 0.0, e_dH = 0.0;
    for (auto & data : DataSet) {
        // Compute diabatic quantity
        at::Tensor H = Hd::compute_Hd_from_input_layer(data->input_layer);
        at::Tensor dH = data->dH.new_empty(data->dH.sizes());
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i; j < Hd::NStates; j++) {
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
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i+1; j < Hd::NStates; j++) {
             H_loss += ( H[i][j] - data-> H[i][j]).pow(2);
            dH_loss += (dH[i][j] - data->dH[i][j]).pow(2).sum();
        }
        H_loss *= 2.0; dH_loss *= 2.0;
        for (size_t i = 0; i < Hd::NStates; i++) {
             H_loss += ( H[i][i] - data-> H[i][i]).pow(2);
            dH_loss += (dH[i][i] - data->dH[i][i]).pow(2).sum();
        }
        e_H  +=  H_loss.item<double>() / data_NStates / data_NStates;
        e_dH += dH_loss.item<double>() / data_NStates / data_NStates;
    }
    e_H  /= DataSet.size();
    e_dH /= DataSet.size() * SSAIC::cartdim;
    return std::make_tuple(std::sqrt(e_H), std::sqrt(e_dH));
}

void train(const std::string & Hd_in, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & data_set, const double & zero_point,
const std::vector<std::string> & chk, const size_t & chk_depth, const std::vector<double> & guess_diag,
const std::string & opt, const size_t & epoch, const size_t & batch_size, const double & learning_rate) {
    std::cout << "Start training\n";
    // Initialize network
    if (guess_diag.empty()) {
        std::vector<double> guess_diag_work(Hd::NStates, 0.0);
        define_Hd(Hd_in, max_depth, freeze,
            chk, chk_depth, guess_diag_work);
    }
    else {
        define_Hd(Hd_in, max_depth, freeze,
            chk, chk_depth, guess_diag);
    }
    // Read data set
    AbInitio::DataSet<AbInitio::RegData> * RegSet;
    AbInitio::DataSet<AbInitio::DegData> * DegSet;
    std::tie(RegSet, DegSet) = AbInitio::read_DataSet(data_set, zero_point);
    std::cout << "Number of regular data = " << RegSet->size_int() << '\n';
    std::cout << "Number of degenerate data = " << DegSet->size_int() << '\n';
    // Initialize underlying modules
    CL::TS::chemistry::initialize_phase_fixing(Hd::NStates);
    set_unit(RegSet->example());
    if (opt == "Adam" || opt == "SGD") {
        // Create data set loader
        auto reg_loader = torch::data::make_data_loader(* RegSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        auto deg_loader = torch::data::make_data_loader(* DegSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        std::cout << "batch size = " << batch_size << '\n';
        // Concatenate network parameters
        std::vector<at::Tensor> parameters;
        for (size_t i = 0; i < Hd::NStates; i++)
        for (size_t j = i; j < Hd::NStates; j++) {
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
                              << "For regular data, RMSD(E) = " << regRMSD_H << ", RMSD(dH) = " << regRMSD_dH << '\n'
                              << "For degenerate data, RMSD(H) = " << degRMSD_H << ", RMSD(dH) = " << degRMSD_dH << '\n';
                    for (size_t i = 0; i < Hd::NStates; i++)
                    for (size_t j = i; j < Hd::NStates; j++)
                    torch::save(Hd::nets[i][j], "Hd"+std::to_string(i)+std::to_string(j)+"_"+std::to_string(iepoch)+".net");
                    torch::save(optimizer, "Hd_"+std::to_string(iepoch)+".opt");
                }
            }
        }
    }
    else {

    }
}

} // namespace train