/*
To load and process ab initio data

The most basic ab initio data is geometry, who is the independent variable for any observable
The geometry itself can also feed unsupervised learning e.g. autoencoder

A common label is Hamiltonian (and gradient), who is classified into regular or degenerate based on degeneracy threshold
A regular Hamiltonian is in adiabatic representation, while a degenerate one is in composite representation
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "observable_net.hpp"
#include "AbInitio.hpp"

namespace AbInitio {

// Degeneracy threshold for Hamiltonian
const double DegThresh = 0.0001;

GeomLoader::GeomLoader() {
    r = at::empty(SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
}
GeomLoader::~GeomLoader() {}

geom::geom() {}
geom::geom(const GeomLoader & loader) {
    at::Tensor q = CL::TS::IC::compute_IC(loader.r);
    SSAq = SSAIC::compute_SSAIC(q);
}
geom::~geom() {}
void geom::to(const c10::DeviceType & device) {
    for (at::Tensor & irred : SSAq) irred = irred.to(device);
}

HamLoader::HamLoader() {}
HamLoader::HamLoader(const int64_t & NStates) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    r      = at::empty(SSAIC::cartdim, top);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, SSAIC::cartdim}, top);
}
HamLoader::~HamLoader() {}
void HamLoader::reset(const int64_t & NStates)  {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    r      = at::empty(SSAIC::cartdim, top);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, SSAIC::cartdim}, top);
}
// Subtract energy zero point
void HamLoader::subtract_zero_point(const double & zero_point) {
    energy -= zero_point;
}

RegHam::RegHam() {}
RegHam::RegHam(const HamLoader & loader, const bool & train_DimRed) {
    at::Tensor q, J_q_r;
    std::tie(q, J_q_r) = CL::TS::IC::compute_IC_J(loader.r);
    q.set_requires_grad(true);
    if (train_DimRed) {
        // Scaled and symmetry adapted internal coordinate and
        // its Jacobian^T w.r.t. Cartesian coordinate
        SSAq = SSAIC::compute_SSAIC(q);
        J_SSAq_r_T.resize(SSAq.size());
        for (size_t irred = 0; irred < SSAq.size(); irred++) {
            at::Tensor J_SSAq_q = at::empty(
                {SSAq[irred].numel(), q.numel()},
                at::TensorOptions().dtype(torch::kFloat64));
            for (size_t i = 0; i < SSAq[irred].numel(); i++) {
                torch::autograd::variable_list g = torch::autograd::grad({SSAq[irred][i]}, {q}, {}, true);
                J_SSAq_q[i].copy_(g[0]);
            }
            J_SSAq_r_T[irred] = (J_SSAq_q.mm(J_q_r)).transpose(0, 1);
        }
        // Stop autograd
        for (at::Tensor & irred : SSAq) irred.detach_();
    }
    else {
        // Observable net input layer and
        // its Jacobian^T w.r.t. Cartesian coordinate
        input_layer = ON::input_layer(DimRed::reduce(SSAIC::compute_SSAIC(q)));
        J_IL_r_T.resize(input_layer.size());
        for (size_t irred = 0; irred < input_layer.size(); irred++) {
            at::Tensor J_IL_q = at::empty(
                {input_layer[irred].size(0), q.size(0)},
                at::TensorOptions().dtype(torch::kFloat64));
            for (size_t i = 0; i < input_layer[irred].size(0); i++) {
                torch::autograd::variable_list g = torch::autograd::grad({input_layer[irred][i]}, {q}, {}, true);
                J_IL_q[i].copy_(g[0]);
            }
            J_IL_r_T[irred] = (J_IL_q.mm(J_q_r)).transpose_(0, 1);
        }
        // Stop autograd
        for (at::Tensor & irred : input_layer) irred.detach_();
    }
    // energy and dH
    energy = loader.energy.clone();
    dH = loader.dH.clone();
}
RegHam::~RegHam() {}
void RegHam::to(const c10::DeviceType & device) {
    for (at::Tensor & irred : SSAq)        irred = irred.to(device);
    for (at::Tensor & irred : J_SSAq_r_T)  irred = irred.to(device);
    for (at::Tensor & irred : input_layer) irred = irred.to(device);
    for (at::Tensor & irred : J_IL_r_T)    irred = irred.to(device);
    energy = energy.to(device);
        dH =     dH.to(device);
}
void RegHam::adjust_weight(const double & Ethresh) {
    double temp = energy[0].item<double>();
    if ( temp > Ethresh) {
        temp = Ethresh / temp;
        weight = temp * temp;
    }
}

DegHam::DegHam() {}
DegHam::DegHam(const HamLoader & loader, const bool & train_DimRed) : RegHam(loader, train_DimRed) {
    // H and dH in composite representation
    H = energy;
    CL::TS::chemistry::composite_representation(H, dH);
}
DegHam::~DegHam() {}
void DegHam::to(const c10::DeviceType & device) {
    for (at::Tensor & irred : SSAq)        irred = irred.to(device);
    for (at::Tensor & irred : J_SSAq_r_T)  irred = irred.to(device);
    for (at::Tensor & irred : input_layer) irred = irred.to(device);
    for (at::Tensor & irred : J_IL_r_T)    irred = irred.to(device);
     H =   H.to(device);
    dH = dH.to(device);
}

DataSet<geom> * read_GeomSet(const std::vector<std::string> & data_set) {
    DataSet<geom> * GeomSet;
    // Count the number of data
    std::vector<size_t> NDataPerSet(data_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        NDataPerSet[its] = CL::utility::NLines(data_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // data set loader
    std::vector<geom *> vec_p_geom(NDataTotal);
    // Read training set files
    size_t count = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        // raw data loader
        std::vector<GeomLoader> RawGeomLoader(NDataPerSet[its]);
        // r
        std::ifstream ifs; ifs.open(data_set[its]+"geom.data");
            for (auto & loader : RawGeomLoader)
            for (size_t i = 0; i < SSAIC::cartdim / 3; i++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; loader.r[3*i  ] = dbletemp;
                ifs >> dbletemp; loader.r[3*i+1] = dbletemp;
                ifs >> dbletemp; loader.r[3*i+2] = dbletemp;
            }
        ifs.close();
        // Insert to data set loader
        for (auto & loader : RawGeomLoader) {
            vec_p_geom[count] = new geom(loader);
            count++;
        }
    }
    // Create DataSet with data set loader
    GeomSet = new DataSet<geom>(vec_p_geom);
    return GeomSet;
}

std::tuple<DataSet<RegHam> *, DataSet<DegHam> *> read_HamSet(
const std::vector<std::string> & data_set, const bool & train_DimRed,
const double & zero_point, const double & weight) {
    // Count the number of data
    std::vector<size_t> NDataPerSet(data_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        NDataPerSet[its] = CL::utility::NLines(data_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // data set loader
    std::vector<RegHam *> vec_p_RegData(NDataTotal);
    std::vector<DegHam *> vec_p_DegData(NDataTotal);
    size_t NRegData = 0;
    size_t NDegData = 0;
    // Read training set files
    for (size_t its = 0; its < data_set.size(); its++) {
        // for file input
        std::ifstream ifs;
        std::string line;
        std::vector<std::string> strs;
        // NStates
        ifs.open(data_set[its]+"energy.data"); std::getline(ifs, line); ifs.close();
        CL::utility::split(line, strs);
        int64_t NStates = strs.size();
        std::ifstream ifs_valid_states; ifs_valid_states.open(data_set[its]+"valid_states");
            if (ifs_valid_states.good()) {
                int64_t valid_states;
                ifs_valid_states >> valid_states;
                NStates = valid_states < NStates ? valid_states: NStates;
            }
        ifs_valid_states.close();
        // raw data loader
        std::vector<HamLoader> RawDataLoader(NDataPerSet[its]);
        for (auto & loader : RawDataLoader) loader.reset(NStates);
        // r
        ifs.open(data_set[its]+"geom.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < SSAIC::cartdim / 3; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; RawDataLoader[i].r[3*j  ] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].r[3*j+1] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].r[3*j+2] = dbletemp;
            }
        ifs.close();
        // energy
        ifs.open(data_set[its]+"energy.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++) {
                std::getline(ifs, line);
                CL::utility::split(line, strs);
                for (size_t j = 0; j < NStates; j++)
                RawDataLoader[i].energy[j] = std::stod(strs[j]);
            }
        ifs.close();
        // dH
        for (size_t istate = 0; istate < NStates; istate++) {
            ifs.open(data_set[its]+"cartgrad-"+std::to_string(istate+1)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++)
                for (size_t j = 0; j < SSAIC::cartdim; j++) {
                    double dbletemp; ifs >> dbletemp;
                    RawDataLoader[i].dH[istate][istate][j] = dbletemp;
                }
            ifs.close();
        for (size_t jstate = istate+1; jstate < NStates; jstate++) {
            ifs.open(data_set[its]+"cartgrad-"+std::to_string(istate+1)+"-"+std::to_string(jstate+1)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++)
                for (size_t j = 0; j < SSAIC::cartdim; j++) {
                    double dbletemp; ifs >> dbletemp;
                    RawDataLoader[i].dH[istate][jstate][j] = dbletemp;
                }
            ifs.close();
        } }
        // Process raw data
        for (auto & loader : RawDataLoader) {
            loader.subtract_zero_point(zero_point);
            // Insert to data set loader
            if (CL::TS::chemistry::check_degeneracy(DegThresh, loader.energy)) {
                vec_p_DegData[NDegData] = new DegHam(loader, train_DimRed);
                NDegData++;
            } else {
                vec_p_RegData[NRegData] = new RegHam(loader, train_DimRed);
                vec_p_RegData[NRegData]->adjust_weight(weight);
                NRegData++;
            }
        }
    }
    // Create DataSet with data set loader
    vec_p_RegData.resize(NRegData);
    vec_p_DegData.resize(NDegData);
    DataSet<RegHam> * RegSet = new DataSet<RegHam>(vec_p_RegData);
    DataSet<DegHam> * DegSet = new DataSet<DegHam>(vec_p_DegData);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace AbInitio