/*
To load and process ab initio data

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "Hd.hpp"
#include "AbInitio.hpp"

namespace AbInitio {

const double DegThresh = 0.0001;

GeomLoader::GeomLoader() {
    r = at::empty(SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
}
GeomLoader::~GeomLoader() {}

geom::geom() {}
geom::geom(const GeomLoader & loader) {
    at::Tensor q = CL::TS::IC::compute_IC(loader.r);
    SAIgeom = SSAIC::compute_SSAIC(q);
}
geom::~geom() {}

DataLoader::DataLoader() {}
DataLoader::~DataLoader() {}
void DataLoader::init(const int64_t & NStates)  {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->r      = at::empty(SSAIC::cartdim, top);
    this->energy = at::empty(NStates, top);
    this->dH     = at::empty({NStates, NStates, SSAIC::cartdim}, top);
}
void DataLoader::cart2int() {
    std::tie(q, J) = CL::TS::IC::compute_IC_J(r);
}
void DataLoader::SubtractRef(const double & zero_point) {
    energy -= zero_point;
}

// Regular data
RegData::RegData() {}
RegData::RegData(DataLoader & loader) {
    // input_layer and J^T
    at::Tensor & q = loader.q;
    q.set_requires_grad(true);
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    input_layer = Hd::input::input_layer(Redgeom);
    JT.resize(input_layer.size());
    for (size_t irred = 0; irred < input_layer.size(); irred++) {
        at::Tensor J_InpLay_q = at::empty(
            {input_layer[irred].size(0), q.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < input_layer[irred].size(0); i++) {
            torch::autograd::variable_list g = torch::autograd::grad({input_layer[irred][i]}, {q}, {}, true);
            J_InpLay_q[i].copy_(g[0]);
        }
        JT[irred] = (J_InpLay_q.mm(loader.J)).transpose(0, 1);
    }
    // Stop autograd
    for (at::Tensor & irred : input_layer) irred.detach_();
    // energy and dH
    energy = loader.energy.clone();
    dH = loader.dH.clone();
}
RegData::~RegData() {}
void RegData::adjust_weight(const double & Ethresh) {
    double temp = energy[0].item<double>();
    if ( temp > Ethresh) {
        temp = Ethresh / temp;
        weight = temp * temp;
    }
}

DegData::DegData() {}
DegData::DegData(DataLoader & loader) {
    // input_layer and J^T
    at::Tensor & q = loader.q;
    q.set_requires_grad(true);
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    input_layer = Hd::input::input_layer(Redgeom);
    JT.resize(input_layer.size());
    for (size_t irred = 0; irred < input_layer.size(); irred++) {
        at::Tensor J_InpLay_q = at::empty(
            {input_layer[irred].size(0), q.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < input_layer[irred].size(0); i++) {
            torch::autograd::variable_list g = torch::autograd::grad({input_layer[irred][i]}, {q}, {}, true);
            J_InpLay_q[i].copy_(g[0]);
        }
        JT[irred] = (J_InpLay_q.mm(loader.J)).transpose(0, 1);
    }
    // Stop autograd
    for (at::Tensor & irred : input_layer) irred.detach_();
    // H and dH
    H = loader.energy; dH = loader.dH;
    CL::TS::chemistry::composite_representation(H, dH);
}
DegData::~DegData() {}

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

std::tuple<DataSet<RegData> *, DataSet<DegData> *> read_DataSet(
const std::vector<std::string> & data_set,
const double & zero_point, const double & weight) {
    // Count the number of data
    std::vector<size_t> NDataPerSet(data_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        NDataPerSet[its] = CL::utility::NLines(data_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // data set loader
    std::vector<RegData *> vec_p_RegData(NDataTotal);
    std::vector<DegData *> vec_p_DegData(NDataTotal);
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
        std::vector<DataLoader> RawDataLoader(NDataPerSet[its]);
        for (auto & loader : RawDataLoader) loader.init(NStates);
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
            // Modify raw data: cart2int and SubtractRef
            loader.cart2int();
            loader.SubtractRef(zero_point);
            // Insert to data set loader
            if (CL::TS::chemistry::check_degeneracy(DegThresh, loader.energy)) {
                vec_p_DegData[NDegData] = new DegData(loader);
                NDegData++;
            } else {
                vec_p_RegData[NRegData] = new RegData(loader);
                vec_p_RegData[NRegData]->adjust_weight(weight);
                NRegData++;
            }
        }
    }
    // Create DataSet with data set loader
    vec_p_RegData.resize(NRegData);
    vec_p_DegData.resize(NDegData);
    DataSet<RegData> * RegSet = new DataSet<RegData>(vec_p_RegData);
    DataSet<DegData> * DegSet = new DataSet<DegData>(vec_p_DegData);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace AbInitio