/*
To load and process ab initio data

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
*/

#include <torch/torch.h>
#include <FortranLibrary.hpp>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "Hd.hpp"
#include "AbInitio.hpp"

namespace AbInitio {

GeomLoader::GeomLoader() {}
GeomLoader::GeomLoader(const int & cartdim, const int & intdim) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->cartgeom = at::empty(cartdim, top);
    this->intgeom  = at::empty(intdim , top);
}
GeomLoader::~GeomLoader() {}
void GeomLoader::init(const int & cartdim, const int & intdim) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->cartgeom = at::empty(cartdim, top);
    this->intgeom  = at::empty(intdim , top);
}
void GeomLoader::cart2int() {
    FL::GT::InternalCoordinate(
        cartgeom.data_ptr<double>(), intgeom.data_ptr<double>(),
        cartgeom.size(0), intgeom.size(0));
}

geom::geom() {}
geom::geom(const GeomLoader & loader) {
    SAIgeom = SSAIC::compute_SSAIC(loader.intgeom);
}
geom::~geom() {}

DataLoader::DataLoader() {}
DataLoader::DataLoader(const int & cartdim, const int & intdim, const int & NStates)
: GeomLoader::GeomLoader(cartdim, intdim) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->BT     = at::empty({cartdim, intdim}, top);
    this->energy = at::empty(NStates, top);
    this->dH     = at::empty({NStates, NStates, cartdim}, top);
}
DataLoader::~DataLoader() {}
void DataLoader::init(const int & cartdim, const int & intdim, const int & NStates)  {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->cartgeom = at::empty(cartdim, top);
    this->intgeom  = at::empty(intdim , top);
    this->BT       = at::empty({cartdim, intdim}, top);
    this->energy   = at::empty(NStates, top);
    this->dH       = at::empty({NStates, NStates, cartdim}, top);
}
void DataLoader::cart2int() {
    FL::GT::WilsonBMatrixAndInternalCoordinate(
        cartgeom.data_ptr<double>(),
        BT.data_ptr<double>(), intgeom.data_ptr<double>(),
        cartgeom.size(0), intgeom.size(0));
}
void DataLoader::SubtractRef(const double & zero_point) {
    energy -= zero_point;
}

// Regular data
RegData::RegData() {}
RegData::RegData(DataLoader & loader) {
    loader.intgeom.set_requires_grad(true);
    // input_layer and J^T
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(loader.intgeom);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> InpLay = Hd::input::input_layer(Redgeom);
    input_layer.resize(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) {
        input_layer[irred] = InpLay[irred].detach();
        input_layer[irred].set_requires_grad(true);
    }
    JT.resize(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) {
        at::Tensor dinput_layer_divide_dintgeom = at::empty(
            {InpLay[irred].size(0), loader.intgeom.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < InpLay[irred].size(0); i++) {
            if (loader.intgeom.grad().defined()) {
                loader.intgeom.grad().detach_();
                loader.intgeom.grad().zero_();
            };
            InpLay[irred][i].backward({}, true);
            dinput_layer_divide_dintgeom[i].copy_(loader.intgeom.grad());
        }
        JT[irred] = loader.BT.mm(dinput_layer_divide_dintgeom.transpose(0, 1));
    }
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
    loader.intgeom.set_requires_grad(true);
    // input_layer and J^T
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(loader.intgeom);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> InpLay = Hd::input::input_layer(Redgeom);
    input_layer.resize(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) {
        input_layer[irred] = InpLay[irred].detach();
        input_layer[irred].set_requires_grad(true);
    }
    JT.resize(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) {
        at::Tensor dinput_layer_divide_dintgeom = at::empty(
            {InpLay[irred].size(0), loader.intgeom.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < InpLay[irred].size(0); i++) {
            if (loader.intgeom.grad().defined()) {
                loader.intgeom.grad().detach_();
                loader.intgeom.grad().zero_();
            };
            InpLay[irred][i].backward({}, true);
            dinput_layer_divide_dintgeom[i].copy_(loader.intgeom.grad());
        }
        JT[irred] = loader.BT.mm(dinput_layer_divide_dintgeom.transpose(0, 1));
    }
    // H and dH
    // Diagonalize ▽H . ▽H
    at::Tensor dHdH = CL::TS::LA::sy3matdotmul(loader.dH, loader.dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true, true);
    dHdH = eigvec.transpose(0, 1);
    // Transform H and dH
    H = dHdH.mm(loader.energy.diag().mm(eigvec));
    dH = CL::TS::LA::UT_A3_U(dHdH, loader.dH, eigvec);
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
        GeomLoader * RawGeomLoader = new GeomLoader[NDataPerSet[its]];
        for (size_t i = 0; i < NDataPerSet[its]; i++) RawGeomLoader[i].init(SSAIC::cartdim, SSAIC::intdim);
        // cartgeom
        std::ifstream ifs; ifs.open(data_set[its]+"geom.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < SSAIC::cartdim / 3; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom[3*j  ] = dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom[3*j+1] = dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom[3*j+2] = dbletemp;
            }
        ifs.close();
        // Process raw data
        for (size_t i = 0; i < NDataPerSet[its]; i++) {
            // Modify raw data
            RawGeomLoader[i].cart2int();
            // Insert to data set loader
            vec_p_geom[count] = new geom(RawGeomLoader[i]);
            count++;
        }
        // Clean up
        delete [] RawGeomLoader;
    }
    // Create DataSet with data set loader
    GeomSet = new DataSet<geom>(vec_p_geom);
    return GeomSet;
}

std::tuple<DataSet<RegData> *, DataSet<DegData> *> read_DataSet(
const std::vector<std::string> & data_set, const double & zero_point) {
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
        int NStates = strs.size();
        std::ifstream ifs_valid_states; ifs_valid_states.open(data_set[its]+"valid_states");
            if (ifs_valid_states.good()) {
                int valid_states;
                ifs_valid_states >> valid_states;
                NStates = valid_states < NStates ? valid_states: NStates;
            }
        ifs_valid_states.close();
        // raw data loader
        DataLoader * RawDataLoader = new DataLoader[NDataPerSet[its]];
        for (size_t i = 0; i < NDataPerSet[its]; i++) RawDataLoader[i].init(SSAIC::cartdim, SSAIC::intdim, NStates);
        // cartgeom
        ifs.open(data_set[its]+"geom.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < SSAIC::cartdim / 3; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom[3*j  ] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom[3*j+1] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom[3*j+2] = dbletemp;
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
        for (size_t i = 0; i < NDataPerSet[its]; i++) {
            // Modify raw data: cart2int and SubtractRef
            RawDataLoader[i].cart2int();
            RawDataLoader[i].SubtractRef(zero_point);
            // Insert to data set loader
            if (CL::TS::chemistry::check_degeneracy(DegThresh, RawDataLoader[i].energy)) {
                vec_p_DegData[NDegData] = new DegData(RawDataLoader[i]);
                NDegData++;
            } else {
                vec_p_RegData[NRegData] = new RegData(RawDataLoader[i]);
                NRegData++;
            }
        }
        // Clean up
        delete [] RawDataLoader;
    }
    // Create DataSet with data set loader
    vec_p_RegData.resize(NRegData);
    vec_p_DegData.resize(NDegData);
    DataSet<RegData> * RegSet = new DataSet<RegData>(vec_p_RegData);
    DataSet<DegData> * DegSet = new DataSet<DegData>(vec_p_DegData);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace AbInitio