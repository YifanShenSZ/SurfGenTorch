/*
To load and process ab initio data

The prerequisite of any fit is data
Specifically, for Hd construction data comes from ab initio

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
*/

#ifndef AbInitio_hpp
#define AbInitio_hpp

#include <type_traits>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../Cpp-Library_v1.0.0/chemistry.hpp"
#include "../Cpp-Library_v1.0.0/torch_LinearAlgebra.hpp"

#define DegThresh 0.0001

namespace AbInitio {

// ab initio data point management
// this is actually float and double specialization
template <class T> class GeomLoader { public:
    int cartdim_, intdim_;
    at::Tensor cartgeom_, intgeom_, BT_;

    inline GeomLoader() {}
    inline GeomLoader(const int & cartdim, const int & intdim) {
        cartdim_ = cartdim;
        intdim_  = intdim ;
        if (std::is_same<T, double>::value) {
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            cartgeom_ = at::empty(cartdim, top);
            intgeom_ = at::empty(intdim, top);
            BT_ = at::empty({cartdim, intdim}, top);
        } else {
            cartgeom_ = at::empty(cartdim);
            intgeom_ = at::empty(intdim);
            BT_ = at::empty({cartdim, intdim});
        }
    }
    inline ~GeomLoader() {}

    void init(const int & cartdim, const int & intdim) {
        cartdim_ = cartdim;
        intdim_  = intdim ;
        if (std::is_same<T, double>::value) {
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            cartgeom_ = at::empty(cartdim, top);
            intgeom_ = at::empty(intdim, top);
            BT_ = at::empty({cartdim, intdim}, top);
        } else {
            cartgeom_ = at::empty(cartdim);
            intgeom_ = at::empty(intdim);
            BT_ = at::empty({cartdim, intdim});
        }
    }
    void cart2int() {
        FL::GeometryTransformation::WilsonBMatrixAndInternalCoordinate(
        cartgeom_.data_ptr<T>(), BT_.data_ptr<T>(), intgeom_.data_ptr<T>(),
        cartdim_, intdim_);
    }
};
template <class T> class geom : protected GeomLoader<T> {
    public:
        inline geom(const GeomLoader<double> & loader)
        : GeomLoader<T>(loader.cartdim_, loader.intdim_) {
            GeomLoader<T>::cartgeom_.copy_(loader.cartgeom_);
            GeomLoader<T>::intgeom_.copy_(loader.intgeom_);
            GeomLoader<T>::BT_.copy_(loader.BT_);
        }
        inline ~geom() {}

        inline int cartdim() const {return GeomLoader<T>::cartdim_;}
        inline int intdim() const {return GeomLoader<T>::intdim_;}
        inline at::Tensor cartgeom() const {return GeomLoader<T>::cartgeom_;}
        inline at::Tensor intgeom() const {return GeomLoader<T>::intgeom_;}
        inline at::Tensor BT() const {return GeomLoader<T>::BT_;}
};
template <class T> class DataLoader : public GeomLoader<T> { public:
    int NStates_;
    at::Tensor energy_, dH_;

    inline DataLoader() {}
    inline DataLoader(const int & cartdim, const int & intdim, const int & NStates)
    : GeomLoader<T>(cartdim, intdim) {
        NStates_ = NStates;
        if (std::is_same<T, double>::value) {
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            energy_ = at::empty(NStates, top);
            dH_ = at::empty({NStates, NStates, cartdim}, top);
        } else {
            energy_ = at::empty(NStates);
            dH_ = at::empty({NStates, NStates, cartdim});
        }
    }
    inline ~DataLoader() {}

    void init(const int & cartdim, const int & intdim, const int & NStates) {
        GeomLoader<T>::cartdim_ = cartdim;
        GeomLoader<T>::intdim_  = intdim ;
        NStates_ = NStates;
        if (std::is_same<T, double>::value) {
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            GeomLoader<T>::cartgeom_ = at::empty(cartdim, top);
            GeomLoader<T>::intgeom_ = at::empty(intdim, top);
            GeomLoader<T>::BT_ = at::empty({cartdim, intdim}, top);
            energy_ = at::empty(NStates, top);
            dH_ = at::empty({NStates, NStates, cartdim}, top);
        } else {
            GeomLoader<T>::cartgeom_ = at::empty(cartdim);
            GeomLoader<T>::intgeom_ = at::empty(intdim);
            GeomLoader<T>::BT_ = at::empty({cartdim, intdim});
            energy_ = at::empty(NStates);
            dH_ = at::empty({NStates, NStates, cartdim});
        }
    }
    template <typename Te> void SubtractRef(const at::Tensor & origin, const Te & zero_point) {
        GeomLoader<T>::intgeom_ -= origin;
        energy_  -= (T)zero_point;
    }
};
template <class T> class RegularData : protected DataLoader<T> {
    protected:
        T weight_ = 1.0;
    public:
        RegularData(const DataLoader<double> & loader)
        : DataLoader<T>(loader.cartdim_, loader.intdim_, loader.NStates_) {
            DataLoader<T>::cartgeom_.copy_(loader.cartgeom_);
            DataLoader<T>::intgeom_.copy_(loader.intgeom_);
            DataLoader<T>::BT_.copy_(loader.BT_);
            DataLoader<T>::energy_.copy_(loader.energy_);
            DataLoader<T>::dH_.copy_(loader.dH_);
        }
        inline ~RegularData() {}

        inline T weight() const {return weight_;}
        inline int cartdim() const {return DataLoader<T>::cartdim_;}
        inline int intdim() const {return DataLoader<T>::intdim_;}
        inline int NStates() const {return DataLoader<T>::NStates_;}
        inline at::Tensor cartgeom() const {return DataLoader<T>::cartgeom_;}
        inline at::Tensor intgeom() const {return DataLoader<T>::intgeom_;}
        inline at::Tensor BT() const {return DataLoader<T>::BT_;}
        inline at::Tensor energy() const {return DataLoader<T>::energy_;}
        inline at::Tensor dH() const {return DataLoader<T>::dH_;}

        template <typename Te> void AdjustWeight(const Te & Ethresh) {
            T temp = energy_[0].item<T>();
            if ( temp > Ethresh) {
                temp = Ethresh / temp;
                weight_ = temp * temp;
            }
        }
};
template <class T> class DegenerateData : public RegularData<T> {
    protected:
        at::Tensor H_;
    public:
        DegenerateData(const DataLoader<double> & loader)
        : RegularData<T>(loader) {
            at::Tensor eigval, eigvec;
            if (std::is_same<T, double>::value) {
                auto top = at::TensorOptions().dtype(torch::kFloat64);
                H_ = at::empty({loader.NStates_, loader.NStates_}, top);
                eigval = at::empty(loader.NStates_, top);
                eigvec = at::empty({loader.NStates_, loader.NStates_}, top);
            } else {
                H_ = at::empty({loader.NStates_, loader.NStates_});
                eigval = at::empty(loader.NStates_);
                eigvec = at::empty({loader.NStates_, loader.NStates_});
            }
            // Diagonalize ▽H . ▽H
            at::Tensor dHdH = torch_LinearAlgebra::matdotmul(DataLoader<T>::dH_, DataLoader<T>::dH_);
            std::tie(eigval, eigvec) = dHdH.symeig(true, true);
            dHdH = eigvec.transpose(0, 1);
            // Transform H and dH
            H_ = dHdH.mm(DataLoader<T>::energy_.diag().mm(eigvec));
            torch_LinearAlgebra::UT_A3_U(dHdH, DataLoader<T>::dH_, eigvec);
        }
        inline ~DegenerateData() {}

        inline at::Tensor H() const {return H_;}
};

template <class T> class DataSet : public torch::data::datasets::Dataset<DataSet<T>, T*> {
    private:
        std::vector<T*> example_;
    public:
        // Override the size method to infer the size of the data set
        inline torch::optional<size_t> size() const override {return example_.size();}
        // Override the get method to load custom data
        inline T* get(size_t index) override {return example_[index];}

        inline DataSet(const std::vector<T*> & example) {example_ = example;}
        inline ~DataSet() {}

        inline size_t size_int() const {return example_.size();}
        //inline std::vector<T*> example() const {return example_;}
};

template <typename T> void read_GeomSet(
const std::vector<std::string> & data_set,
const at::Tensor & origin, const int & intdim,
DataSet<geom<T>> * & GeomSet) {
    // Count the number of data
    std::vector<size_t> NDataPerSet(data_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        NDataPerSet[its] = general::NLines(data_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // data set loader
    std::vector<geom<T>*> vec_p_geom(NDataTotal);
    // a piece of data constructor: cartdim
    int cartdim = general::NStrings(data_set[0]+"cartgrad-1.data") / NDataPerSet[0];
    // Read training set files
    int count = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        // raw data loader
        GeomLoader<double> * RawGeomLoader = new GeomLoader<double>[NDataPerSet[its]];
        for (size_t i = 0; i < NDataPerSet[its]; i++) RawGeomLoader[i].init(cartdim, intdim);
        // cartgeom
        std::ifstream ifs;
        ifs.open(data_set[its]+"geom.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < cartdim / 3; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom_[3*j  ] = dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom_[3*j+1] = dbletemp;
                ifs >> dbletemp; RawGeomLoader[i].cartgeom_[3*j+2] = dbletemp;
            }
        ifs.close();
        // Process raw data
        for (size_t i = 0; i < NDataPerSet[its]; i++) {
            // Modify raw data: cart2int and SubtractRef
            RawGeomLoader[i].cart2int();
            RawGeomLoader[i].intgeom_ -= origin;
            // Insert to data set loader
            vec_p_geom[count] = new geom<T>(RawGeomLoader[i]);
            count++;
        }
        // Clean up
        delete [] RawGeomLoader;
    }
    // Create DataSet with data set loader
    GeomSet = new DataSet<geom<T>>(vec_p_geom);
}

template <typename T> void read_DataSet(
const std::vector<std::string> & data_set, const std::vector<size_t> & valid_states,
const at::Tensor & origin, const double & zero_point, const int & intdim,
DataSet<RegularData<T>> * & RegularDataSet, DataSet<DegenerateData<T>> * & DegenerateDataSet) {
    // Count the number of data
    std::vector<size_t> NDataPerSet(data_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < data_set.size(); its++) {
        NDataPerSet[its] = general::NLines(data_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // data set loader
    std::vector<RegularData<T>*> vec_p_RegularData(NDataTotal);
    std::vector<DegenerateData<T>*> vec_p_DegenerateData(NDataTotal);
    size_t NRegularData = 0;
    size_t NDegenerateData = 0;
    // a piece of data constructor: cartdim
    int cartdim = general::NStrings(data_set[0]+"cartgrad-1.data") / NDataPerSet[0];
    // Read training set files
    for (size_t its = 0; its < data_set.size(); its++) {
        // for file input
        std::ifstream ifs;
        std::string line;
        std::vector<std::string> line_split;
        // NStates
        ifs.open(data_set[its]+"energy.data"); std::getline(ifs, line); ifs.close();
        general::split(line, line_split);
        int NStates = line_split.size();
        if (NStates < valid_states[its]) std::cout
        << "Warning: training set " << its << " contains " << NStates << " electronic states\n"
        << "         but " << valid_states[its] << " were requested\n";
        NStates = NStates < valid_states[its] ? NStates : valid_states[its];
        // raw data loader
        DataLoader<double> * RawDataLoader = new DataLoader<double>[NDataPerSet[its]];
        for (size_t i = 0; i < NDataPerSet[its]; i++) RawDataLoader[i].init(cartdim, intdim, NStates);
        // cartgeom
        ifs.open(data_set[its]+"geom.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < cartdim / 3; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j  ] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j+1] = dbletemp;
                ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j+2] = dbletemp;
            }
        ifs.close();
        // energy
        ifs.open(data_set[its]+"energy.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++)
            for (size_t j = 0; j < NStates; j++) {
                double dbletemp; ifs >> dbletemp;
                RawDataLoader[i].energy_[j] = dbletemp;
            }
        ifs.close();
        // dH
        for (size_t istate = 0; istate < NStates; istate++) {
            ifs.open(data_set[its]+"cartgrad-"+std::to_string(istate+1)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++)
                for (size_t j = 0; j < cartdim; j++) {
                    double dbletemp; ifs >> dbletemp;
                    RawDataLoader[i].dH_[istate][istate][j] = dbletemp;
                }
            ifs.close();
        for (size_t jstate = istate; jstate < NStates; jstate++) {
            ifs.open(data_set[its]+"cartgrad-"+std::to_string(istate+1)+"-"+std::to_string(jstate+1)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++)
                for (size_t j = 0; j < cartdim; j++) {
                    double dbletemp; ifs >> dbletemp;
                    RawDataLoader[i].dH_[istate][jstate][j] = dbletemp;
                }
            ifs.close();
        } }
        // Process raw data
        for (size_t i = 0; i < NDataPerSet[its]; i++) {
            // Modify raw data: cart2int and SubtractRef
            RawDataLoader[i].cart2int();
            RawDataLoader[i].SubtractRef(origin, zero_point);
            // Insert to data set loader
            if (chemistry::CheckDegeneracy(DegThresh, RawDataLoader[i].energy_.data_ptr<double>(), RawDataLoader[i].NStates_)) {
                vec_p_DegenerateData[NDegenerateData] = new DegenerateData<T>(RawDataLoader[i]);
                NDegenerateData++;
            } else {
                vec_p_RegularData[NRegularData] = new RegularData<T>(RawDataLoader[i]);
                NRegularData++;
            }
        }
        // Clean up
        delete [] RawDataLoader;
    }
    // Create DataSet with data set loader
    vec_p_RegularData.resize(NRegularData);
    vec_p_DegenerateData.resize(NDegenerateData);
    RegularDataSet = new DataSet<RegularData<T>>(vec_p_RegularData);
    DegenerateDataSet = new DataSet<DegenerateData<T>>(vec_p_DegenerateData);
}

} // namespace AbInitio

#endif