#ifndef AbInitio_hpp
#define AbInitio_hpp

#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../Cpp-Library_v1.0.0/torch_LinearAlgebra.hpp"

#define QuasiDegThresh 0.0001

// If tensor creation is involved in a template,
// then it is somehow a specialization for float,
// because libtorch cannot specify tensor data type with c++ type name
// and I just create default data type (float) tensor
namespace AbInitio {

template <class T> class DataLoader { public:
    int cartdim_, intdim_, NStates_;
    at::Tensor cartgeom_, intgeom_, BT_;
    at::Tensor energy_, dH_;

    inline DataLoader(const int & cartdim, const int & intdim, const int & NStates) {
        cartdim_ = cartdim;
        intdim_  = intdim ;
        NStates_ = NStates;
        cartgeom_ = at::empty(cartdim);
        intgeom_ = at::empty(intdim);
        BT_ = at::empty({cartdim, intdim});
        energy_ = at::empty(NStates);
        dH_ = at::empty({NStates, NStates, cartdim});
    }
    inline ~DataLoader() {}

    void cart2int() {
        FL::GeometryTransformation::WilsonBMatrixAndInternalCoordinate(
            cartgeom_.data_ptr<float>(), BT_.data_ptr<float>(), intgeom_.data_ptr<float>(),
            cartdim_, intdim_
        );
    }
    template <typename T0> void SubtractRef(const at::Tensor & origin, const T0 & zero_point) {
        intgeom_ -= origin;
        energy_  -= (T)zero_point;
    }
};

template <> class DataLoader<double> { public:
    int cartdim_, intdim_, NStates_;
    at::Tensor cartgeom_, intgeom_, BT_;
    at::Tensor energy_, dH_;

    inline DataLoader(const int & cartdim, const int & intdim, const int & NStates) {
        cartdim_ = cartdim;
        intdim_  = intdim ;
        NStates_ = NStates;
        auto top = at::TensorOptions().dtype(torch::kFloat64);
        cartgeom_ = at::empty(cartdim, top);
        intgeom_ = at::empty(intdim, top);
        BT_ = at::empty({cartdim, intdim}, top);
        energy_ = at::empty(NStates, top);
        dH_ = at::empty({NStates, NStates, cartdim}, top);
    }
    inline ~DataLoader() {}

    void cart2int() {
        FL::GeometryTransformation::WilsonBMatrixAndInternalCoordinate(
            cartgeom_.data_ptr<double>(), BT_.data_ptr<double>(), intgeom_.data_ptr<double>(),
            cartdim_, intdim_
        );
    }
    template <typename T0> void SubtractRef(const at::Tensor & origin, const T0 & zero_point) {
        intgeom_ -= origin;
        energy_  -= (double)zero_point;
    }
};

template <class T> class RegularData : protected DataLoader<T> {
    protected:
        T weight_ = 1.0;
    public:
        RegularData(DataLoader<double> loader)
        : DataLoader<T>(loader.cartdim_, loader.intdim_, loader.NStates_) {
            DataLoader<T>::cartgeom_.copy_(loader.cartgeom_);
            DataLoader<T>::intgeom_.copy_(loader.intgeom_);
            DataLoader<T>::BT_.copy_(loader.BT_);
            DataLoader<T>::energy_.copy_(loader.energy_);
            DataLoader<T>::dH_.copy_(loader.dH_);
        }
        inline ~RegularData() {}

        inline T weight() const {return weight_;}
        inline int cartdim() const {return cartdim_;}
        inline int intdim() const {return intdim_;}
        inline int NStates() const {return NStates_;}
        inline at::Tensor cartgeom() const {return cartgeom_;}
        inline at::Tensor intgeom() const {return intgeom_;}
        inline at::Tensor BT() const {return BT_;}
        inline at::Tensor energy() const {return energy_;}
        inline at::Tensor dH() const {return dH_;}

        template <typename Te> void AdjustWeight(const Te & Ethresh) {
            T temp = energy_[0].item<T>();
            if ( temp > Ethresh) {
                temp = Ethresh / temp;
                weight_ = temp * temp;
            }
        }
};

template <> class RegularData<double> : protected DataLoader<double> {
    protected:
        double weight_ = 1.0;
    public:
        RegularData(DataLoader<double> loader)
        : DataLoader<double>(loader.cartdim_, loader.intdim_, loader.NStates_) {
            DataLoader<double>::cartgeom_.copy_(loader.cartgeom_);
            DataLoader<double>::intgeom_.copy_(loader.intgeom_);
            DataLoader<double>::BT_.copy_(loader.BT_);
            DataLoader<double>::energy_.copy_(loader.energy_);
            DataLoader<double>::dH_.copy_(loader.dH_);
        }
        inline ~RegularData() {}

        inline double weight() const {return weight_;}
        inline int cartdim() const {return cartdim_;}
        inline int intdim() const {return intdim_;}
        inline int NStates() const {return NStates_;}
        inline at::Tensor cartgeom() const {return cartgeom_;}
        inline at::Tensor intgeom() const {return intgeom_;}
        inline at::Tensor BT() const {return BT_;}
        inline at::Tensor energy() const {return energy_;}
        inline at::Tensor dH() const {return dH_;}

        template <typename Te> void AdjustWeight(const Te & Ethresh) {
            double temp = energy_[0].item<double>();
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
        DegenerateData(DataLoader<double> loader) : RegularData<T>(loader) {
            H_ = at::empty({loader.NStates_, loader.NStates_});
        }
        inline ~DegenerateData() {}

        inline at::Tensor H() const {return H_;}

        void CompositeRepresentation() {
            // Diagonalize ▽H . ▽H
            at::Tensor eigval = at::empty(DataLoader<T>::NStates_);
            at::Tensor eigvec = at::empty({DataLoader<T>::NStates_, DataLoader<T>::NStates_});
            at::Tensor dHdH = torch_LinearAlgebra::matdotmul(DataLoader<T>::dH_, DataLoader<T>::dH_);
            std::tie(eigval, eigvec) = dHdH.symeig(true, true);
            dHdH = eigvec.transpose(0, 1);
            // Transform H and dH
            H_ = dHdH.mm(DataLoader<T>::energy_.diag().mm(eigvec));
            torch_LinearAlgebra::UT_A3_U(dHdH, DataLoader<T>::dH_, eigvec);
        }
};

template <> class DegenerateData<double> : public RegularData<double> {
    protected:
        at::Tensor H_;
    public:
        DegenerateData(DataLoader<double> loader) : RegularData<double>(loader) {
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            H_ = at::empty({loader.NStates_, loader.NStates_}, top);
        }
        inline ~DegenerateData() {}

        inline at::Tensor H() const {return H_;}

        void CompositeRepresentation() {
            // Diagonalize ▽H . ▽H
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            at::Tensor eigval = at::empty(NStates_, top);
            at::Tensor eigvec = at::empty({NStates_, NStates_}, top);
            at::Tensor dHdH = at::empty({NStates_, NStates_}, top);
            torch_LinearAlgebra::matdotmul(dH_, dH_, dHdH);
            std::tie(eigval, eigvec) = dHdH.symeig(true, true);
            dHdH = eigvec.transpose(0, 1);
            // Transform H and dH
            H_ = dHdH.mm(energy_.diag().mm(eigvec));
            torch_LinearAlgebra::UT_A3_U(dHdH, dH_, eigvec);
        }
};

template <class T> class DataSet : public torch::data::Dataset<DataSet<T>> {
    private:
        size_t size_;
        std::vector<T> example_;
    public:
        inline DataSet(std::vector<T> example) {
            size_    = example.size();
            example_ = example;
        }
        inline DataSet(const size_t & size, const T ** loader) {
            size_ = size;
            example_.resize(size);
            for (size_t i = 0; i < size; i++)  example_[i] = * loader[i];
        }
        inline ~DataSet() {}

        // Override the size method to infer the size of the RegularData set.
        inline torch::optional<size_t> size() const override {return size_;}
        inline std::vector<T> example() const {return example_;}

        // Override the get method to load custom RegularData.
        inline torch::data::Example<> get(size_t index) override {return example_[index]};
};

template <typename T> void InitializeTrainingSet(
const std::string & format, const std::vector<std::string> & training_set, const std::vector<size_t> & valid_states,
const at::Tensor & origin, const double & zero_point, const int & intdim,
DataSet<RegularData<T>> * & RegularDataSet, DataSet<DegenerateData<T>> * & DegenerateDataSet
) {
    // Count the number of data
    std::vector<size_t> NDataPerSet(training_set.size());
    size_t NDataTotal = 0;
    for (size_t its = 0; its < training_set.size(); its++) {
        NDataPerSet[its] = general::NLines(training_set[its]+"energy.data");
        NDataTotal += NDataPerSet[its];
    }
    // Data set loader
    RegularData<T> ** pp_RegularData = new RegularData<T> * [NDataTotal];
    DegenerateData<T> ** pp_DegenerateData = new DegenerateData<T> * [NDataTotal];
    size_t NRegularData = 0;
    size_t NDegenerateData = 0;
    // A piece of data constructor: cartdim
    int cartdim = general::NStrings(training_set[0]+"cartgrad-1.data") / NDataPerSet[0];
    // Read training set files
    for (size_t its = 0; its < training_set.size(); its++) {
        std::ifstream ifs;
        std::string line;
        std::vector<std::string> line_split;
        // NStates
        ifs.open(training_set[its]+"energy.data"); std::getline(ifs, line); ifs.close();
        general::split(line, line_split);
        int NStates = line_split.size();
        if (NStates < valid_states[its]) std::cout
        << "Warning: training set " << its << " contains " << NStates << " electronic states\n"
        << "         but " << valid_states[its] << " were requested\n";
        NStates = NStates < valid_states[its] ? NStates : valid_states[its];
        // Raw data loader
        std::vector<DataLoader<double>> RawDataLoader(NDataPerSet[its], DataLoader<double>(cartdim, intdim, NStates));
        // cartgeom
        if (format == "Columbus7") {
            ifs.open(training_set[its]+"geom.data");
                for (size_t i = 0; i < NDataPerSet[its]; i++) {
                    for (size_t j = 0; j < cartdim / 3; j++) {
                        double dbletemp;
                        ifs >> line;
                        ifs >> dbletemp;
                        ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j  ] = dbletemp;
                        ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j+1] = dbletemp;
                        ifs >> dbletemp; RawDataLoader[i].cartgeom_[3*j+2] = dbletemp;
                    }
                }
            ifs.close();
        } else {throw std::out_of_range("TO BE IMPLEMENTED");}
        // energy
        ifs.open(training_set[its]+"energy.data");
            for (size_t i = 0; i < NDataPerSet[its]; i++) {
                std::getline(ifs, line);
                general::split(line, line_split);
                for (size_t j = 0; j < NStates; j++) RawDataLoader[i].energy_[j] = std::stod(line_split[j]);
            }
        ifs.close();
        // dH
        for (size_t istate = 0; istate < NStates; istate++) {
            ifs.open(training_set[its]+"cartgrad-"+std::to_string(istate)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++) {
                    for (size_t j = 0; j < cartdim; j++) {
                        double dbletemp;
                        ifs >> dbletemp;
                        RawDataLoader[i].dH_[istate][istate][j] = dbletemp;
                    }
                }
            ifs.close();
        for (size_t jstate = istate; jstate < NStates; jstate++) {
            ifs.open(training_set[its]+"cartgrad-"+std::to_string(istate)+"-"+std::to_string(jstate)+".data");
                for (size_t i = 0; i < NDataPerSet[its]; i++) {
                    for (size_t j = 0; j < cartdim; j++) {
                        double dbletemp;
                        ifs >> dbletemp;
                        RawDataLoader[i].dH_[istate][jstate][j] = dbletemp;
                    }
                }
            ifs.close();
        } }
        // Process raw data
        for (size_t i = 0; i < NDataPerSet[its]; i++) {
            // Modify raw data: cart2int and SubtractRef
            RawDataLoader[i].cart2int();
            RawDataLoader[i].SubtractRef(origin, zero_point);
            // Insert to data set loader
            if (chemistry::CheckDegeneracy(QuasiDegThresh, RawDataLoader[i].energy_.data_ptr<double>(), RawDataLoader[i].NStates_)) {
                pp_DegenerateData[NDegenerateData] = new DegenerateData<T>(RawDataLoader[i]);
                pp_DegenerateData[NDegenerateData]->CompositeRepresentation();
                NDegenerateData++;
            } else {
                pp_RegularData[NRegularData] = new RegularData<T>(RawDataLoader[i]);
                NRegularData++;
            }
        }
    }
    // Create DataSet with data set loader
    RegularDataSet = new DataSet<RegularData<T>>(NRegularData, pp_RegularData);
    DegenerateDataSet = new DataSet<DegenerateData<T>>(NDegenerateData, pp_DegenerateData);
    // Clean up
    delete [] pp_RegularData;
    delete [] pp_DegenerateData;
}

} // namespace AbInitio

#endif