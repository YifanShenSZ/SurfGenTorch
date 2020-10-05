/*
To load and process ab initio data

The most basic ab initio data is geometry, who is the independent variable for any observable
The geometry itself can also feed unsupervised learning e.g. autoencoder

A common label is Hamiltonian (and gradient), who is classified into regular or degenerate based on degeneracy threshold
A regular Hamiltonian is in adiabatic representation, while a degenerate one is in composite representation
*/

#ifndef AbInitio_hpp
#define AbInitio_hpp

#include <torch/torch.h>

namespace AbInitio {

// Store Cartesian coordinate geometry
class GeomLoader { public:
    at::Tensor r;

    GeomLoader();
    ~GeomLoader();
};

// Store scaled and symmetry adapted internal coordinate geometry
class geom { public:
    std::vector<at::Tensor> SSAq;

    geom();
    geom(const GeomLoader & loader);
    ~geom();
};

// Store Hamiltonian and gradient data
class HamLoader : public GeomLoader { public:
    // Energy and Cartesian coordinate gradient
    at::Tensor energy, dH;

    HamLoader();
    HamLoader(const int64_t & NStates);
    ~HamLoader();

    void reset(const int64_t & NStates);
    // Subtract energy zero point
    void subtract_zero_point(const double & zero_point);
};

// Store regular Hamiltonian and gradient data in adiabatic representation
class RegHam : public geom { public:
    double weight = 1.0;
    // Jacobian^T of scaled and symmetry adapted internal coordinate w.r.t. Cartesian coordinate
    std::vector<at::Tensor> J_SSAq_r_T;
    // If not training dimensionality reduction,
    // observable net input layer and its Jacobian^T w.r.t. Cartesian coordinate
    // can be determined during data collection, before training
    std::vector<at::Tensor> input_layer, J_IL_r_T;
    // Energy and Cartesian coordinate gradient
    at::Tensor energy, dH;

    RegHam();
    RegHam(const HamLoader & loader, const bool & train_DimRed);
    ~RegHam();

    void adjust_weight(const double & Ethresh);
};

// Store degenerate Hamiltonian and gradient data in composite representation
class DegHam : public RegHam { public:
    // Composite representation Hamiltonian
    at::Tensor H;

    DegHam();
    DegHam(const HamLoader & loader, const bool & train_DimRed);
    ~DegHam();
};

template <class T> class DataSet : public torch::data::datasets::Dataset<DataSet<T>, T*> {
    private:
        std::vector<T*> example_;
    public:
        inline DataSet(const std::vector<T*> & example) {example_ = example;}
        inline ~DataSet() {for (T* & data : example_) delete [] data;}

        inline std::vector<T*> example() const {return example_;}

        // Override the size method to infer the size of the data set
        inline torch::optional<size_t> size() const override {return example_.size();}
        // Override the get method to load custom data
        inline T* get(size_t index) override {return example_[index];}

        inline size_t size_int() const {return example_.size();}        
};

DataSet<geom> * read_GeomSet(const std::vector<std::string> & data_set);

std::tuple<DataSet<RegHam> *, DataSet<DegHam> *> read_HamSet(
const std::vector<std::string> & data_set, const bool & train_DimRed,
const double & zero_point = 0.0, const double & weight = 1.0);

} // namespace AbInitio

#endif