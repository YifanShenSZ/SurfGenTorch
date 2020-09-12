/*
To load and process ab initio data

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
*/

#ifndef AbInitio_hpp
#define AbInitio_hpp

#include <torch/torch.h>

namespace AbInitio {

class GeomLoader { public:
    at::Tensor r, q;

    GeomLoader();
    ~GeomLoader();

    void cart2int();
};

class geom { public:
    std::vector<at::Tensor> SAIgeom;

    geom();
    geom(const GeomLoader & loader);
    ~geom();
};

class DataLoader : public GeomLoader { public:
    at::Tensor J, energy, dH;

    DataLoader();
    ~DataLoader();

    void init(const int64_t & NStates);
    void cart2int();
    void SubtractRef(const double & zero_point);
};

// Regular data
class RegData { public:
    double weight = 1.0;
    std::vector<at::Tensor> input_layer, JT;
    at::Tensor energy, dH;

    RegData();
    RegData(DataLoader & loader);
    ~RegData();

    void adjust_weight(const double & Ethresh);
};

// Degenerate data
class DegData { public:
    std::vector<at::Tensor> input_layer, JT;
    at::Tensor H, dH;

    DegData();
    DegData(DataLoader & loader);
    ~DegData();
};

template <class T> class DataSet : public torch::data::datasets::Dataset<DataSet<T>, T*> {
    private:
        std::vector<T*> example_;
    public:
        inline DataSet(const std::vector<T*> & example) {example_ = example;}
        inline ~DataSet() {}

        inline std::vector<T*> example() const {return example_;}

        // Override the size method to infer the size of the data set
        inline torch::optional<size_t> size() const override {return example_.size();}
        // Override the get method to load custom data
        inline T* get(size_t index) override {return example_[index];}

        inline size_t size_int() const {return example_.size();}        
};

DataSet<geom> * read_GeomSet(const std::vector<std::string> & data_set);

std::tuple<DataSet<RegData> *, DataSet<DegData> *> read_DataSet(
const std::vector<std::string> & data_set,
const double & zero_point = 0.0, const double & weight = 1.0);

} // namespace AbInitio

#endif