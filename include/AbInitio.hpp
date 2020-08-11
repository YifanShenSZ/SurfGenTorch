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

#include <torch/torch.h>

namespace AbInitio {

class GeomLoader { public:
    at::Tensor cartgeom, intgeom;

    GeomLoader();
    GeomLoader(const int & cartdim, const int & intdim);
    ~GeomLoader();

    void init(const int & cartdim, const int & intdim);
    void cart2int();
};

class geom { public:
    std::vector<at::Tensor> SAIgeom;

    geom();
    geom(const GeomLoader & loader);
    ~geom();
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

} // namespace AbInitio

#endif