/*
To load and process ab initio data

The prerequisite of any fit is data
Specifically, for Hd construction data comes from ab initio

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
*/

#include <torch/torch.h>
#include <FortranLibrary.hpp>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/LinearAlgebra.hpp>
#include <CppLibrary/chemistry.hpp>

#include "SSAIC.hpp"
#include "AbInitio.hpp"

namespace AbInitio {

GeomLoader::GeomLoader() {}
GeomLoader::GeomLoader(const int & cartdim, const int & intdim) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->cartgeom = at::empty(cartdim, top);
    this->intgeom  = at::empty(intdim,  top);
}
GeomLoader::~GeomLoader() {}

void GeomLoader::init(const int & cartdim, const int & intdim) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    this->cartgeom = at::empty(cartdim, top);
    this->intgeom  = at::empty(intdim,  top);
}
void GeomLoader::cart2int() {
    FL::GT::InternalCoordinate(
    cartgeom.data_ptr<double>(), intgeom.data_ptr<double>(),
    SSAIC::cartdim, SSAIC::intdim);
}

geom::geom() {}
geom::geom(const GeomLoader & loader) {
    SAIgeom = SSAIC::compute_SSAIC(loader.intgeom);
}
geom::~geom() {}

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

} // namespace AbInitio