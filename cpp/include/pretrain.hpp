#ifndef pretrain_hpp
#define pretrain_hpp

#include <torch/torch.h>

#include "AbInitio.hpp"
#include "net.hpp"

namespace DimRed {

double RMSD(const size_t & irred, const std::shared_ptr<Net> & net, const std::vector<AbInitio::geom *> & GeomSet);

void pretrain(const size_t & irred, const size_t & max_depth,
const std::vector<std::string> & data_set,
const std::vector<std::string> & chk, const size_t & chk_depth,
const std::string & opt, const size_t & epoch);

namespace FLopt {
    void initialize(const size_t & irred_, const std::shared_ptr<Net> & net_, const std::vector<AbInitio::geom *> & GeomSet_);
    void optimize(const std::string & opt);
} // namespace FLopt

} // namespace DimRed

#endif