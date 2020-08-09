#ifndef pretrain_hpp
#define pretrain_hpp

#include <torch/torch.h>

#include "AbInitio.hpp"
#include "net.hpp"

namespace DimRed {

void pretrain(const size_t & irred, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & data_set,
const std::vector<std::string> & chk, const size_t & chk_depth,
const std::string & opt = "TR", const size_t & epoch = 1000);

namespace FLopt {
    void initialize(const std::shared_ptr<Net> & net_,
        const size_t & irred_, const size_t & freeze_,
        const std::vector<AbInitio::geom *> & GeomSet_);
    void optimize(const std::string & opt, const size_t & epoch);
} // namespace FLopt

} // namespace DimRed

#endif