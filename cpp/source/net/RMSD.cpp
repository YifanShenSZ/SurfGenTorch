// Evaluate the root mean square deviation

#include "../../include/SSAIC.hpp"
#include "../../include/pretrain.hpp"

namespace DimRed {
    double RMSD(const size_t & irred, const std::shared_ptr<Net> & net, const std::vector<AbInitio::geom*> & GeomSet) {
        double e = 0.0;
        torch::NoGradGuard no_grad;
        for (auto & geom : GeomSet) {
            e += torch::mse_loss(net->forward(geom->SAIgeom[irred]), geom->SAIgeom[irred],
                 at::Reduction::Sum).item<double>();
        }
        e /= (double)GeomSet.size();
        e /= (double)SSAIC::NSAIC_per_irred[irred];
        return std::sqrt(e);
    }
} // namespace DimRed