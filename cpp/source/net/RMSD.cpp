// Evaluate the root mean square deviation

#include "../../include/SSAIC.hpp"
#include "../../include/net.hpp"

namespace DimRed {
    double RMSD(const size_t & irred, const std::shared_ptr<Net> & net, const std::vector<AbInitio::geom*> & geom_set) {
        double e = 0.0;
        torch::NoGradGuard no_grad;
        for (auto & geom : geom_set) {
            e += torch::mse_loss(net->forward(geom->SAIgeom[irred]), geom->SAIgeom[irred],
                 at::Reduction::Sum).item<double>();
        }
        e /= (double)geom_set.size();
        e /= (double)std::accumulate(SSAIC::NSAIC_per_irred.begin(), SSAIC::NSAIC_per_irred.end(), 0);;
        e = std::sqrt(e);
        return e;
    }
} // namespace DimRed