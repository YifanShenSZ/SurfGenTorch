/*
Scaled and symmetry adapted internal coordinate (SSAIC)

The procedure of this module:
    1. Define internal coordinates
    2. Nondimensionalize the internal coordinates:
       for length, dic = (ic - origin) / origin
       for angle , dic =  ic - origin
    3. Scale the dimensionless internal coordinates:
       if no scaler      : sic = nic
       elif scaler = self: sic = pi * erf(nic)
       else              : sic = nic * exp(-alpha * scaler nic)
    4. Symmetry adapted linear combinate the scaled dimensionless internal coordinates
*/

#include <regex>
#include <torch/torch.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/LinearAlgebra.hpp>
#include <CppLibrary/chemistry.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"

namespace SSAIC {

// Symmetry adapted linear combination
struct SymmAdLinComb {
    std::vector<double> coeff;
    std::vector<size_t> IntCoord;

    SymmAdLinComb() {}
    ~SymmAdLinComb() {}
};

// The rule of internal coordinates who are scaled by others
// self is scaled by scaler with alpha
struct OthScalRul {
    size_t self, scaler;
    double alpha;

    OthScalRul() {}
    OthScalRul(const std::vector<std::string> & input_line) {
        self   = std::stoul(input_line[0]) - 1;
        scaler = std::stoul(input_line[1]) - 1;
        alpha  = std::stod (input_line[2]);
    }
    ~OthScalRul() {}
};

// Internal coordinate dimension, not necessarily = cartdim - 6 or 5
int intdim;
// Fortran-Library internal coordinate definition
std::vector<FL::GT::IntCoordDef> IntCoordDef;
// Cartesian coordinate dimension
int cartdim;
// Internal coordinate origin
at::Tensor origin;

// Internal coordinates who are scaled by themselves are picked out by self_scaling matrix
// The scaled internal coordinate vector is q = erf(self_scaling.mv(q)) + self_complete.mv(q)
at::Tensor self_scaling, self_complete;
// other_scaling[i][0] is scaled by [i][1] with alpha = [i][2]
std::vector<OthScalRul> other_scaling;
// Number of irreducible representations
size_t NIrred;
// Number of symmetry adapted internal coordinates per irreducible
std::vector<size_t> NSAIC_per_irred;
// symmetry_adaptation[i][j] contains the definition of
// j-th symmetry adapted internal coordinate in i-th irreducible
std::vector<std::vector<SymmAdLinComb>> symmetry_adaptation;

void define_SSAIC(const std::string & SSAIC_in) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    // Read SSAIC.in, define scale and symmetry
    std::ifstream ifs; ifs.open(SSAIC_in);
        std::string line;
        std::vector<std::string> strs_vec;
        std::forward_list<std::string> strs_flist;
        // File format of internal coordinate definition and origin
        std::string format;
        std::getline(ifs, line);
        std::getline(ifs, format);
        CL::utility::trim(format);
        // Internal coordinate definition file
        std::string IntCoordDef_file;
        std::getline(ifs, line);
        std::getline(ifs, IntCoordDef_file);
        CL::utility::trim(IntCoordDef_file);
        // Internal coordinate origin file
        std::string origin_file;
        std::getline(ifs, line);
        std::getline(ifs, origin_file);
        CL::utility::trim(origin_file);
        // Internal coordinates who are scaled by themselves
        std::vector<size_t> self_scaling_vector;
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            if (! std::regex_match(line, std::regex("\\ *\\d+\\ *"))) break;
            self_scaling_vector.push_back(std::stoul(line)-1);
        }
        // Internal coordinates who are scaled by others
        while (true) {
            std::getline(ifs, line); CL::utility::split(line, strs_vec);
            if (! std::regex_match(strs_vec[0], std::regex("\\d+"))) break;
            other_scaling.push_back(OthScalRul(strs_vec));
        }
        // Number of symmetry adapted coordinates per irreducible
        std::getline(ifs, line); CL::utility::split(line, strs_vec);
        NIrred = strs_vec.size();
        NSAIC_per_irred.resize(NIrred);
        for (size_t i = 0; i < NIrred; i++) NSAIC_per_irred[i] = std::stoul(strs_vec[i]);
        // Symmetry adapted linear combinations of each irreducible
        symmetry_adaptation.resize(NIrred);
        std::getline(ifs, line);
        for (std::vector<SymmAdLinComb> & SALCs : symmetry_adaptation) {
            int count = -1;
            while (true) {
                std::getline(ifs, line);
                if (! ifs.good()) break;
                CL::utility::split(line, strs_flist);
                if (! std::regex_match(strs_flist.front(), std::regex("-?\\d+\\.?\\d*"))) break;
                if (std::regex_match(strs_flist.front(), std::regex("\\d+"))) {
                    count++;
                    SALCs.push_back(SymmAdLinComb());
                    strs_flist.pop_front();
                }
                SALCs[count].coeff.push_back(std::stod(strs_flist.front()));
                strs_flist.pop_front();
                SALCs[count].IntCoord.push_back(std::stoul(strs_flist.front())-1);
            }
            // Normalize linear combination coefficients
            for (SymmAdLinComb & SALC : SALCs) {
                double norm = CL::LA::norm2(SALC.coeff);
                for (size_t k = 0; k < SALC.coeff.size(); k++)
                SALC.coeff[k] /= norm;
            }
        }
    ifs.close();
    // Define internal coordinate
    FL::GT::FetchInternalCoordinateDefinition(format, IntCoordDef_file, intdim, IntCoordDef);
    FL::GT::DefineInternalCoordinate(format, IntCoordDef_file);
    std::cout << "Number of internal coordinates: " << intdim << '\n';
    // Generate self_scaling and self_complete matrices
    self_scaling = at::zeros({intdim, intdim}, top);
    self_complete = at::eye(intdim, top);
    for (size_t & scaling : self_scaling_vector) {
        self_scaling[scaling][scaling] = 1.0;
        self_complete[scaling][scaling] = 0.0;
    }
    // Define origin
    origin = at::empty(intdim, top);
    if (format == "Columbus7") {
        CL::chemistry::xyz_mass<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GT::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    }
    else {
        CL::chemistry::xyz<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GT::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    }
}

std::vector<at::Tensor> compute_SSAIC(const at::Tensor & q) {
    // Nondimensionalize
    at::Tensor work = q - origin;
    for (size_t i = 0; i < intdim; i++)
    if (IntCoordDef[i].motion[0].type == "stretching")
    work[i] /= origin[i];
// Need a periodic check someday, e.g. torsion1 + torsion2 should belong to [-pi, pi] rather than [-2pi, 2pi]
    // Scale
    for (OthScalRul & scaling : other_scaling) work[scaling.self] *= at::exp(-scaling.alpha * work[scaling.scaler]);
    work = M_PI * at::erf(self_scaling.mv(work)) + self_complete.mv(work);
    // Symmetrize
    std::vector<at::Tensor> SSAgeom(NIrred);
    for (size_t irred = 0; irred < NIrred; irred++) {
        SSAgeom[irred] = q.new_zeros(symmetry_adaptation[irred].size());
        for (size_t i = 0; i < SSAgeom[irred].size(0); i++) {
            for (size_t j = 0; j < symmetry_adaptation[irred][i].coeff.size(); j++)
            SSAgeom[irred][i] += symmetry_adaptation[irred][i].coeff[j]
                * work[symmetry_adaptation[irred][i].IntCoord[j]];
        }
    }
    return SSAgeom;
}

} // namespace SSAIC