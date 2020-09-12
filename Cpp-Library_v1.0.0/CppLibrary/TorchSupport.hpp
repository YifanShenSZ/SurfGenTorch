// Support for libtorch

#ifndef TorchSupport_hpp
#define TorchSupport_hpp

#include <torch/torch.h>

namespace CL { namespace TS {

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters);

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters);

/*
Additional linear algebra for libtorch tensor

Nomenclature (following LAPACK):
    ge  = general
    sy  = real symmetric
    asy = anti symmetric
    po  = real symmetric positive definite
Only use upper triangle of sy & po, strictly upper triangle of asy, otherwise specified

Symmetric high order tensor definition:
    3rd-order tensor: A_ijk = A_jik
*/
namespace LA {
    double triple_product(const at::Tensor & a, const at::Tensor & b, const at::Tensor & c);

    // Matrix dot multiplication for 3rd-order tensor A and B
    // A.size(2) == B.size(2), A.size(1) == B.size(0)
    // result_ij = A_ikm * B_kjm
    at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B);
    void ge3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result);
    // For symmetric A and B
    at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B);
    void sy3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result);

    // Unitary transformation for symmetric 3rd-order tensor A
    // result_ijm = U^T_ia * A_abm * U_bj
    at::Tensor UT_A3_U(const at::Tensor & UT, const at::Tensor & A, const at::Tensor & U);
    at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & U);
    // On exit A harvests the result
    void UT_A3_U_InPlace(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U);
    void UT_A3_U_InPlace(at::Tensor & A, const at::Tensor & U);
} // namespace LA

/*
An interal coordinate is the linear combination of several translationally and rotationally invariant displacements
but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately
unless appropriate metric tensor is applied

Nomenclature:
    cartdim & intdim: Cartesian & internal space dimensionality
    r: Cartesian coordinate vector
    q: internal coordinate vector
    J: the Jacobian matrix of q over r

Warning:
    * J of bending is singular at 0 or pi,
      so please avoid using bending in those cases
    * J of out of plane is singular at +-pi/2,
      so please avoid using out of plane in those cases
    * Backward propagation through q may be problematic for torsion when q = 0 or pi,
      so please use J explicitly in those cases
*/
namespace IC {
    struct InvolvedMotion {
        // Motion type
        std::string type;
        // Involved atoms
        std::vector<size_t> atom;
        // Linear combination coefficient
        double coeff;
        // For torsion only, deafult = -pi
        // if (the dihedral angle < min)       angle += 2pi
        // if (the dihedral angle > min + 2pi) angle -= 2pi
        double min;

        InvolvedMotion();
        InvolvedMotion(const std::string & type, const std::vector<size_t> & atom, const double & coeff, const double & min = -M_PI);
        ~InvolvedMotion();
    };
    struct IntCoordDef {
        std::vector<InvolvedMotion> motion;
    
        IntCoordDef();
        ~IntCoordDef();
    };

    // Store different internal coordinate definitions
    extern std::vector<std::vector<IntCoordDef>> definitions;

    // Input:  file format (Columbus7, default), internal coordinate definition file
    // Output: intdim, internal coordinate definition ID
    std::tuple<int64_t, size_t> define_IC(const std::string & format, const std::string & file);

    // Convert r to q according to ID-th internal coordinate definition
    at::Tensor compute_IC(const at::Tensor & r, const size_t & ID = 0);

    // From r, generate q & J according to ID-th internal coordinate definition
    std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r, const size_t & ID = 0);
}

namespace chemistry {
    bool check_degeneracy(const double & threshold, const at::Tensor & energy);

    // Transform adiabatic energy (H) and gradient (dH) to composite representation
    void composite_representation(at::Tensor & H, at::Tensor & dH);

    // Matrix off-diagonal elements do not have determinate phase, because
    // the eigenvectors defining a representation have indeterminate phase difference
    void initialize_phase_fixing(const size_t & NStates_);
    // Fix M by minimizing || M - ref ||_F^2
    void fix(at::Tensor & M, const at::Tensor & ref);
    // Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    void fix(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight);
} // namespace chemistry

} // namespace TS
} // namespace CL

#endif