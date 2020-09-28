#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>
#include <FortranLibrary.hpp>

#include <libSGT.hpp>

namespace sad {
    // Wrapper control
    size_t state_of_interest;

    // Adiabatic gradient and Hessian wrapper
    void g(double * g, const double * q, const int & intdim, const int & intdim_) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        CL::TS::LA::UT_A3_U_(dH, state);
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
    }
    void h(double * h, const double * q, const int & intdim, const int & intdim_) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor energy, dH, ddH;
        std::tie(energy, dH, ddH) = libSGT::compute_energy_dHa_ddHa_int(q_tensor);
        std::memcpy(h, ddH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * intdim * sizeof(double));
    }

    // Diabatic gradient and Hessian wrapper
    void gd(double * g, const double * q, const int & intdim, const int & intdim_) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
    }
    void hd(double * h, const double * q, const int & intdim, const int & intdim_) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH, ddH;
        std::tie(H, dH, ddH) = libSGT::compute_Hd_dHd_ddHd_int(q_tensor);
        std::memcpy(h, ddH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * intdim * sizeof(double));
    }

    void search_sad(double * q, const int & intdim, size_t state, bool diabatic) {
        state_of_interest = state;
        if (diabatic) {
            FL::NO::TrustRegion(gd, hd, q, intdim, intdim);
        }
        else {
            FL::NO::TrustRegion(g, h, q, intdim, intdim);
        }
    }
} // namespace sad