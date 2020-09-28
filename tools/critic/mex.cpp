#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>
#include <FortranLibrary.hpp>

#include <libSGT.hpp>

namespace mex {
    // Wrapper control
    size_t state_of_interest;

    // Adiabatic energy, gradient, Hessian, gap, gap gradient, gap Hessian wrapper
    void e(double & e, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        at::Tensor H = libSGT::compute_Hd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig();
        e = energy[state_of_interest].item<double>();
    }
    void g(double * g, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        CL::TS::LA::UT_A3_U_(dH, state);
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
    }
    int e_g(double & e, double * g, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        CL::TS::LA::UT_A3_U_(dH, state);
        e = energy[state_of_interest].item<double>();
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
        return 0;
    }
    int h(double * h, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor energy, dH, ddH;
        std::tie(energy, dH, ddH) = libSGT::compute_energy_dHa_ddHa_int(q_tensor);
        std::memcpy(h, ddH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * intdim * sizeof(double));
        return 0;
    }
    void gap(double * gap, const double * q, const int & M, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        at::Tensor H = libSGT::compute_Hd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig();
        gap[0] = (energy[state_of_interest + 1] - energy[state_of_interest]).item<double>();
    }
    void gapgrad(double * g, const double * q, const int & M, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        at::Tensor energy, state;
        std::tie(energy, state) = H.symeig(true);
        CL::TS::LA::UT_A3_U_(dH, state);
        at::Tensor grad = dH[state_of_interest + 1][state_of_interest + 1] - dH[state_of_interest][state_of_interest];
        std::memcpy(g, grad.data_ptr<double>(), grad.numel() * sizeof(double));
    }
    int gaphess(double * h, const double * q, const int & M, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor energy, dH, ddH;
        std::tie(energy, dH, ddH) = libSGT::compute_energy_dHa_ddHa_int(q_tensor);
        at::Tensor hess = ddH[state_of_interest + 1][state_of_interest + 1] - ddH[state_of_interest][state_of_interest];
        std::memcpy(h, hess.data_ptr<double>(), hess.numel() * sizeof(double));
        return 0;
    }

    void search_mex(double * q, const int & intdim, size_t state, std::string opt) {
        state_of_interest = state;
        FL::NO::AugmentedLagrangian(e, g, e_g, h, gap, gapgrad, gaphess, q, intdim, 1);
    }
} // namespace mex