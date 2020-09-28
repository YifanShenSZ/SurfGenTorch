#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>
#include <FortranLibrary.hpp>

#include <libSGT.hpp>

namespace min {
    // Wrapper control
    size_t state_of_interest;

    // Adiabatic energy and gradient and Hessian wrapper
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

    // Diabatic "energy" and gradient and Hessian wrapper
    void ed(double & e, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        at::Tensor H = libSGT::compute_Hd_int(q_tensor);
        e = H[state_of_interest][state_of_interest].item<double>();
    }
    void gd(double * g, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
    }
    int ed_gd(double & e, double * g, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH;
        std::tie(H, dH) = libSGT::compute_Hd_dHd_int(q_tensor);
        e = H[state_of_interest][state_of_interest].item<double>();
        std::memcpy(g, dH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * sizeof(double));
        return 0;
    }
    int hd(double * h, const double * q, const int & intdim) {
        at::Tensor q_tensor = at::from_blob(const_cast<double *>(q), intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH, ddH;
        std::tie(H, dH, ddH) = libSGT::compute_Hd_dHd_ddHd_int(q_tensor);
        std::memcpy(h, ddH[state_of_interest][state_of_interest].data_ptr<double>(), intdim * intdim * sizeof(double));
        return 0;
    }

    void search_min(double * q, const int & intdim, size_t state, bool diabatic, std::string opt) {
        state_of_interest = state;
        if (diabatic) {
            if (opt == "CG") {
                FL::NO::ConjugateGradient(ed, gd, ed_gd, q, intdim);
            }
            else {
                FL::NO::BFGS(ed, gd, ed_gd, hd, q, intdim);
            }
        }
        else {
            if (opt == "CG") {
                FL::NO::ConjugateGradient(e, g, e_g, q, intdim);
            }
            else {
                FL::NO::BFGS(e, g, e_g, h, q, intdim);
            }
        }
    }
} // namespace min