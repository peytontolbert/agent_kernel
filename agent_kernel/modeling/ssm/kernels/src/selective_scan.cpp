#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace agent_kernel::ssm {

void selective_scan_forward_cuda(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& delta_bias,
    bool delta_softplus,
    torch::Tensor& out,
    torch::Tensor& last_state,
    torch::Tensor& x_hist,
    torch::Tensor& y_base);

void selective_scan_backward_cuda(
    const torch::Tensor& dout,
    const torch::Tensor& dlast_state,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& x_hist,
    const torch::Tensor& y_base,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& delta_bias,
    bool delta_softplus,
    torch::Tensor& du,
    torch::Tensor& ddelta,
    torch::Tensor& dA_full,
    torch::Tensor& dB,
    torch::Tensor& dC,
    torch::Tensor& dD_full,
    torch::Tensor& dz,
    torch::Tensor& ddelta_bias_full);

std::vector<torch::Tensor> selective_scan_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& delta_bias,
    bool delta_softplus) {
  CHECK_INPUT(u);
  CHECK_INPUT(delta);
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  if (D.has_value()) {
    CHECK_INPUT(*D);
  }
  if (z.has_value()) {
    CHECK_INPUT(*z);
  }
  if (delta_bias.has_value()) {
    CHECK_INPUT(*delta_bias);
  }

  TORCH_CHECK(u.dim() == 3, "u must have shape (batch, dim, length)");
  TORCH_CHECK(delta.sizes() == u.sizes(), "delta must match u");
  TORCH_CHECK(A.dim() == 2, "A must have shape (dim, d_state)");
  TORCH_CHECK(B.dim() == 4, "B must have normalized shape (batch, dim, d_state, length)");
  TORCH_CHECK(C.dim() == 4, "C must have normalized shape (batch, dim, d_state, length)");
  TORCH_CHECK(A.size(0) == u.size(1), "A dimension mismatch");
  TORCH_CHECK(B.size(0) == u.size(0) && B.size(1) == u.size(1) && B.size(3) == u.size(2), "B shape mismatch");
  TORCH_CHECK(C.size(0) == u.size(0) && C.size(1) == u.size(1) && C.size(3) == u.size(2), "C shape mismatch");
  TORCH_CHECK(B.size(2) == A.size(1) && C.size(2) == A.size(1), "d_state mismatch");

  auto out = torch::zeros_like(u);
  auto last_state = torch::zeros({u.size(0), u.size(1), A.size(1)}, u.options());
  auto x_hist = torch::zeros({u.size(0), u.size(1), u.size(2), A.size(1)}, u.options());
  auto y_base = torch::zeros_like(u);

  selective_scan_forward_cuda(
      u,
      delta,
      A,
      B,
      C,
      D,
      z,
      delta_bias,
      delta_softplus,
      out,
      last_state,
      x_hist,
      y_base);

  return {out, last_state, x_hist, y_base};
}

std::vector<torch::Tensor> selective_scan_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& dlast_state,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& x_hist,
    const torch::Tensor& y_base,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& delta_bias,
    bool delta_softplus) {
  CHECK_INPUT(dout);
  CHECK_INPUT(dlast_state);
  CHECK_INPUT(u);
  CHECK_INPUT(delta);
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  CHECK_INPUT(x_hist);
  CHECK_INPUT(y_base);
  if (D.has_value()) {
    CHECK_INPUT(*D);
  }
  if (z.has_value()) {
    CHECK_INPUT(*z);
  }
  if (delta_bias.has_value()) {
    CHECK_INPUT(*delta_bias);
  }

  TORCH_CHECK(dlast_state.dim() == 3, "dlast_state must have shape (batch, dim, d_state)");
  TORCH_CHECK(dlast_state.size(0) == u.size(0) && dlast_state.size(1) == u.size(1) && dlast_state.size(2) == A.size(1),
              "dlast_state shape mismatch");

  auto du = torch::zeros_like(u);
  auto ddelta = torch::zeros_like(delta);
  auto dA_full = torch::zeros({u.size(0), u.size(1), A.size(1)}, A.options());
  auto dB = torch::zeros_like(B);
  auto dC = torch::zeros_like(C);
  auto dD_full = torch::zeros({u.size(0), u.size(1)}, u.options());
  auto dz = z.has_value() ? torch::zeros_like(*z) : torch::zeros({0}, u.options());
  auto ddelta_bias_full = delta_bias.has_value() ? torch::zeros({u.size(0), u.size(1)}, delta.options())
                                                 : torch::zeros({0}, delta.options());

  selective_scan_backward_cuda(
      dout,
      dlast_state,
      u,
      delta,
      A,
      B,
      C,
      x_hist,
      y_base,
      D,
      z,
      delta_bias,
      delta_softplus,
      du,
      ddelta,
      dA_full,
      dB,
      dC,
      dD_full,
      dz,
      ddelta_bias_full);

  return {du, ddelta, dA_full, dB, dC, dD_full, dz, ddelta_bias_full};
}

}  // namespace agent_kernel::ssm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &agent_kernel::ssm::selective_scan_fwd, "Native selective scan forward");
  m.def("bwd", &agent_kernel::ssm::selective_scan_bwd, "Native selective scan backward");
}
