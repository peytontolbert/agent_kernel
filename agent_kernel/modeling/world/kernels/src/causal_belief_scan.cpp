#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32 || (x).scalar_type() == torch::kFloat64, #x " must be float32 or float64")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOATING(x)

namespace agent_kernel::world {

std::vector<torch::Tensor> causal_belief_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_belief_scan_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_log_belief,
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor final_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_belief_scan_fwd(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_size) {
  CHECK_INPUT(local_logits);
  CHECK_INPUT(transition_log_probs);
  CHECK_INPUT(transition_context);
  CHECK_INPUT(initial_log_belief);

  TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape (batch, length, states)");
  TORCH_CHECK(
      transition_log_probs.dim() == 1 || transition_log_probs.dim() == 2,
      "transition_log_probs must have shape (states,) or (states, states)");
  TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits");
  TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape (batch, states)");
  TORCH_CHECK(initial_log_belief.size(0) == local_logits.size(0), "initial_log_belief batch mismatch");
  TORCH_CHECK(initial_log_belief.size(1) == local_logits.size(2), "initial_log_belief state mismatch");
  if (transition_log_probs.dim() == 1) {
    TORCH_CHECK(transition_log_probs.size(0) == local_logits.size(2), "transition_log_probs state mismatch");
  } else {
    TORCH_CHECK(
        transition_log_probs.size(0) == local_logits.size(2) && transition_log_probs.size(1) == local_logits.size(2),
        "transition_log_probs state mismatch");
  }
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  return causal_belief_scan_forward_cuda(
      local_logits,
      transition_log_probs,
      transition_context,
      initial_log_belief,
      transition_gate,
      transition_bandwidth,
      chunk_size);
}

std::vector<torch::Tensor> causal_belief_scan_bwd(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_log_belief,
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor final_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_size) {
  CHECK_INPUT(grad_beliefs);
  CHECK_INPUT(grad_final_log_belief);
  CHECK_INPUT(local_logits);
  CHECK_INPUT(transition_log_probs);
  CHECK_INPUT(transition_context);
  CHECK_INPUT(initial_log_belief);
  CHECK_INPUT(beliefs);
  CHECK_INPUT(final_log_belief);

  TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs");
  TORCH_CHECK(grad_final_log_belief.sizes() == final_log_belief.sizes(), "grad_final_log_belief must match final_log_belief");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  return causal_belief_scan_backward_cuda(
      grad_beliefs,
      grad_final_log_belief,
      local_logits,
      transition_log_probs,
      transition_context,
      initial_log_belief,
      beliefs,
      final_log_belief,
      transition_gate,
      transition_bandwidth,
      chunk_size);
}

}  // namespace agent_kernel::world

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &agent_kernel::world::causal_belief_scan_fwd, "Native causal belief scan forward");
  m.def("bwd", &agent_kernel::world::causal_belief_scan_bwd, "Native causal belief scan backward");
}
