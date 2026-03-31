#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace agent_kernel::world {

#define CUDA_KERNEL_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()

template <typename acc_t>
__device__ __forceinline__ acc_t negative_infinity() {
  return -std::numeric_limits<acc_t>::infinity();
}

template <typename acc_t>
__device__ acc_t block_reduce_max(acc_t value, acc_t* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = max(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename acc_t>
__device__ acc_t block_reduce_sum(acc_t value, acc_t* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename scalar_t>
__global__ void causal_belief_forward_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ transition_log_probs_t,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    scalar_t transition_gate,
    int transition_bandwidth,
    int seq_len,
    int num_states,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
  using acc_t = at::acc_type<scalar_t, true>;

  const int batch_index = blockIdx.x;
  const int state_index = threadIdx.x;
  const bool active = state_index < num_states;

  extern __shared__ unsigned char shared_raw[];
  auto* prev = reinterpret_cast<acc_t*>(shared_raw);
  auto* pred = prev + blockDim.x;
  auto* scratch = pred + blockDim.x;

  prev[state_index] = active ? static_cast<acc_t>(initial_log_belief[batch_index * num_states + state_index]) : negative_infinity<acc_t>();
  __syncthreads();

  for (int offset = 0; offset < chunk_len; ++offset) {
    const int pos = chunk_start + offset;
    const int base = (batch_index * seq_len + pos) * num_states;

    acc_t pred_j = negative_infinity<acc_t>();
    if (active) {
      acc_t pred_max = negative_infinity<acc_t>();
      const int source_start = max(0, state_index - transition_bandwidth);
      const int source_end = min(num_states - 1, state_index + transition_bandwidth);
      for (int source = source_start; source <= source_end; ++source) {
        pred_max = max(
            pred_max,
            prev[source] + static_cast<acc_t>(transition_log_probs_t[state_index * num_states + source]));
      }
      acc_t pred_sum = acc_t(0);
      for (int source = source_start; source <= source_end; ++source) {
        pred_sum += exp(
            prev[source] +
                static_cast<acc_t>(transition_log_probs_t[state_index * num_states + source]) -
                pred_max);
      }
      pred_j = log(max(pred_sum, acc_t(1.0e-20))) + pred_max;
      pred[state_index] = pred_j;
    }
    __syncthreads();

    const acc_t obs_value = active
        ? static_cast<acc_t>(local_logits[base + state_index]) +
              static_cast<acc_t>(transition_gate) *
                  (pred_j + static_cast<acc_t>(transition_context[base + state_index]))
        : negative_infinity<acc_t>();
    const acc_t obs_max = block_reduce_max(obs_value, scratch);
    const acc_t obs_exp = active ? exp(obs_value - obs_max) : acc_t(0);
    const acc_t obs_sum = block_reduce_sum(obs_exp, scratch);
    const acc_t log_norm = log(max(obs_sum, acc_t(1.0e-20))) + obs_max;
    if (active) {
      const acc_t q = obs_value - log_norm;
      beliefs[base + state_index] = static_cast<scalar_t>(q);
      prev[state_index] = q;
    } else {
      prev[state_index] = negative_infinity<acc_t>();
    }
    __syncthreads();
  }

  if (active) {
    final_log_belief[batch_index * num_states + state_index] = static_cast<scalar_t>(prev[state_index]);
  }
}

template <typename scalar_t>
__global__ void causal_belief_forward_diag_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ transition_log_diag,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    scalar_t transition_gate,
    int seq_len,
    int num_states,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
  using acc_t = at::acc_type<scalar_t, true>;

  const int batch_index = blockIdx.x;
  const int state_index = threadIdx.x;
  const bool active = state_index < num_states;

  extern __shared__ unsigned char shared_raw[];
  auto* prev = reinterpret_cast<acc_t*>(shared_raw);
  auto* scratch = prev + blockDim.x;

  prev[state_index] = active ? static_cast<acc_t>(initial_log_belief[batch_index * num_states + state_index]) : negative_infinity<acc_t>();
  __syncthreads();

  for (int offset = 0; offset < chunk_len; ++offset) {
    const int pos = chunk_start + offset;
    const int base = (batch_index * seq_len + pos) * num_states;
    const acc_t pred_j = active
        ? prev[state_index] + static_cast<acc_t>(transition_log_diag[state_index])
        : negative_infinity<acc_t>();
    const acc_t obs_value = active
        ? static_cast<acc_t>(local_logits[base + state_index]) +
              static_cast<acc_t>(transition_gate) *
                  (pred_j + static_cast<acc_t>(transition_context[base + state_index]))
        : negative_infinity<acc_t>();
    const acc_t obs_max = block_reduce_max(obs_value, scratch);
    const acc_t obs_exp = active ? exp(obs_value - obs_max) : acc_t(0);
    const acc_t obs_sum = block_reduce_sum(obs_exp, scratch);
    const acc_t log_norm = log(max(obs_sum, acc_t(1.0e-20))) + obs_max;
    if (active) {
      const acc_t q = obs_value - log_norm;
      beliefs[base + state_index] = static_cast<scalar_t>(q);
      prev[state_index] = q;
    }
    __syncthreads();
  }

  if (active) {
    final_log_belief[batch_index * num_states + state_index] = static_cast<scalar_t>(prev[state_index]);
  }
}

template <typename scalar_t>
__global__ void causal_belief_backward_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_log_belief,
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ transition_log_probs,
    const scalar_t* __restrict__ transition_log_probs_t,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_chunk_log_belief,
    const scalar_t* __restrict__ beliefs,
    scalar_t transition_gate,
    int transition_bandwidth,
    int seq_len,
    int num_states,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ grad_local_logits,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_chunk_log_belief,
    scalar_t* __restrict__ grad_transition_per_batch,
    scalar_t* __restrict__ grad_transition_gate_per_batch) {
  using acc_t = at::acc_type<scalar_t, true>;

  const int batch_index = blockIdx.x;
  const int state_index = threadIdx.x;
  const bool active = state_index < num_states;

  extern __shared__ unsigned char shared_raw[];
  auto* prev = reinterpret_cast<acc_t*>(shared_raw);
  auto* pred = prev + blockDim.x;
  auto* q = pred + blockDim.x;
  auto* carry = q + blockDim.x;
  auto* grad_pred = carry + blockDim.x;
  auto* scratch = grad_pred + blockDim.x;

  carry[state_index] = active ? static_cast<acc_t>(grad_final_log_belief[batch_index * num_states + state_index]) : acc_t(0);
  __syncthreads();

  acc_t gate_grad_accum = acc_t(0);
  scalar_t* grad_transition_batch = grad_transition_per_batch + batch_index * num_states * num_states;

  for (int offset = chunk_len - 1; offset >= 0; --offset) {
    const int pos = chunk_start + offset;
    const int base = (batch_index * seq_len + pos) * num_states;

    if (active) {
      prev[state_index] = (offset == 0)
          ? static_cast<acc_t>(initial_chunk_log_belief[batch_index * num_states + state_index])
          : static_cast<acc_t>(beliefs[(batch_index * seq_len + (pos - 1)) * num_states + state_index]);
      q[state_index] = static_cast<acc_t>(beliefs[base + state_index]);
    } else {
      prev[state_index] = negative_infinity<acc_t>();
      q[state_index] = negative_infinity<acc_t>();
    }
    __syncthreads();

    acc_t pred_j = negative_infinity<acc_t>();
    if (active) {
      acc_t pred_max = negative_infinity<acc_t>();
      const int source_start = max(0, state_index - transition_bandwidth);
      const int source_end = min(num_states - 1, state_index + transition_bandwidth);
      for (int source = source_start; source <= source_end; ++source) {
        pred_max = max(
            pred_max,
            prev[source] + static_cast<acc_t>(transition_log_probs_t[state_index * num_states + source]));
      }
      acc_t pred_sum = acc_t(0);
      for (int source = source_start; source <= source_end; ++source) {
        pred_sum += exp(
            prev[source] +
                static_cast<acc_t>(transition_log_probs_t[state_index * num_states + source]) -
                pred_max);
      }
      pred_j = log(max(pred_sum, acc_t(1.0e-20))) + pred_max;
      pred[state_index] = pred_j;
    }
    __syncthreads();

    const acc_t gq = active ? static_cast<acc_t>(grad_beliefs[base + state_index]) + carry[state_index] : acc_t(0);
    const acc_t gq_sum = block_reduce_sum(gq, scratch);
    const acc_t ga = active ? gq - exp(q[state_index]) * gq_sum : acc_t(0);

    if (active) {
      grad_local_logits[base + state_index] = static_cast<scalar_t>(ga);
      grad_transition_context[base + state_index] = static_cast<scalar_t>(static_cast<acc_t>(transition_gate) * ga);
      grad_pred[state_index] = static_cast<acc_t>(transition_gate) * ga;
      gate_grad_accum += ga * (pred_j + static_cast<acc_t>(transition_context[base + state_index]));
    } else {
      grad_pred[state_index] = acc_t(0);
    }
    __syncthreads();

    acc_t prev_grad = acc_t(0);
    if (active) {
      const acc_t prev_state = prev[state_index];
      const int target_start = max(0, state_index - transition_bandwidth);
      const int target_end = min(num_states - 1, state_index + transition_bandwidth);
      for (int target = target_start; target <= target_end; ++target) {
        const acc_t weight = exp(prev_state + static_cast<acc_t>(transition_log_probs[state_index * num_states + target]) - pred[target]);
        const acc_t contrib = grad_pred[target] * weight;
        grad_transition_batch[state_index * num_states + target] += static_cast<scalar_t>(contrib);
        prev_grad += contrib;
      }
    }
    carry[state_index] = active ? prev_grad : acc_t(0);
    __syncthreads();
  }

  if (active) {
    grad_initial_chunk_log_belief[batch_index * num_states + state_index] = static_cast<scalar_t>(carry[state_index]);
  }
  const acc_t gate_sum = block_reduce_sum(gate_grad_accum, scratch);
  if (state_index == 0) {
    grad_transition_gate_per_batch[batch_index] += static_cast<scalar_t>(gate_sum);
  }
}

template <typename scalar_t>
__global__ void causal_belief_backward_diag_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_log_belief,
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ transition_log_diag,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_chunk_log_belief,
    const scalar_t* __restrict__ beliefs,
    scalar_t transition_gate,
    int seq_len,
    int num_states,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ grad_local_logits,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_chunk_log_belief,
    scalar_t* __restrict__ grad_transition_diag_per_batch,
    scalar_t* __restrict__ grad_transition_gate_per_batch) {
  using acc_t = at::acc_type<scalar_t, true>;

  const int batch_index = blockIdx.x;
  const int state_index = threadIdx.x;
  const bool active = state_index < num_states;

  extern __shared__ unsigned char shared_raw[];
  auto* prev = reinterpret_cast<acc_t*>(shared_raw);
  auto* q = prev + blockDim.x;
  auto* carry = q + blockDim.x;
  auto* scratch = carry + blockDim.x;

  carry[state_index] = active ? static_cast<acc_t>(grad_final_log_belief[batch_index * num_states + state_index]) : acc_t(0);
  __syncthreads();

  acc_t gate_grad_accum = acc_t(0);
  scalar_t* grad_transition_batch = grad_transition_diag_per_batch + batch_index * num_states;

  for (int offset = chunk_len - 1; offset >= 0; --offset) {
    const int pos = chunk_start + offset;
    const int base = (batch_index * seq_len + pos) * num_states;

    if (active) {
      prev[state_index] = (offset == 0)
          ? static_cast<acc_t>(initial_chunk_log_belief[batch_index * num_states + state_index])
          : static_cast<acc_t>(beliefs[(batch_index * seq_len + (pos - 1)) * num_states + state_index]);
      q[state_index] = static_cast<acc_t>(beliefs[base + state_index]);
    } else {
      prev[state_index] = negative_infinity<acc_t>();
      q[state_index] = negative_infinity<acc_t>();
    }
    __syncthreads();

    const acc_t pred_j = active
        ? prev[state_index] + static_cast<acc_t>(transition_log_diag[state_index])
        : negative_infinity<acc_t>();
    const acc_t gq = active ? static_cast<acc_t>(grad_beliefs[base + state_index]) + carry[state_index] : acc_t(0);
    const acc_t gq_sum = block_reduce_sum(gq, scratch);
    const acc_t ga = active ? gq - exp(q[state_index]) * gq_sum : acc_t(0);

    if (active) {
      const acc_t grad_pred = static_cast<acc_t>(transition_gate) * ga;
      grad_local_logits[base + state_index] = static_cast<scalar_t>(ga);
      grad_transition_context[base + state_index] = static_cast<scalar_t>(grad_pred);
      grad_transition_batch[state_index] += static_cast<scalar_t>(grad_pred);
      carry[state_index] = grad_pred;
      gate_grad_accum += ga * (pred_j + static_cast<acc_t>(transition_context[base + state_index]));
    } else {
      carry[state_index] = acc_t(0);
    }
    __syncthreads();
  }

  if (active) {
    grad_initial_chunk_log_belief[batch_index * num_states + state_index] = static_cast<scalar_t>(carry[state_index]);
  }
  const acc_t gate_sum = block_reduce_sum(gate_grad_accum, scratch);
  if (state_index == 0) {
    grad_transition_gate_per_batch[batch_index] += static_cast<scalar_t>(gate_sum);
  }
}

inline int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename scalar_t>
void launch_forward_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_probs_t,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
  const int batch_size = static_cast<int>(local_logits.size(0));
  const int seq_len = static_cast<int>(local_logits.size(1));
  const int num_states = static_cast<int>(local_logits.size(2));
  const int threads = next_power_of_two(num_states);
  TORCH_CHECK(threads <= 1024, "causal belief scan supports at most 1024 latent states");
  const dim3 grid(batch_size);
  const dim3 block(threads);
  const auto shared_bytes = static_cast<size_t>(3 * threads * sizeof(typename at::acc_type<scalar_t, true>));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  causal_belief_forward_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      local_logits.data_ptr<scalar_t>(),
      transition_log_probs_t.data_ptr<scalar_t>(),
      transition_context.data_ptr<scalar_t>(),
      initial_log_belief.data_ptr<scalar_t>(),
      static_cast<scalar_t>(transition_gate),
      static_cast<int>(transition_bandwidth),
      seq_len,
      num_states,
      static_cast<int>(chunk_start),
      static_cast<int>(chunk_len),
      beliefs.data_ptr<scalar_t>(),
      final_log_belief.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

template <typename scalar_t>
void launch_forward_diag_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_diag,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
  const int batch_size = static_cast<int>(local_logits.size(0));
  const int seq_len = static_cast<int>(local_logits.size(1));
  const int num_states = static_cast<int>(local_logits.size(2));
  const int threads = next_power_of_two(num_states);
  TORCH_CHECK(threads <= 1024, "causal belief scan supports at most 1024 latent states");
  const dim3 grid(batch_size);
  const dim3 block(threads);
  const auto shared_bytes = static_cast<size_t>(2 * threads * sizeof(typename at::acc_type<scalar_t, true>));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  causal_belief_forward_diag_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      local_logits.data_ptr<scalar_t>(),
      transition_log_diag.data_ptr<scalar_t>(),
      transition_context.data_ptr<scalar_t>(),
      initial_log_belief.data_ptr<scalar_t>(),
      static_cast<scalar_t>(transition_gate),
      seq_len,
      num_states,
      static_cast<int>(chunk_start),
      static_cast<int>(chunk_len),
      beliefs.data_ptr<scalar_t>(),
      final_log_belief.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

template <typename scalar_t>
void launch_backward_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_log_belief,
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_probs,
    const torch::Tensor& transition_log_probs_t,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_chunk_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_chunk_log_belief,
    const torch::Tensor& grad_transition_per_batch,
    const torch::Tensor& grad_transition_gate_per_batch) {
  const int batch_size = static_cast<int>(local_logits.size(0));
  const int seq_len = static_cast<int>(local_logits.size(1));
  const int num_states = static_cast<int>(local_logits.size(2));
  const int threads = next_power_of_two(num_states);
  TORCH_CHECK(threads <= 1024, "causal belief scan supports at most 1024 latent states");
  const dim3 grid(batch_size);
  const dim3 block(threads);
  const auto shared_bytes = static_cast<size_t>(6 * threads * sizeof(typename at::acc_type<scalar_t, true>));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  causal_belief_backward_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      grad_beliefs.data_ptr<scalar_t>(),
      grad_final_log_belief.data_ptr<scalar_t>(),
      local_logits.data_ptr<scalar_t>(),
      transition_log_probs.data_ptr<scalar_t>(),
      transition_log_probs_t.data_ptr<scalar_t>(),
      transition_context.data_ptr<scalar_t>(),
      initial_chunk_log_belief.data_ptr<scalar_t>(),
      beliefs.data_ptr<scalar_t>(),
      static_cast<scalar_t>(transition_gate),
      static_cast<int>(transition_bandwidth),
      seq_len,
      num_states,
      static_cast<int>(chunk_start),
      static_cast<int>(chunk_len),
      grad_local_logits.data_ptr<scalar_t>(),
      grad_transition_context.data_ptr<scalar_t>(),
      grad_initial_chunk_log_belief.data_ptr<scalar_t>(),
      grad_transition_per_batch.data_ptr<scalar_t>(),
      grad_transition_gate_per_batch.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

template <typename scalar_t>
void launch_backward_diag_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_log_belief,
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_diag,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_chunk_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_chunk_log_belief,
    const torch::Tensor& grad_transition_diag_per_batch,
    const torch::Tensor& grad_transition_gate_per_batch) {
  const int batch_size = static_cast<int>(local_logits.size(0));
  const int seq_len = static_cast<int>(local_logits.size(1));
  const int num_states = static_cast<int>(local_logits.size(2));
  const int threads = next_power_of_two(num_states);
  TORCH_CHECK(threads <= 1024, "causal belief scan supports at most 1024 latent states");
  const dim3 grid(batch_size);
  const dim3 block(threads);
  const auto shared_bytes = static_cast<size_t>(4 * threads * sizeof(typename at::acc_type<scalar_t, true>));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  causal_belief_backward_diag_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      grad_beliefs.data_ptr<scalar_t>(),
      grad_final_log_belief.data_ptr<scalar_t>(),
      local_logits.data_ptr<scalar_t>(),
      transition_log_diag.data_ptr<scalar_t>(),
      transition_context.data_ptr<scalar_t>(),
      initial_chunk_log_belief.data_ptr<scalar_t>(),
      beliefs.data_ptr<scalar_t>(),
      static_cast<scalar_t>(transition_gate),
      seq_len,
      num_states,
      static_cast<int>(chunk_start),
      static_cast<int>(chunk_len),
      grad_local_logits.data_ptr<scalar_t>(),
      grad_transition_context.data_ptr<scalar_t>(),
      grad_initial_chunk_log_belief.data_ptr<scalar_t>(),
      grad_transition_diag_per_batch.data_ptr<scalar_t>(),
      grad_transition_gate_per_batch.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

std::vector<torch::Tensor> causal_belief_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t transition_bandwidth,
    int64_t chunk_size) {
  c10::cuda::CUDAGuard device_guard(local_logits.device());
  const auto seq_len = local_logits.size(1);
  auto beliefs = torch::empty_like(local_logits);
  auto prev = initial_log_belief.contiguous();
  auto final_log_belief = torch::empty_like(initial_log_belief);
  if (seq_len == 0) {
    final_log_belief.copy_(initial_log_belief);
    return {beliefs, final_log_belief};
  }
  if (transition_log_probs.dim() == 1) {
    AT_DISPATCH_FLOATING_TYPES(local_logits.scalar_type(), "causal_belief_scan_forward_diag_cuda", [&] {
      for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        const int64_t chunk_len = std::min(chunk_size, seq_len - chunk_start);
        launch_forward_diag_chunk<scalar_t>(
            local_logits,
            transition_log_probs.contiguous(),
            transition_context,
            prev,
            transition_gate,
            chunk_start,
            chunk_len,
            beliefs,
            final_log_belief);
        prev = final_log_belief.contiguous();
      }
    });
    return {beliefs, final_log_belief};
  }
  auto transition_log_probs_t = transition_log_probs.transpose(0, 1).contiguous();
  AT_DISPATCH_FLOATING_TYPES(local_logits.scalar_type(), "causal_belief_scan_forward_cuda", [&] {
    for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
      const int64_t chunk_len = std::min(chunk_size, seq_len - chunk_start);
      launch_forward_chunk<scalar_t>(
          local_logits,
          transition_log_probs_t,
          transition_context,
          prev,
          transition_gate,
          transition_bandwidth,
          chunk_start,
          chunk_len,
          beliefs,
          final_log_belief);
      prev = final_log_belief.contiguous();
    }
  });
  return {beliefs, final_log_belief};
}

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
    int64_t chunk_size) {
  c10::cuda::CUDAGuard device_guard(local_logits.device());
  const auto batch_size = local_logits.size(0);
  const auto seq_len = local_logits.size(1);
  const auto num_states = local_logits.size(2);

  auto grad_local_logits = torch::zeros_like(local_logits);
  auto grad_transition_context = torch::zeros_like(transition_context);
  auto grad_transition_gate_per_batch = torch::zeros({batch_size}, local_logits.options());
  auto carry = grad_final_log_belief.contiguous();
  auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);

  if (seq_len == 0) {
    grad_initial_log_belief.copy_(grad_final_log_belief);
    return {
        grad_local_logits,
        torch::zeros_like(transition_log_probs),
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate_per_batch.sum().reshape({1}),
    };
  }
  if (transition_log_probs.dim() == 1) {
    auto grad_transition_diag_per_batch = torch::zeros({batch_size, num_states}, local_logits.options());
    AT_DISPATCH_FLOATING_TYPES(local_logits.scalar_type(), "causal_belief_scan_backward_diag_cuda", [&] {
      for (int64_t chunk_end = seq_len; chunk_end > 0; chunk_end -= chunk_size) {
        const int64_t chunk_start = std::max<int64_t>(0, chunk_end - chunk_size);
        const int64_t this_chunk_len = chunk_end - chunk_start;
        auto prev = chunk_start == 0 ? initial_log_belief.contiguous() : beliefs.select(1, chunk_start - 1).contiguous();
        auto chunk_grad_initial = torch::zeros_like(initial_log_belief);
        launch_backward_diag_chunk<scalar_t>(
            grad_beliefs,
            carry,
            local_logits,
            transition_log_probs.contiguous(),
            transition_context,
            prev,
            beliefs,
            transition_gate,
            chunk_start,
            this_chunk_len,
            grad_local_logits,
            grad_transition_context,
            chunk_grad_initial,
            grad_transition_diag_per_batch,
            grad_transition_gate_per_batch);
        carry = chunk_grad_initial.contiguous();
      }
    });
    grad_initial_log_belief.copy_(carry);
    auto grad_transition_log_diag = grad_transition_diag_per_batch.sum(0);
    auto grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
    return {
        grad_local_logits,
        grad_transition_log_diag,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
    };
  }
  auto grad_transition_per_batch = torch::zeros({batch_size, num_states, num_states}, local_logits.options());
  auto transition_log_probs_t = transition_log_probs.transpose(0, 1).contiguous();

  AT_DISPATCH_FLOATING_TYPES(local_logits.scalar_type(), "causal_belief_scan_backward_cuda", [&] {
    for (int64_t chunk_end = seq_len; chunk_end > 0; chunk_end -= chunk_size) {
      const int64_t chunk_start = std::max<int64_t>(0, chunk_end - chunk_size);
      const int64_t this_chunk_len = chunk_end - chunk_start;
      auto prev = chunk_start == 0 ? initial_log_belief.contiguous() : beliefs.select(1, chunk_start - 1).contiguous();
      auto chunk_grad_initial = torch::zeros_like(initial_log_belief);
      launch_backward_chunk<scalar_t>(
          grad_beliefs,
          carry,
          local_logits,
          transition_log_probs,
          transition_log_probs_t,
          transition_context,
          prev,
          beliefs,
          transition_gate,
          transition_bandwidth,
          chunk_start,
          this_chunk_len,
          grad_local_logits,
          grad_transition_context,
          chunk_grad_initial,
          grad_transition_per_batch,
          grad_transition_gate_per_batch);
      carry = chunk_grad_initial.contiguous();
    }
  });
  grad_initial_log_belief.copy_(carry);

  auto grad_transition_log_probs = grad_transition_per_batch.sum(0);
  auto grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
  return {
      grad_local_logits,
      grad_transition_log_probs,
      grad_transition_context,
      grad_initial_log_belief,
      grad_transition_gate,
  };
}

}  // namespace agent_kernel::world
