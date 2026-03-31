#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace agent_kernel::ssm {

#define CUDA_KERNEL_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()

template <typename acc_t>
__device__ __forceinline__ acc_t sigmoid_acc(acc_t x) {
  return acc_t(1) / (acc_t(1) + exp(-x));
}

template <typename acc_t>
__device__ __forceinline__ acc_t softplus_acc(acc_t x) {
  if (x > acc_t(20)) {
    return x;
  }
  if (x < acc_t(-20)) {
    return exp(x);
  }
  return log1p(exp(x));
}

template <typename acc_t>
__device__ __forceinline__ acc_t silu_acc(acc_t x) {
  return x * sigmoid_acc(x);
}

template <typename acc_t>
__device__ __forceinline__ acc_t silu_grad_acc(acc_t x) {
  const acc_t sig = sigmoid_acc(x);
  return sig * (acc_t(1) + x * (acc_t(1) - sig));
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

inline int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename scalar_t>
__global__ void selective_scan_forward_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ delta_bias,
    bool has_D,
    bool has_z,
    bool has_delta_bias,
    bool delta_softplus,
    int batch,
    int dim,
    int length,
    int d_state,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ last_state,
    scalar_t* __restrict__ x_hist,
    scalar_t* __restrict__ y_base) {
  const int line = blockIdx.x;
  const int total = batch * dim;
  if (line >= total) {
    return;
  }

  using acc_t = at::acc_type<scalar_t, true>;
  const int tid = threadIdx.x;
  const int d = line % dim;
  const acc_t d_skip = has_D ? static_cast<acc_t>(D[d]) : acc_t(0);
  const acc_t d_bias = has_delta_bias ? static_cast<acc_t>(delta_bias[d]) : acc_t(0);
  extern __shared__ unsigned char shared_raw[];
  auto* scratch = reinterpret_cast<acc_t*>(shared_raw);

  const scalar_t* u_ptr = u + line * length;
  const scalar_t* delta_ptr = delta + line * length;
  const scalar_t* B_ptr = B + line * d_state * length;
  const scalar_t* C_ptr = C + line * d_state * length;
  const scalar_t* z_ptr = has_z ? z + line * length : nullptr;
  const scalar_t* A_ptr = A + d * d_state;

  scalar_t* out_ptr = out + line * length;
  scalar_t* y_ptr = y_base + line * length;
  scalar_t* x_ptr = x_hist + line * length * d_state;
  scalar_t* last_ptr = last_state + line * d_state;

  for (int t = 0; t < length; ++t) {
    const acc_t raw_dt = static_cast<acc_t>(delta_ptr[t]) + d_bias;
    const acc_t dt = delta_softplus ? softplus_acc(raw_dt) : raw_dt;
    const acc_t u_t = static_cast<acc_t>(u_ptr[t]);
    acc_t y_partial = acc_t(0);
    for (int n = tid; n < d_state; n += blockDim.x) {
      const acc_t x_prev = (t == 0) ? acc_t(0) : static_cast<acc_t>(x_ptr[(t - 1) * d_state + n]);
      const acc_t a = static_cast<acc_t>(A_ptr[n]);
      const acc_t deltaA = exp(dt * a);
      const acc_t b_t = static_cast<acc_t>(B_ptr[n * length + t]);
      const acc_t c_t = static_cast<acc_t>(C_ptr[n * length + t]);
      const acc_t x_t = deltaA * x_prev + dt * b_t * u_t;
      x_ptr[t * d_state + n] = static_cast<scalar_t>(x_t);
      y_partial += x_t * c_t;
    }
    const acc_t y_t = block_reduce_sum(y_partial, scratch);
    if (tid == 0) {
      y_ptr[t] = static_cast<scalar_t>(y_t);
      acc_t output = y_t + d_skip * u_t;
      if (has_z) {
        output *= silu_acc(static_cast<acc_t>(z_ptr[t]));
      }
      out_ptr[t] = static_cast<scalar_t>(output);
    }
    __syncthreads();
  }

  for (int n = tid; n < d_state; n += blockDim.x) {
    last_ptr[n] = x_ptr[(length - 1) * d_state + n];
  }
}

template <typename scalar_t>
__global__ void selective_scan_backward_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ dlast_state,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ x_hist,
    const scalar_t* __restrict__ y_base,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ delta_bias,
    bool has_D,
    bool has_z,
    bool has_delta_bias,
    bool delta_softplus,
    int batch,
    int dim,
    int length,
    int d_state,
    scalar_t* __restrict__ du,
    scalar_t* __restrict__ ddelta,
    scalar_t* __restrict__ dA_full,
    scalar_t* __restrict__ dB,
    scalar_t* __restrict__ dC,
    scalar_t* __restrict__ dD_full,
    scalar_t* __restrict__ dz,
    scalar_t* __restrict__ ddelta_bias_full,
    scalar_t* __restrict__ grad_x_work) {
  const int line = blockIdx.x;
  const int total = batch * dim;
  if (line >= total) {
    return;
  }

  using acc_t = at::acc_type<scalar_t, true>;
  const int tid = threadIdx.x;
  const int d = line % dim;
  const acc_t d_skip = has_D ? static_cast<acc_t>(D[d]) : acc_t(0);
  const acc_t d_bias = has_delta_bias ? static_cast<acc_t>(delta_bias[d]) : acc_t(0);
  extern __shared__ unsigned char shared_raw[];
  auto* scratch = reinterpret_cast<acc_t*>(shared_raw);
  __shared__ acc_t g_y_shared;

  const scalar_t* dout_ptr = dout + line * length;
  const scalar_t* dlast_ptr = dlast_state + line * d_state;
  const scalar_t* u_ptr = u + line * length;
  const scalar_t* delta_ptr = delta + line * length;
  const scalar_t* B_ptr = B + line * d_state * length;
  const scalar_t* C_ptr = C + line * d_state * length;
  const scalar_t* x_ptr = x_hist + line * length * d_state;
  const scalar_t* y_ptr = y_base + line * length;
  const scalar_t* z_ptr = has_z ? z + line * length : nullptr;
  const scalar_t* A_ptr = A + d * d_state;

  scalar_t* du_ptr = du + line * length;
  scalar_t* ddelta_ptr = ddelta + line * length;
  scalar_t* dA_ptr = dA_full + line * d_state;
  scalar_t* dB_ptr = dB + line * d_state * length;
  scalar_t* dC_ptr = dC + line * d_state * length;
  scalar_t* dD_ptr = dD_full + line;
  scalar_t* dz_ptr = has_z ? dz + line * length : nullptr;
  scalar_t* dbias_ptr = has_delta_bias ? ddelta_bias_full + line : nullptr;
  scalar_t* grad_x_ptr = grad_x_work + line * d_state;

  for (int n = tid; n < d_state; n += blockDim.x) {
    grad_x_ptr[n] = dlast_ptr[n];
    dA_ptr[n] = scalar_t(0);
  }
  if (tid == 0 && has_D) {
    dD_ptr[0] = scalar_t(0);
  }
  if (tid == 0 && has_delta_bias) {
    dbias_ptr[0] = scalar_t(0);
  }
  __syncthreads();

  for (int t = length - 1; t >= 0; --t) {
    const acc_t raw_dt = static_cast<acc_t>(delta_ptr[t]) + d_bias;
    const acc_t dt = delta_softplus ? softplus_acc(raw_dt) : raw_dt;
    const acc_t dt_scale = delta_softplus ? sigmoid_acc(raw_dt) : acc_t(1);
    const acc_t u_t = static_cast<acc_t>(u_ptr[t]);
    if (tid == 0) {
      const acc_t y_pre = static_cast<acc_t>(y_ptr[t]) + d_skip * u_t;
      acc_t g_y = static_cast<acc_t>(dout_ptr[t]);
      if (has_z) {
        const acc_t z_t = static_cast<acc_t>(z_ptr[t]);
        g_y *= silu_acc(z_t);
        dz_ptr[t] = static_cast<scalar_t>(static_cast<acc_t>(dout_ptr[t]) * y_pre * silu_grad_acc(z_t));
      }
      g_y_shared = g_y;
    }
    __syncthreads();
    const acc_t g_y = g_y_shared;

    acc_t du_partial = (has_D && tid == 0) ? g_y * d_skip : acc_t(0);
    acc_t dD_partial = (has_D && tid == 0) ? g_y * u_t : acc_t(0);
    acc_t ddelta_partial = acc_t(0);

    for (int n = tid; n < d_state; n += blockDim.x) {
      const acc_t x_t = static_cast<acc_t>(x_ptr[t * d_state + n]);
      const acc_t x_prev = (t == 0) ? acc_t(0) : static_cast<acc_t>(x_ptr[(t - 1) * d_state + n]);
      const acc_t a = static_cast<acc_t>(A_ptr[n]);
      const acc_t b_t = static_cast<acc_t>(B_ptr[n * length + t]);
      const acc_t c_t = static_cast<acc_t>(C_ptr[n * length + t]);
      const acc_t deltaA = exp(dt * a);
      const acc_t grad_x = static_cast<acc_t>(grad_x_ptr[n]) + g_y * c_t;

      dC_ptr[n * length + t] = static_cast<scalar_t>(g_y * x_t);
      dB_ptr[n * length + t] = static_cast<scalar_t>(grad_x * dt * u_t);
      dA_ptr[n] = static_cast<scalar_t>(static_cast<acc_t>(dA_ptr[n]) + grad_x * x_prev * dt * deltaA);

      du_partial += grad_x * dt * b_t;
      ddelta_partial += grad_x * (x_prev * a * deltaA + b_t * u_t);
      grad_x_ptr[n] = static_cast<scalar_t>(grad_x * deltaA);
    }

    const acc_t du_sum = block_reduce_sum(du_partial, scratch);
    const acc_t ddelta_sum = block_reduce_sum(ddelta_partial, scratch);
    const acc_t dD_sum = has_D ? block_reduce_sum(dD_partial, scratch) : acc_t(0);
    if (tid == 0) {
      du_ptr[t] = static_cast<scalar_t>(du_sum);
      ddelta_ptr[t] = static_cast<scalar_t>(ddelta_sum * dt_scale);
      if (has_D) {
        dD_ptr[0] = static_cast<scalar_t>(static_cast<acc_t>(dD_ptr[0]) + dD_sum);
      }
      if (has_delta_bias) {
        dbias_ptr[0] = static_cast<scalar_t>(static_cast<acc_t>(dbias_ptr[0]) + ddelta_sum * dt_scale);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t>
void launch_forward(
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
    torch::Tensor& y_base) {
  const int batch = u.size(0);
  const int dim = u.size(1);
  const int length = u.size(2);
  const int d_state = A.size(1);
  const int total = batch * dim;
  const int threads = std::min(256, next_power_of_two(std::max(1, d_state)));
  const int blocks = total;
  const auto shared_bytes = static_cast<size_t>(threads * sizeof(typename at::acc_type<scalar_t, true>));

  selective_scan_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getDefaultCUDAStream()>>>(
      u.data_ptr<scalar_t>(),
      delta.data_ptr<scalar_t>(),
      A.data_ptr<scalar_t>(),
      B.data_ptr<scalar_t>(),
      C.data_ptr<scalar_t>(),
      D.has_value() ? D->data_ptr<scalar_t>() : nullptr,
      z.has_value() ? z->data_ptr<scalar_t>() : nullptr,
      delta_bias.has_value() ? delta_bias->data_ptr<scalar_t>() : nullptr,
      D.has_value(),
      z.has_value(),
      delta_bias.has_value(),
      delta_softplus,
      batch,
      dim,
      length,
      d_state,
      out.data_ptr<scalar_t>(),
      last_state.data_ptr<scalar_t>(),
      x_hist.data_ptr<scalar_t>(),
      y_base.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

template <typename scalar_t>
void launch_backward(
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
    torch::Tensor& ddelta_bias_full) {
  const int batch = u.size(0);
  const int dim = u.size(1);
  const int length = u.size(2);
  const int d_state = A.size(1);
  const int total = batch * dim;
  const int threads = std::min(256, next_power_of_two(std::max(1, d_state)));
  const int blocks = total;
  const auto shared_bytes = static_cast<size_t>(threads * sizeof(typename at::acc_type<scalar_t, true>));
  auto grad_x_work = torch::zeros({batch, dim, d_state}, u.options());

  selective_scan_backward_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getDefaultCUDAStream()>>>(
      dout.data_ptr<scalar_t>(),
      dlast_state.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(),
      delta.data_ptr<scalar_t>(),
      A.data_ptr<scalar_t>(),
      B.data_ptr<scalar_t>(),
      C.data_ptr<scalar_t>(),
      x_hist.data_ptr<scalar_t>(),
      y_base.data_ptr<scalar_t>(),
      D.has_value() ? D->data_ptr<scalar_t>() : nullptr,
      z.has_value() ? z->data_ptr<scalar_t>() : nullptr,
      delta_bias.has_value() ? delta_bias->data_ptr<scalar_t>() : nullptr,
      D.has_value(),
      z.has_value(),
      delta_bias.has_value(),
      delta_softplus,
      batch,
      dim,
      length,
      d_state,
      du.data_ptr<scalar_t>(),
      ddelta.data_ptr<scalar_t>(),
      dA_full.data_ptr<scalar_t>(),
      dB.data_ptr<scalar_t>(),
      dC.data_ptr<scalar_t>(),
      dD_full.numel() > 0 ? dD_full.data_ptr<scalar_t>() : nullptr,
      dz.numel() > 0 ? dz.data_ptr<scalar_t>() : nullptr,
      ddelta_bias_full.numel() > 0 ? ddelta_bias_full.data_ptr<scalar_t>() : nullptr,
      grad_x_work.data_ptr<scalar_t>());
  CUDA_KERNEL_CHECK();
}

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
    torch::Tensor& y_base) {
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "agent_kernel_selective_scan_forward", [&] {
    launch_forward<scalar_t>(u, delta, A, B, C, D, z, delta_bias, delta_softplus, out, last_state, x_hist, y_base);
  });
}

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
    torch::Tensor& ddelta_bias_full) {
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "agent_kernel_selective_scan_backward", [&] {
    launch_backward<scalar_t>(
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
  });
}

}  // namespace agent_kernel::ssm
