# SSM Kernels

This package is the CUDA-kernel boundary for state-space modeling work.

Use it for:

- custom CUDA kernels
- Triton kernels
- compiled native selective-scan adapters
- kernel build helpers and metadata
- explicit wrappers around native kernels owned by `agent_kernel`

Do not use it for:

- task execution
- verifier logic
- unattended trust logic
- policy routing in the live kernel

Design rule:

- CUDA-first implementations are acceptable here because the modeling stack is
  expected to run on GPU-capable machines.
- Python or PyTorch reference paths should still exist one layer above this
  package so development, testing, and reduced environments remain possible.

The intended flow is:

- `agent_kernel/modeling/ssm/kernels/`: low-level fused kernels
- `agent_kernel/modeling/ssm/`: safe wrappers and fallbacks
- future TOLBERT-improved models: call those wrappers

`other_repos/mamba` is reference material only. The implementation surface here
is native to `agent_kernel`.
