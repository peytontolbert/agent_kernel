# State-Space Modeling

This package holds state-space modeling primitives for the future improved
TOLBERT family.

Current scope:

- selective-scan integration boundary
- native backend contract owned by `agent_kernel`
- custom CUDA-kernel boundary under `kernels/`
- pure PyTorch selective-scan fallback for development and tests
- explicit CUDA selective-scan status and metadata reporting

Intended future scope:

- TOLBERT-improved SSM blocks
- learned world/transition state modules
- long-horizon latent-memory components
- liftoff-gated checkpoint evaluation

This code is intentionally not wired into the live agent runtime yet. It is a
modeling surface under `agent_kernel/modeling/` for seed-to-liftoff work.

CUDA is an expected target for this package. Low-level custom kernels should
live under [`kernels/`](/data/agentkernel/agent_kernel/modeling/ssm/kernels),
with wrapper and fallback logic kept in the parent package.

The current native selective-scan kernel is validated for `float32` and
`float64`. Reduced-precision CUDA paths should not be enabled until recurrence
state is accumulated in a numerically stable wider format.

Implicit CUDA builds are gated by the retained `tolbert_model_bundle`
`build_policy`. Until the retained artifact says the supervised and synthetic
dataset thresholds are met, the runtime wrapper will refuse opportunistic
kernel autobuild and leave compilation to an explicit operator action.

`other_repos/mamba` can still be used as a reference repository, but this
package should not import `mamba_ssm` directly.
