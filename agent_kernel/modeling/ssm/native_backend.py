from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NativeSSMBackendStatus:
    available: bool
    compiled_extension: bool
    repo_root: str
    extension_name: str
    detail: str = ""


def native_ssm_repo_root() -> Path:
    return Path(__file__).resolve().parent


def native_selective_scan_extension_name() -> str:
    return "agent_kernel_selective_scan_cuda"


def recommended_native_build_command() -> str:
    return "python -m agent_kernel.modeling.ssm.kernels.build"


def native_ssm_backend_status() -> NativeSSMBackendStatus:
    from .kernels.cuda_selective_scan import cuda_selective_scan_status

    cuda_status = cuda_selective_scan_status()
    if cuda_status.available:
        return NativeSSMBackendStatus(
            available=True,
            compiled_extension=True,
            repo_root=str(native_ssm_repo_root()),
            extension_name=native_selective_scan_extension_name(),
            detail="",
        )
    return NativeSSMBackendStatus(
        available=False,
        compiled_extension=bool(cuda_status.compiled_extension),
        repo_root=str(native_ssm_repo_root()),
        extension_name=native_selective_scan_extension_name(),
        detail=(
            cuda_status.detail
            or f"native selective-scan CUDA extension is unavailable; build with {recommended_native_build_command()}"
        ),
    )
