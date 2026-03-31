import importlib
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch.nn.functional")
import torch

from agent_kernel.modeling.ssm import selective_scan, selective_scan_ref
from agent_kernel.modeling.ssm.native_backend import native_ssm_backend_status
from agent_kernel.modeling.ssm.kernels import (
    cuda_selective_scan_status,
    selective_scan_cuda_fn,
    selective_scan_cuda_metadata,
)

selective_scan_module = importlib.import_module("agent_kernel.modeling.ssm.selective_scan")
cuda_selective_scan_module = importlib.import_module("agent_kernel.modeling.ssm.kernels.cuda_selective_scan")


def test_selective_scan_ref_matches_manual_recurrence():
    u = torch.tensor([[[1.0, 2.0, 3.0]]])
    delta = torch.tensor([[[1.0, 1.0, 1.0]]])
    A = torch.tensor([[-0.5]])
    B = torch.tensor([[[0.25, 0.25, 0.25]]])
    C = torch.tensor([[[2.0, 2.0, 2.0]]])
    D = torch.tensor([0.1])
    z = torch.tensor([[[0.0, 0.5, 1.0]]])
    delta_bias = torch.tensor([0.2])

    result = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )

    x = 0.0
    expected = []
    dt = torch.nn.functional.softplus(delta + delta_bias.view(1, 1, 1))
    for index in range(u.shape[-1]):
        delta_t = dt[0, 0, index].item()
        x = torch.exp(delta_t * A[0, 0]).item() * x + delta_t * B[0, 0, index].item() * u[0, 0, index].item()
        y = x * C[0, 0, index].item() + D[0].item() * u[0, 0, index].item()
        y *= torch.nn.functional.silu(z[0, 0, index]).item()
        expected.append(y)

    assert torch.allclose(result.output, torch.tensor([expected], dtype=u.dtype).unsqueeze(0), atol=1e-6)
    assert result.last_state is not None
    assert result.last_state.shape == (1, 1, 1)


def test_selective_scan_prefers_python_ref_when_requested():
    u = torch.randn(2, 3, 4)
    delta = torch.randn(2, 3, 4)
    A = torch.randn(3, 5)
    B = torch.randn(2, 5, 4)
    C = torch.randn(2, 5, 4)

    result = selective_scan(u, delta, A, B, C, prefer_python_ref=True)

    assert result.backend == "python_ref"
    assert result.output.shape == (2, 3, 4)


def test_selective_scan_ref_accepts_initial_state_for_incremental_equivalence():
    torch.manual_seed(0)
    u = torch.randn(2, 3, 5)
    delta = torch.randn(2, 3, 5)
    A = torch.randn(3, 4)
    B = torch.randn(2, 4, 5)
    C = torch.randn(2, 4, 5)

    full = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        return_last_state=True,
    )
    first = selective_scan_ref(
        u[:, :, :2],
        delta[:, :, :2],
        A,
        B[:, :, :2],
        C[:, :, :2],
        return_last_state=True,
    )
    second = selective_scan_ref(
        u[:, :, 2:],
        delta[:, :, 2:],
        A,
        B[:, :, 2:],
        C[:, :, 2:],
        return_last_state=True,
        initial_state=first.last_state,
    )

    stitched = torch.cat([first.output, second.output], dim=-1)
    assert first.last_state is not None
    assert second.last_state is not None
    assert torch.allclose(stitched, full.output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(second.last_state, full.last_state, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selective_scan_cuda_missing_extension_fails_fast(monkeypatch):
    u = torch.randn(2, 3, 4, device="cuda")
    delta = torch.randn(2, 3, 4, device="cuda")
    A = torch.randn(3, 5, device="cuda")
    B = torch.randn(2, 5, 4, device="cuda")
    C = torch.randn(2, 5, 4, device="cuda")

    monkeypatch.setattr(
        selective_scan_module,
        "cuda_selective_scan_status",
        lambda: type("Status", (), {"available": False, "detail": "missing native extension"})(),
    )

    with pytest.raises(RuntimeError, match="missing native extension"):
        selective_scan(u, delta, A, B, C)


def test_native_ssm_backend_status_reports_expected_shape():
    status = native_ssm_backend_status()

    assert isinstance(status.available, bool)
    assert status.extension_name == "agent_kernel_selective_scan_cuda"


def test_cuda_selective_scan_status_reports_expected_fields():
    status = cuda_selective_scan_status()

    assert status.source == "native"
    assert isinstance(status.available, bool)
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.compiled_extension, bool)


def test_cuda_selective_scan_status_surfaces_import_failure_detail(monkeypatch):
    monkeypatch.setattr(cuda_selective_scan_module, "_LOADED_EXTENSION", None)
    monkeypatch.setattr(cuda_selective_scan_module, "_LOAD_ERROR_DETAIL", "synthetic import failure")
    monkeypatch.setattr(cuda_selective_scan_module, "_import_optional", lambda name: None)
    monkeypatch.setattr(cuda_selective_scan_module, "_autobuild_allowed", lambda: (False, ""))

    status = cuda_selective_scan_status()

    assert status.available is False
    assert "synthetic import failure" in status.detail


def test_cuda_selective_scan_metadata_has_expected_extension_name():
    metadata = selective_scan_cuda_metadata()

    assert metadata["compiled_extension_name"] == "agent_kernel_selective_scan_cuda"
    assert "agent_kernel/modeling/ssm/kernels/src/selective_scan.cpp" in metadata["compiled_extension_sources"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_selective_scan_matches_python_reference():
    status = cuda_selective_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native selective-scan extension is unavailable")

    torch.manual_seed(0)
    batch, dim, length, d_state = 2, 4, 5, 3
    u = torch.randn(batch, dim, length, device="cuda", requires_grad=True)
    delta = torch.randn(batch, dim, length, device="cuda", requires_grad=True)
    A = torch.randn(dim, d_state, device="cuda", requires_grad=True)
    B = torch.randn(batch, d_state, length, device="cuda", requires_grad=True)
    C = torch.randn(batch, d_state, length, device="cuda", requires_grad=True)
    D = torch.randn(dim, device="cuda", requires_grad=True)
    z = torch.randn(batch, dim, length, device="cuda", requires_grad=True)
    delta_bias = torch.randn(dim, device="cuda", requires_grad=True)

    out_cuda, last_cuda = selective_scan_cuda_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    ref = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )

    assert last_cuda is not None
    assert ref.last_state is not None
    assert torch.allclose(out_cuda, ref.output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(last_cuda, ref.last_state, atol=5e-3, rtol=1e-4)

    loss_cuda = out_cuda.sum() + last_cuda.sum()
    loss_ref = ref.output.sum() + ref.last_state.sum()
    loss_cuda.backward()
    grad_cuda = {
        "u": u.grad.detach().clone(),
        "delta": delta.grad.detach().clone(),
        "A": A.grad.detach().clone(),
        "B": B.grad.detach().clone(),
        "C": C.grad.detach().clone(),
        "D": D.grad.detach().clone(),
        "z": z.grad.detach().clone(),
        "delta_bias": delta_bias.grad.detach().clone(),
    }
    for tensor in (u, delta, A, B, C, D, z, delta_bias):
        tensor.grad.zero_()
    loss_ref.backward()
    grad_ref = {
        "u": u.grad.detach().clone(),
        "delta": delta.grad.detach().clone(),
        "A": A.grad.detach().clone(),
        "B": B.grad.detach().clone(),
        "C": C.grad.detach().clone(),
        "D": D.grad.detach().clone(),
        "z": z.grad.detach().clone(),
        "delta_bias": delta_bias.grad.detach().clone(),
    }

    for name in grad_cuda:
        assert torch.allclose(grad_cuda[name], grad_ref[name], atol=5e-2, rtol=1e-3), name
