"""PyTorch reference implementation for RMSNorm — used as correctness ground truth."""

import torch


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Pure PyTorch RMSNorm. Runs in float32 for maximum precision."""
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f32 * rms * weight.float()).to(x.dtype)


def make_test_inputs(M: int = 2048, N: int = 4096, device: str = "cuda"):
    """Deterministic test tensors for reproducible verification."""
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.float16, device=device)
    weight = torch.ones(N, dtype=torch.float16, device=device)
    return {"x": x, "weight": weight}
