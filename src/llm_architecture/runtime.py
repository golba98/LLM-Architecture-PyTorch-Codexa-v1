"""Device and arithmetic policies usable without importing the trainer."""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import torch

@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved precision behavior for a device."""

    name: str
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool

def resolve_device(requested: str) -> torch.device:
    """Resolve auto, CPU, or CUDA without an implicit CUDA fallback."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "device must be one of auto, cpu, cuda; "
            f"got {requested!r}."
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable.")
    return torch.device(requested)

def resolve_precision(requested: str, device: torch.device) -> PrecisionPolicy:
    """Validate and resolve the requested arithmetic precision."""

    if requested not in {"fp32", "fp16", "bf16"}:
        raise ValueError(
            "precision must be one of fp32, fp16, bf16; "
            f"got {requested!r}."
        )
    if device.type == "cpu":
        if requested != "fp32":
            raise ValueError(
                f"Precision {requested} is not supported on CPU; use fp32."
            )
        return PrecisionPolicy("fp32", None, False)
    if device.type != "cuda":
        raise ValueError(f"Unsupported device type {device.type!r}.")
    if requested == "fp32":
        return PrecisionPolicy("fp32", None, False)
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 was requested but is unsupported.")
        return PrecisionPolicy("bf16", torch.bfloat16, False)
    return PrecisionPolicy("fp16", torch.float16, True)

def autocast_dtype(policy: PrecisionPolicy) -> torch.dtype | None:
    return policy.autocast_dtype

def requires_grad_scaler(policy: PrecisionPolicy) -> bool:
    return policy.uses_grad_scaler

def autocast_context(
    device: torch.device,
    policy: PrecisionPolicy,
) -> Any:
    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=policy.autocast_dtype,
    )

def create_grad_scaler(
    device: torch.device,
    policy: PrecisionPolicy,
) -> torch.amp.GradScaler | None:
    if not policy.uses_grad_scaler:
        return None
    return torch.amp.GradScaler(device.type, enabled=True)
