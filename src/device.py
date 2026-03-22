"""Device selection helpers shared across training and evaluation entrypoints."""

from __future__ import annotations

import torch


AUTO_DEVICE = "auto"
AUTO_DEVICE_HELP = "Device to use. Default 'auto' resolves cuda -> mps -> cpu."


def is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(
        mps_backend is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def default_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    else:
        raw = str(device).strip().lower() if device is not None else AUTO_DEVICE
        resolved = torch.device(default_device_type() if raw in {"", AUTO_DEVICE} else raw)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    if resolved.type == "mps" and not is_mps_available():
        raise ValueError("MPS was requested, but no MPS device is available.")
    return resolved


def resolve_device_name(device: str | torch.device | None = None) -> str:
    return str(resolve_torch_device(device))
