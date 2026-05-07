from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchDeviceSelection:
    device: str
    reason: str
    gpu_name: str | None = None
    cuda_version: str | None = None
    capability: str | None = None
    arch_list: tuple[str, ...] = ()


def _safe_call(func: Any, *args: Any) -> Any:
    try:
        return func(*args)
    except Exception:
        return None


def select_torch_device(torch_module: Any, preferred: str = "auto") -> TorchDeviceSelection:
    """Choose a Torch device only after checking that CUDA can run kernels."""

    normalized = (preferred or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("preferred must be one of: auto, cuda, cpu")

    if normalized == "cpu":
        return TorchDeviceSelection(device="cpu", reason="CPU requested by configuration.")

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return TorchDeviceSelection(device="cpu", reason="CUDA is not available.")

    gpu_name = _safe_call(cuda.get_device_name, 0) or "unknown GPU"
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)

    capability = None
    capability_tuple = _safe_call(cuda.get_device_capability, 0)
    if capability_tuple and len(capability_tuple) >= 2:
        capability = f"sm_{capability_tuple[0]}{capability_tuple[1]}"

    arch_list = tuple(_safe_call(cuda.get_arch_list) or ())
    if capability and arch_list and capability not in arch_list:
        supported = ", ".join(arch_list)
        return TorchDeviceSelection(
            device="cpu",
            reason=(
                f"GPU {gpu_name} is capability {capability}, but this PyTorch "
                f"build does not support {capability} (supports {supported})."
            ),
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            capability=capability,
            arch_list=arch_list,
        )

    try:
        torch_module.zeros(1, device="cuda")
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize()
    except Exception as exc:
        return TorchDeviceSelection(
            device="cpu",
            reason=f"CUDA smoke test failed on {gpu_name}: {exc}",
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            capability=capability,
            arch_list=arch_list,
        )

    return TorchDeviceSelection(
        device="cuda",
        reason="CUDA smoke test passed.",
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        capability=capability,
        arch_list=arch_list,
    )
