from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profit_pilot.utils.torch_device import select_torch_device


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        name: str = "Tesla P100-PCIE-16GB",
        capability: tuple[int, int] = (6, 0),
        arch_list: list[str] | None = None,
    ) -> None:
        self._available = available
        self._name = name
        self._capability = capability
        self._arch_list = arch_list or ["sm_70", "sm_75", "sm_80"]
        self.synchronized = False

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int = 0) -> str:
        return self._name

    def get_device_capability(self, index: int = 0) -> tuple[int, int]:
        return self._capability

    def get_arch_list(self) -> list[str]:
        return self._arch_list

    def synchronize(self) -> None:
        self.synchronized = True


class FakeTorch:
    def __init__(self, cuda: FakeCuda, *, zeros_raises: Exception | None = None) -> None:
        self.cuda = cuda
        self.version = type("Version", (), {"cuda": "12.8"})()
        self._zeros_raises = zeros_raises

    def zeros(self, *args, **kwargs):
        if self._zeros_raises is not None:
            raise self._zeros_raises
        return [0.0]


def test_select_torch_device_falls_back_when_cuda_is_unavailable() -> None:
    selection = select_torch_device(FakeTorch(FakeCuda(available=False)))

    assert selection.device == "cpu"
    assert "not available" in selection.reason


def test_select_torch_device_falls_back_when_gpu_arch_is_not_supported() -> None:
    selection = select_torch_device(
        FakeTorch(
            FakeCuda(
                name="Tesla P100-PCIE-16GB",
                capability=(6, 0),
                arch_list=["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
            )
        )
    )

    assert selection.device == "cpu"
    assert "sm_60" in selection.reason
    assert "does not support" in selection.reason


def test_select_torch_device_uses_cuda_after_successful_smoke_test() -> None:
    fake_cuda = FakeCuda(
        name="Tesla T4",
        capability=(7, 5),
        arch_list=["sm_70", "sm_75", "sm_80"],
    )

    selection = select_torch_device(FakeTorch(fake_cuda))

    assert selection.device == "cuda"
    assert selection.reason == "CUDA smoke test passed."
    assert fake_cuda.synchronized is True


def test_select_torch_device_falls_back_when_cuda_kernel_smoke_test_fails() -> None:
    selection = select_torch_device(
        FakeTorch(
            FakeCuda(
                name="Tesla T4",
                capability=(7, 5),
                arch_list=["sm_70", "sm_75", "sm_80"],
            ),
            zeros_raises=RuntimeError("CUDA error: no kernel image is available for execution on the device"),
        )
    )

    assert selection.device == "cpu"
    assert "CUDA smoke test failed" in selection.reason
    assert "no kernel image" in selection.reason
