from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class DeviceMixin:
    """Mixin providing device (CPU/CUDA) tests."""

    descriptor: "OperatorDescriptor"

    def test_cpu_device(self):
        """Test operator works on CPU."""
        if "test_cpu_device" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(device="cpu")
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cpu"
        assert torch.isfinite(result).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test operator works on CUDA."""
        if "test_cuda_device" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(device="cuda")
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cuda"

        # Compare against CPU
        cpu_inputs = tuple(t.cpu() for t in inputs)
        expected = self.descriptor.func(*cpu_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(
            result.cpu(), expected, rtol=rtol, atol=atol
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_complex(self):
        """Test complex dtypes on CUDA."""
        if "test_cuda_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(
            dtype=torch.complex128, device="cuda"
        )
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cuda"
        assert result.dtype == torch.complex128
