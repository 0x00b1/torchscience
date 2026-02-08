from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class AutocastMixin:
    """Mixin providing autocast (AMP) tests."""

    descriptor: "OperatorDescriptor"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda_float16(self):
        """Test CUDA autocast with float16."""
        if "test_autocast_cuda_float16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.float16

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda_bfloat16(self):
        """Test CUDA autocast with bfloat16."""
        if "test_autocast_cuda_bfloat16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.bfloat16

    def test_autocast_cpu_bfloat16(self):
        """Test CPU autocast with bfloat16."""
        if "test_autocast_cpu_bfloat16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cpu")

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.bfloat16
