from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class LowPrecisionMixin:
    """Mixin providing float16/bfloat16 tests."""

    descriptor: "OperatorDescriptor"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype: torch.dtype):
        """Test forward pass with low-precision dtypes."""
        if "test_low_precision_forward" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

        # Compare against float32 reference
        fp32_inputs = tuple(t.to(torch.float32) for t in inputs)
        expected = self.descriptor.func(*fp32_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(dtype)
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtype_preservation(self, dtype: torch.dtype):
        """Test that low-precision dtype is preserved."""
        if (
            "test_low_precision_dtype_preservation"
            in self.descriptor.skip_tests
        ):
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_cuda(self, dtype: torch.dtype):
        """Test low-precision on CUDA."""
        if "test_low_precision_cuda" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype, device="cuda")
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype
        assert result.device.type == "cuda"
