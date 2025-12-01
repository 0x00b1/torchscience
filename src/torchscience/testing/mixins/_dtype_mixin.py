from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class DtypeMixin:
    """Mixin providing dtype handling tests."""

    descriptor: "OperatorDescriptor"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_dtypes(self, dtype: torch.dtype):
        """Test operator with real floating-point dtypes."""
        if "test_real_dtypes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype: torch.dtype):
        """Test operator with complex dtypes."""
        if "test_complex_dtypes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        if "test_dtype_preservation" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for dtype in [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]:
            inputs = self._make_standard_inputs(dtype=dtype)
            result = self.descriptor.func(*inputs)
            assert result.dtype == dtype, (
                f"Expected {dtype}, got {result.dtype}"
            )
