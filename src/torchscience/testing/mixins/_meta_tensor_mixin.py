from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class MetaTensorMixin:
    """Mixin providing meta tensor tests."""

    descriptor: "OperatorDescriptor"

    def test_meta_tensor_shape_inference(self):
        """Test shape inference with meta tensors."""
        if "test_meta_tensor_shape_inference" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_meta:
            pytest.skip("Operator does not support meta tensors")

        inputs = []
        for spec in self.descriptor.input_specs:
            inputs.append(torch.empty(5, dtype=torch.float64, device="meta"))

        result = self.descriptor.func(*inputs)
        assert result.device.type == "meta"
        assert result.shape == (5,)

    def test_meta_tensor_large_shape(self):
        """Test meta tensors with large shapes (no memory allocation)."""
        if "test_meta_tensor_large_shape" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_meta:
            pytest.skip("Operator does not support meta tensors")

        inputs = []
        for spec in self.descriptor.input_specs:
            inputs.append(
                torch.empty(10000, 10000, dtype=torch.float64, device="meta")
            )

        result = self.descriptor.func(*inputs)
        assert result.device.type == "meta"
        assert result.shape == (10000, 10000)
