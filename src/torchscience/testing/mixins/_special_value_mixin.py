from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class SpecialValueMixin:
    """Mixin providing special value tests."""

    descriptor: "OperatorDescriptor"

    def test_special_values(self):
        """Test all configured special values."""
        if "test_special_values" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.special_values:
            inputs = tuple(
                torch.tensor([v], dtype=torch.float64) for v in spec.inputs
            )
            result = self.descriptor.func(*inputs)
            expected = torch.tensor([spec.expected], dtype=torch.float64)
            (
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=spec.rtol,
                    atol=spec.atol,
                ),
                f"Special value {spec.inputs} -> {spec.expected} failed: {spec.description}",
            )
