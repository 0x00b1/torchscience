from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class NanInfMixin:
    """Mixin providing NaN/Inf propagation tests for n-ary operators."""

    descriptor: "OperatorDescriptor"

    def _make_nan_test_inputs(
        self, nan_position: int = 0
    ) -> list[torch.Tensor]:
        """Create test inputs with NaN in the specified input position."""
        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            if i == nan_position:
                # This input gets NaN
                x = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)
            else:
                # Other inputs get normal values from their valid range
                low, high = spec.default_real_range
                x = torch.tensor(
                    [
                        low + (high - low) * 0.3,
                        low + (high - low) * 0.5,
                        low + (high - low) * 0.7,
                    ],
                    dtype=torch.float64,
                )
            inputs.append(x)
        return inputs

    def _make_inf_test_inputs(
        self, inf_position: int = 0
    ) -> list[torch.Tensor]:
        """Create test inputs with Inf in the specified input position."""
        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            if i == inf_position:
                # This input gets Inf values
                x = torch.tensor(
                    [1.0, float("inf"), float("-inf")], dtype=torch.float64
                )
            else:
                # Other inputs get normal values from their valid range
                low, high = spec.default_real_range
                x = torch.tensor(
                    [
                        low + (high - low) * 0.3,
                        low + (high - low) * 0.5,
                        low + (high - low) * 0.7,
                    ],
                    dtype=torch.float64,
                )
            inputs.append(x)
        return inputs

    def test_nan_propagation(self):
        """Test that NaN inputs produce NaN outputs."""
        if "test_nan_propagation" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        # Test NaN in the first input
        inputs = self._make_nan_test_inputs(nan_position=0)
        result = self.descriptor.func(*inputs)
        assert torch.isnan(result[1]), "NaN in first input should propagate"

    def test_nan_propagation_all_inputs(self):
        """Test NaN propagation from each input position."""
        if "test_nan_propagation_all_inputs" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for i in range(len(self.descriptor.input_specs)):
            inputs = self._make_nan_test_inputs(nan_position=i)
            result = self.descriptor.func(*inputs)
            assert torch.isnan(result[1]), (
                f"NaN in input {i} should propagate to output"
            )

    def test_inf_handling(self):
        """Test behavior with infinite inputs."""
        if "test_inf_handling" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        # Test Inf in the first input
        inputs = self._make_inf_test_inputs(inf_position=0)
        result = self.descriptor.func(*inputs)

        # Just verify it doesn't crash - specific behavior depends on function
        assert result.shape == (3,)

    def test_inf_handling_all_inputs(self):
        """Test Inf handling from each input position."""
        if "test_inf_handling_all_inputs" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for i in range(len(self.descriptor.input_specs)):
            inputs = self._make_inf_test_inputs(inf_position=i)
            result = self.descriptor.func(*inputs)
            # Just verify it doesn't crash
            assert result.shape == (3,), f"Inf in input {i} should not crash"
