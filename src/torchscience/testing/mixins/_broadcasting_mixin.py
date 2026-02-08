from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class BroadcastingMixin:
    """Mixin providing broadcasting tests for n-ary operators."""

    descriptor: "OperatorDescriptor"

    def test_broadcast_scalar_with_tensor(self):
        """Test broadcasting scalar with tensor (first input scalar)."""
        if "test_broadcast_scalar_with_tensor" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires arity >= 2")

        # First input is scalar, rest are tensors
        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            shape = () if i == 0 else (5,)
            tensor = self._make_input_for_spec(
                spec, torch.float64, "cpu", shape
            )
            inputs.append(tensor)

        result = self.descriptor.func(*inputs)
        assert result.shape == (5,)

    def test_broadcast_tensor_with_scalar(self):
        """Test broadcasting tensor with scalar (last input scalar)."""
        if "test_broadcast_tensor_with_scalar" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires arity >= 2")

        # Last input is scalar, rest are tensors
        inputs = []
        n = len(self.descriptor.input_specs)
        for i, spec in enumerate(self.descriptor.input_specs):
            shape = () if i == n - 1 else (5,)
            tensor = self._make_input_for_spec(
                spec, torch.float64, "cpu", shape
            )
            inputs.append(tensor)

        result = self.descriptor.func(*inputs)
        assert result.shape == (5,)

    def test_broadcast_different_shapes(self):
        """Test broadcasting with different shapes."""
        if "test_broadcast_different_shapes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires arity >= 2")

        # Create inputs with shapes that broadcast: (3, 1), (1, 4), ...
        inputs = []
        shapes = [(3, 1), (1, 4), (1, 1), (1, 1)]  # Up to 4 inputs
        expected_shape = (3, 4)

        for i, spec in enumerate(self.descriptor.input_specs):
            shape = shapes[i] if i < len(shapes) else (1, 1)
            tensor = self._make_input_for_spec(
                spec, torch.float64, "cpu", shape
            )
            inputs.append(tensor)

        result = self.descriptor.func(*inputs)
        assert result.shape == expected_shape

    def test_broadcast_batch_dimensions(self):
        """Test broadcasting with batch dimensions."""
        if "test_broadcast_batch_dimensions" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires arity >= 2")

        # First input has batch dim, others are unbatched
        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            shape = (4, 3) if i == 0 else (3,)
            tensor = self._make_input_for_spec(
                spec, torch.float64, "cpu", shape
            )
            inputs.append(tensor)

        result = self.descriptor.func(*inputs)
        assert result.shape == (4, 3)

    def test_broadcast_all_different_shapes(self):
        """Test broadcasting where all inputs have different shapes."""
        if "test_broadcast_all_different_shapes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires arity >= 2")

        # Each input has a unique expandable shape
        shape_patterns = [
            (2, 1, 1),
            (1, 3, 1),
            (1, 1, 4),
            (1, 1, 1),
        ]

        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            shape = shape_patterns[i] if i < len(shape_patterns) else (1, 1, 1)
            tensor = self._make_input_for_spec(
                spec, torch.float64, "cpu", shape
            )
            inputs.append(tensor)

        result = self.descriptor.func(*inputs)

        # Expected shape depends on arity - only dimensions used by inputs contribute
        arity = self.descriptor.arity
        if arity == 2:
            expected_shape = (2, 3, 1)  # Only first two shape patterns used
        elif arity == 3:
            expected_shape = (2, 3, 4)  # First three shape patterns used
        else:
            expected_shape = (2, 3, 4)  # Four or more inputs

        assert result.shape == expected_shape
