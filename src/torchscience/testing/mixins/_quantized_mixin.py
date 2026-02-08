from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class QuantizedMixin:
    """Mixin providing quantized tensor tests for n-ary operators."""

    descriptor: "OperatorDescriptor"

    def _create_quantized_inputs(
        self, qtype: torch.dtype = torch.quint8
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Create quantized test inputs based on descriptor's input_specs.

        Returns:
            Tuple of (quantized_inputs, dequantized_inputs)
        """
        quantized_inputs = []
        dequantized_inputs = []

        for spec in self.descriptor.input_specs:
            # Generate values within the input's valid range
            low, high = spec.default_real_range

            # Clamp range for quantization (avoid extreme values)
            low = max(low, -10.0)
            high = min(high, 10.0)

            # Create float tensor with values in range
            values = torch.tensor(
                [
                    low + (high - low) * 0.2,
                    low + (high - low) * 0.4,
                    low + (high - low) * 0.6,
                    low + (high - low) * 0.8,
                ],
                dtype=torch.float32,
            )

            # Calculate appropriate scale and zero_point
            val_min, val_max = values.min().item(), values.max().item()
            scale = (val_max - val_min) / 255.0 if val_max != val_min else 0.1

            if qtype == torch.quint8:
                zero_point = int(-val_min / scale) if scale > 0 else 128
                zero_point = max(0, min(255, zero_point))
            else:  # qint8
                zero_point = 0

            qx = torch.quantize_per_tensor(values, scale, zero_point, qtype)
            quantized_inputs.append(qx)
            dequantized_inputs.append(qx.dequantize())

        return quantized_inputs, dequantized_inputs

    @pytest.mark.parametrize("qtype", [torch.quint8, torch.qint8])
    def test_quantized_basic(self, qtype: torch.dtype):
        """Test basic quantized tensor support."""
        if "test_quantized_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_quantized:
            pytest.skip("Operator does not support quantized tensors")

        # Create quantized inputs for all arguments
        quantized_inputs, dequantized_inputs = self._create_quantized_inputs(
            qtype
        )

        # Compute with quantized inputs
        result = self.descriptor.func(*quantized_inputs)

        # Verify result is quantized
        assert result.is_quantized

        # Compare with dequantized computation
        expected = self.descriptor.func(*dequantized_inputs)

        # Quantization introduces some error, use relaxed tolerances
        torch.testing.assert_close(
            result.dequantize(), expected, rtol=0.1, atol=0.1
        )

    def test_quantized_mixed_with_float(self):
        """Test quantized mixed with float inputs (first arg quantized)."""
        if "test_quantized_mixed_with_float" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_quantized:
            pytest.skip("Operator does not support quantized tensors")

        if self.descriptor.arity < 2:
            pytest.skip("Mixed quantized/float test requires arity >= 2")

        # Create inputs: first quantized, rest float
        quantized_inputs, dequantized_inputs = self._create_quantized_inputs(
            torch.quint8
        )
        mixed_inputs = [quantized_inputs[0]] + dequantized_inputs[1:]

        # Compute with mixed inputs
        result = self.descriptor.func(*mixed_inputs)

        # Compare with fully dequantized computation
        expected = self.descriptor.func(*dequantized_inputs)

        if result.is_quantized:
            torch.testing.assert_close(
                result.dequantize(), expected, rtol=0.1, atol=0.1
            )
        else:
            torch.testing.assert_close(result, expected, rtol=0.1, atol=0.1)

    def test_quantized_preserves_scale(self):
        """Test that output quantization uses appropriate scale."""
        if "test_quantized_preserves_scale" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_quantized:
            pytest.skip("Operator does not support quantized tensors")

        quantized_inputs, _ = self._create_quantized_inputs(torch.quint8)

        result = self.descriptor.func(*quantized_inputs)

        # Verify result has valid quantization parameters
        assert result.is_quantized
        assert result.q_scale() > 0
