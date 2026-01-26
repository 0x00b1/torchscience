import math

import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import ackley


class TestAckley:
    """Tests for the Ackley test function."""

    def test_minimum_at_origin(self):
        """Test that the global minimum is 0 at the origin."""
        x = torch.zeros(5)
        result = ackley(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_positive_away_from_origin(self):
        """Test that function is positive away from origin."""
        x = torch.ones(3)
        result = ackley(x)
        assert result.item() > 0

    def test_matches_python_reference(self):
        """Test that C++ kernel matches Python reference implementation."""
        x = torch.tensor([0.5, -0.3, 1.7, -2.1, 0.0])
        result = ackley(x)
        # Python reference
        a, b, c = 20.0, 0.2, 2.0 * math.pi
        mean_sq = torch.mean(x**2, -1)
        mean_cos = torch.mean(torch.cos(c * x), -1)
        expected = (
            -a * torch.exp(-b * torch.sqrt(mean_sq))
            - torch.exp(mean_cos)
            + a
            + math.e
        )
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_1d(self):
        """Test with 1-dimensional input."""
        x = torch.tensor([0.0])
        result = ackley(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_symmetry(self):
        """Test that f(x) = f(-x) (Ackley is symmetric)."""
        x = torch.tensor([1.0, -2.0, 0.5])
        result_pos = ackley(x)
        result_neg = ackley(-x)
        torch.testing.assert_close(
            result_pos, result_neg, atol=1e-6, rtol=1e-6
        )

    def test_origin_is_global_minimum(self):
        """Test that the origin gives the smallest function value.

        Note: The gradient at exactly x=0 is NaN due to the sqrt(mean(x^2))
        singularity, so we verify minimality by checking function values.
        """
        origin_val = ackley(torch.zeros(3))
        nearby_val = ackley(torch.tensor([0.1, -0.1, 0.1]))
        assert origin_val.item() < nearby_val.item()

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        # Avoid x=0 where sqrt(mean(x^2)) has gradient singularity
        x = torch.tensor(
            [0.5, -0.3, 0.7], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(ackley, (x,))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor(
            [0.5, -0.3, 0.7], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradgradcheck(ackley, (x,))

    def test_batch(self):
        """Test with batched input."""
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = ackley(x)
        # First should be 0 (origin)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )
        # Second should be positive
        assert result[1].item() > 0

    def test_output_shape(self):
        """Test that output shape is input shape without last dimension."""
        test_cases = [
            ((3,), ()),
            ((4, 3), (4,)),
            ((2, 3, 5), (2, 3)),
        ]
        for input_shape, expected_shape in test_cases:
            x = torch.randn(input_shape)
            result = ackley(x)
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape} for input shape {input_shape}, "
                f"got {result.shape}"
            )

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(3, 5, device="meta")
        result = ackley(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x = torch.randn(5, dtype=dtype)
        result = ackley(x)
        assert result.dtype == dtype

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x = torch.tensor([1, 2, 3])
        with pytest.raises(
            RuntimeError, match="ackley requires floating-point"
        ):
            ackley(x)

    def test_0d_input_raises(self):
        """Test that scalar (0-dim) input raises an error."""
        x = torch.tensor(1.0)
        with pytest.raises(
            RuntimeError, match="ackley requires at least 1 dimension"
        ):
            ackley(x)
