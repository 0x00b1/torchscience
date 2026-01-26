import math

import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import rastrigin


class TestRastrigin:
    """Tests for the Rastrigin test function."""

    def test_minimum_at_origin(self):
        """Test that the global minimum is 0 at the origin."""
        x = torch.zeros(5)
        result = rastrigin(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_at_integers(self):
        """Test at integer points: x_i^2 - 10*cos(2*pi*x_i) = x_i^2 - 10."""
        # At integer points, cos(2*pi*x_i) = 1, so each term is x_i^2 - 10
        # f(x) = 10*n + sum(x_i^2 - 10) = 10*n + sum(x_i^2) - 10*n = sum(x_i^2)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = rastrigin(x)
        expected = torch.tensor(14.0)  # 1 + 4 + 9
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matches_python_reference(self):
        """Test that C++ kernel matches Python reference implementation."""
        x = torch.tensor([0.5, -0.3, 1.7, -2.1, 0.0])
        result = rastrigin(x)
        n = x.shape[-1]
        expected = 10.0 * n + torch.sum(
            x**2 - 10.0 * torch.cos(2.0 * math.pi * x), -1
        )
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_1d(self):
        """Test with 1-dimensional input."""
        x = torch.tensor([0.0])
        result = rastrigin(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_gradient_at_minimum(self):
        """Test gradient is zero at the global minimum."""
        x = torch.zeros(3, requires_grad=True)
        result = rastrigin(x)
        result.backward()
        torch.testing.assert_close(
            x.grad, torch.zeros(3), atol=1e-5, rtol=1e-5
        )

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        # Use points not near local minima to avoid numerical issues
        x = torch.tensor(
            [0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(rastrigin, (x,))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor(
            [0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradgradcheck(rastrigin, (x,))

    def test_batch(self):
        """Test with batched input."""
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = rastrigin(x)
        expected = torch.tensor([0.0, 2.0])  # At integers: sum(x_i^2)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Test that output shape is input shape without last dimension."""
        test_cases = [
            ((3,), ()),
            ((4, 3), (4,)),
            ((2, 3, 5), (2, 3)),
        ]
        for input_shape, expected_shape in test_cases:
            x = torch.randn(input_shape)
            result = rastrigin(x)
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape} for input shape {input_shape}, "
                f"got {result.shape}"
            )

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(3, 5, device="meta")
        result = rastrigin(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x = torch.randn(5, dtype=dtype)
        result = rastrigin(x)
        assert result.dtype == dtype

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x = torch.tensor([1, 2, 3])
        with pytest.raises(
            RuntimeError, match="rastrigin requires floating-point"
        ):
            rastrigin(x)

    def test_0d_input_raises(self):
        """Test that scalar (0-dim) input raises an error."""
        x = torch.tensor(1.0)
        with pytest.raises(
            RuntimeError, match="rastrigin requires at least 1 dimension"
        ):
            rastrigin(x)
