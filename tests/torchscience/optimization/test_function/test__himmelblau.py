import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import himmelblau


class TestHimmelblau:
    """Tests for the Himmelblau test function."""

    def test_minimum_1(self):
        """Test global minimum at (3, 2)."""
        x1 = torch.tensor(3.0)
        x2 = torch.tensor(2.0)
        result = himmelblau(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_minimum_2(self):
        """Test global minimum at (-2.805118, 3.131312)."""
        x1 = torch.tensor(-2.805118)
        x2 = torch.tensor(3.131312)
        result = himmelblau(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-3, rtol=1e-3
        )

    def test_minimum_3(self):
        """Test global minimum at (-3.779310, -3.283186)."""
        x1 = torch.tensor(-3.779310)
        x2 = torch.tensor(-3.283186)
        result = himmelblau(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-3, rtol=1e-3
        )

    def test_minimum_4(self):
        """Test global minimum at (3.584428, -1.848126)."""
        x1 = torch.tensor(3.584428)
        x2 = torch.tensor(-1.848126)
        result = himmelblau(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-3, rtol=1e-3
        )

    def test_origin(self):
        """Test function value at origin."""
        x1 = torch.tensor(0.0)
        x2 = torch.tensor(0.0)
        result = himmelblau(x1, x2)
        # (0+0-11)^2 + (0+0-7)^2 = 121+49 = 170
        expected = torch.tensor(170.0)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_matches_python_reference(self):
        """Test that C++ kernel matches Python reference implementation."""
        x1 = torch.tensor([0.0, 1.0, 3.0, -2.0, 4.0])
        x2 = torch.tensor([0.0, 1.0, 2.0, 1.0, -1.0])
        result = himmelblau(x1, x2)
        expected = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_at_minimum(self):
        """Test gradient is zero at a global minimum."""
        x1 = torch.tensor(3.0, requires_grad=True)
        x2 = torch.tensor(2.0, requires_grad=True)
        result = himmelblau(x1, x2)
        result.backward()
        torch.testing.assert_close(
            x1.grad, torch.tensor(0.0), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            x2.grad, torch.tensor(0.0), atol=1e-5, rtol=1e-5
        )

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x1 = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(himmelblau, (x1, x2))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x1 = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(himmelblau, (x1, x2))

    def test_batch(self):
        """Test with batched input."""
        x1 = torch.tensor([3.0, 0.0])
        x2 = torch.tensor([2.0, 0.0])
        result = himmelblau(x1, x2)
        expected = torch.tensor([0.0, 170.0])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Test that output shape follows broadcasting rules."""
        test_cases = [
            ((), (), ()),
            ((3,), (3,), (3,)),
            ((3,), (), (3,)),
            ((2, 3), (3,), (2, 3)),
        ]
        for shape1, shape2, expected_shape in test_cases:
            x1 = torch.randn(shape1)
            x2 = torch.randn(shape2)
            result = himmelblau(x1, x2)
            assert result.shape == expected_shape

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x1 = torch.empty(3, device="meta")
        x2 = torch.empty(3, device="meta")
        result = himmelblau(x1, x2)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x1 = torch.randn((), dtype=dtype)
        x2 = torch.randn((), dtype=dtype)
        result = himmelblau(x1, x2)
        assert result.dtype == dtype

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x1 = torch.tensor(3)
        x2 = torch.tensor(2)
        with pytest.raises(
            RuntimeError, match="himmelblau requires floating-point"
        ):
            himmelblau(x1, x2)
