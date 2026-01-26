import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import beale


class TestBeale:
    """Tests for the Beale test function."""

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_minimum_value(self):
        """Test that the global minimum is 0 at (3, 0.5)."""
        x1 = torch.tensor(3.0)
        x2 = torch.tensor(0.5)
        result = beale(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_origin(self):
        """Test function value at origin."""
        x1 = torch.tensor(0.0)
        x2 = torch.tensor(0.0)
        result = beale(x1, x2)
        # (1.5)^2 + (2.25)^2 + (2.625)^2 = 2.25 + 5.0625 + 6.890625 = 14.203125
        expected = torch.tensor(14.203125)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_matches_python_reference(self):
        """Test that C++ kernel matches Python reference implementation."""
        x1 = torch.tensor([0.0, 1.0, 2.0, -1.0, 3.0])
        x2 = torch.tensor([0.0, 1.0, -1.0, 0.5, 0.5])
        result = beale(x1, x2)
        # Python reference
        expected = (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2**2) ** 2
            + (2.625 - x1 + x1 * x2**3) ** 2
        )
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_at_minimum(self):
        """Test gradient is zero at the global minimum."""
        x1 = torch.tensor(3.0, requires_grad=True)
        x2 = torch.tensor(0.5, requires_grad=True)
        result = beale(x1, x2)
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
        x2 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(beale, (x1, x2))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x1 = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(beale, (x1, x2))

    # =========================================================================
    # Batch and broadcasting tests
    # =========================================================================

    def test_batch(self):
        """Test with batched input."""
        x1 = torch.tensor([3.0, 0.0])
        x2 = torch.tensor([0.5, 0.0])
        result = beale(x1, x2)
        expected = torch.tensor([0.0, 14.203125])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_broadcast(self):
        """Test broadcasting: scalar x1 with batched x2."""
        x1 = torch.tensor(3.0)
        x2 = torch.tensor([0.5, 0.0])
        result = beale(x1, x2)
        # f(3, 0.5) = 0
        # f(3, 0) = (1.5-3)^2 + (2.25-3)^2 + (2.625-3)^2 = 2.25+0.5625+0.140625=2.953125
        expected_0 = 0.0
        expected_1 = (1.5 - 3) ** 2 + (2.25 - 3) ** 2 + (2.625 - 3) ** 2
        expected = torch.tensor([expected_0, expected_1])
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
            result = beale(x1, x2)
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape} for inputs {shape1} and {shape2}, "
                f"got {result.shape}"
            )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x1 = torch.empty(3, device="meta")
        x2 = torch.empty(3, device="meta")
        result = beale(x1, x2)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x1 = torch.randn((), dtype=dtype)
        x2 = torch.randn((), dtype=dtype)
        result = beale(x1, x2)
        assert result.dtype == dtype

    # =========================================================================
    # Error handling tests
    # =========================================================================

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x1 = torch.tensor(3)
        x2 = torch.tensor(1)
        with pytest.raises(
            RuntimeError, match="beale requires floating-point"
        ):
            beale(x1, x2)
