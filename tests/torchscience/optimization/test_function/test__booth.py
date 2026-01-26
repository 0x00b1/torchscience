import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import booth


class TestBooth:
    """Tests for the Booth test function."""

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_minimum_value(self):
        """Test that the global minimum is 0 at (1, 3)."""
        x1 = torch.tensor(1.0)
        x2 = torch.tensor(3.0)
        result = booth(x1, x2)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-8, rtol=1e-8
        )

    def test_origin(self):
        """Test function value at origin: (0+0-7)^2 + (0+0-5)^2 = 49+25 = 74."""
        x1 = torch.tensor(0.0)
        x2 = torch.tensor(0.0)
        result = booth(x1, x2)
        expected = torch.tensor(74.0)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_known_values(self):
        """Test function at various known points."""
        test_cases = [
            (1.0, 3.0, 0.0),  # global minimum
            (0.0, 0.0, 74.0),  # origin: (0+0-7)^2 + (0+0-5)^2 = 49+25
            (1.0, 0.0, 45.0),  # (1+0-7)^2 + (2+0-5)^2 = 36+9 = 45
            (0.0, 1.0, 41.0),  # (0+2-7)^2 + (0+1-5)^2 = 25+16 = 41
        ]
        for x1_val, x2_val, expected_val in test_cases:
            x1 = torch.tensor(x1_val)
            x2 = torch.tensor(x2_val)
            result = booth(x1, x2)
            torch.testing.assert_close(
                result,
                torch.tensor(expected_val),
                atol=1e-5,
                rtol=1e-5,
                msg=f"Failed for ({x1_val}, {x2_val})",
            )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_at_minimum(self):
        """Test gradient is zero at the global minimum."""
        x1 = torch.tensor(1.0, requires_grad=True)
        x2 = torch.tensor(3.0, requires_grad=True)
        result = booth(x1, x2)
        result.backward()
        torch.testing.assert_close(
            x1.grad, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            x2.grad, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x1 = torch.randn((), dtype=torch.float64, requires_grad=True)
        x2 = torch.randn((), dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(booth, (x1, x2))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x1 = torch.randn((), dtype=torch.float64, requires_grad=True)
        x2 = torch.randn((), dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(booth, (x1, x2))

    # =========================================================================
    # Batch and broadcasting tests
    # =========================================================================

    def test_batch(self):
        """Test with batched input."""
        x1 = torch.tensor([1.0, 0.0])
        x2 = torch.tensor([3.0, 0.0])
        result = booth(x1, x2)
        expected = torch.tensor([0.0, 74.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_broadcast(self):
        """Test broadcasting: scalar x1 with batched x2."""
        x1 = torch.tensor(1.0)
        x2 = torch.tensor([3.0, 0.0])
        result = booth(x1, x2)
        # (1+6-7)^2 + (2+3-5)^2 = 0+0 = 0 for x2=3
        # (1+0-7)^2 + (2+0-5)^2 = 36+9 = 45 for x2=0
        expected = torch.tensor([0.0, 45.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_output_shape(self):
        """Test that output shape follows broadcasting rules."""
        test_cases = [
            ((), (), ()),
            ((3,), (3,), (3,)),
            ((3,), (), (3,)),
            ((2, 3), (3,), (2, 3)),
            ((2, 3), (2, 1), (2, 3)),
        ]
        for shape1, shape2, expected_shape in test_cases:
            x1 = torch.randn(shape1)
            x2 = torch.randn(shape2)
            result = booth(x1, x2)
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
        result = booth(x1, x2)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    def test_meta_shape_scalar(self):
        """Test meta tensor with scalar input."""
        x1 = torch.empty((), device="meta")
        x2 = torch.empty((), device="meta")
        result = booth(x1, x2)
        assert result.shape == ()
        assert result.device.type == "meta"

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x1 = torch.randn((), dtype=dtype)
        x2 = torch.randn((), dtype=dtype)
        result = booth(x1, x2)
        assert result.dtype == dtype

    # =========================================================================
    # Error handling tests
    # =========================================================================

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x1 = torch.tensor(1)
        x2 = torch.tensor(3)
        with pytest.raises(
            RuntimeError, match="booth requires floating-point"
        ):
            booth(x1, x2)
