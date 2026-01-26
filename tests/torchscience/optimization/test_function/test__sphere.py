import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import sphere


class TestSphere:
    """Tests for the Sphere test function."""

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_minimum_at_origin(self):
        """Test that the global minimum is 0 at the origin."""
        x = torch.zeros(5)
        result = sphere(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-8, rtol=1e-8
        )

    def test_known_value(self):
        """Test function value at a known point."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sphere(x)
        expected = torch.tensor(14.0)  # 1 + 4 + 9
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_known_values(self):
        """Test function at various known points."""
        test_cases = [
            (torch.tensor([0.0]), 0.0),
            (torch.tensor([1.0]), 1.0),
            (torch.tensor([1.0, 1.0]), 2.0),
            (torch.tensor([3.0, 4.0]), 25.0),
            (torch.tensor([1.0, 2.0, 3.0, 4.0]), 30.0),
        ]
        for x, expected in test_cases:
            result = sphere(x)
            torch.testing.assert_close(
                result,
                torch.tensor(expected),
                atol=1e-6,
                rtol=1e-6,
                msg=f"Failed for x={x.tolist()}",
            )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient(self):
        """Test gradient is 2*x."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = sphere(x)
        result.backward()
        expected_grad = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-6)

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(sphere, (x,))

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(sphere, (x,))

    # =========================================================================
    # Batch dimension tests
    # =========================================================================

    def test_batch(self):
        """Test with batched input."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = sphere(x)
        expected = torch.tensor([5.0, 25.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_batch_2d(self):
        """Test with 3D input (2D batch of points)."""
        x = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [2.0, 3.0]],
            ]
        )
        result = sphere(x)
        expected = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 13.0],
            ]
        )
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_output_shape(self):
        """Test that output shape is input shape without last dimension."""
        test_cases = [
            ((3,), ()),
            ((4, 3), (4,)),
            ((2, 3, 5), (2, 3)),
            ((2, 3, 4, 5), (2, 3, 4)),
        ]
        for input_shape, expected_shape in test_cases:
            x = torch.randn(input_shape)
            result = sphere(x)
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape} for input shape {input_shape}, "
                f"got {result.shape}"
            )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(3, 5, device="meta")
        result = sphere(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    def test_meta_shape_1d(self):
        """Test meta tensor with 1D input."""
        x = torch.empty(7, device="meta")
        result = sphere(x)
        assert result.shape == ()
        assert result.device.type == "meta"

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        x = torch.randn(5, dtype=dtype)
        result = sphere(x)
        assert result.dtype == dtype

    # =========================================================================
    # Error handling tests
    # =========================================================================

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x = torch.tensor([1, 2, 3])
        with pytest.raises(
            RuntimeError, match="sphere requires floating-point"
        ):
            sphere(x)

    def test_0d_input_raises(self):
        """Test that scalar (0-dim) input raises an error."""
        x = torch.tensor(1.0)
        with pytest.raises(
            RuntimeError, match="sphere requires at least 1 dimension"
        ):
            sphere(x)
