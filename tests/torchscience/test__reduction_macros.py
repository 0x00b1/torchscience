"""Tests for reduction operator macro infrastructure."""

import pytest
import torch

import torchscience._csrc  # noqa: F401


class TestSumSquaresForward:
    """Test sum_squares forward pass."""

    def test_1d_all_dims(self):
        """Test reducing all dims of 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = float(1**2 + 2**2 + 3**2)
        assert result.shape == ()
        assert torch.allclose(result, torch.tensor(expected))

    def test_2d_all_dims(self):
        """Test reducing all dims of 2D tensor."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = 1 + 4 + 9 + 16
        assert result.shape == ()
        assert torch.allclose(
            result, torch.tensor(expected, dtype=torch.float)
        )

    def test_2d_dim0(self):
        """Test reducing along dim 0."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [0], False)
        expected = torch.tensor([1 + 9, 4 + 16], dtype=torch.float)
        assert result.shape == (2,)
        assert torch.allclose(result, expected)

    def test_2d_dim1(self):
        """Test reducing along dim 1."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], False)
        expected = torch.tensor([1 + 4, 9 + 16], dtype=torch.float)
        assert result.shape == (2,)
        assert torch.allclose(result, expected)

    def test_keepdim_true(self):
        """Test keepdim=True."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], True)
        assert result.shape == (2, 1)

    def test_3d_multiple_dims(self):
        """Test reducing multiple dims of 3D tensor."""
        x = torch.randn(2, 3, 4)
        op = torch.ops.torchscience.sum_squares
        result = op(x, [0, 2], False)
        # Should reduce dims 0 and 2, keeping dim 1
        assert result.shape == (3,)
        # Verify correctness
        expected = (x**2).sum(dim=(0, 2))
        assert torch.allclose(result, expected)

    def test_negative_dim(self):
        """Test with negative dimension index."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [-1], False)
        expected = torch.tensor([1 + 4, 9 + 16], dtype=torch.float)
        assert torch.allclose(result, expected)


class TestSumSquaresBackward:
    """Test sum_squares backward pass."""

    def test_gradient_all_dims(self):
        """Test gradient when reducing all dims."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        result.backward()
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad)

    def test_gradient_single_dim(self):
        """Test gradient when reducing single dim."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], False)
        result.sum().backward()
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad)

    def test_gradient_with_keepdim(self):
        """Test gradient with keepdim=True."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], True)
        result.sum().backward()
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad)


class TestSumSquaresGradcheck:
    """Test gradients with torch.autograd.gradcheck."""

    def test_gradcheck_all_dims(self):
        """Gradcheck for all dims reduction."""
        x = torch.randn(5, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, None, False)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradcheck_single_dim(self):
        """Gradcheck for single dim reduction."""
        x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, [1], False)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradcheck_multiple_dims(self):
        """Gradcheck for multiple dims reduction."""
        x = torch.randn(2, 3, 4, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, [0, 2], False)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradcheck_keepdim(self):
        """Gradcheck with keepdim=True."""
        x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, [1], True)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradgradcheck_all_dims(self):
        """Second-order gradient check for all dims."""
        x = torch.randn(5, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, None, False)

        assert torch.autograd.gradgradcheck(fn, (x,), raise_exception=True)


class TestSumSquaresDtypes:
    """Test sum_squares with various dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test with float32 and float64."""
        x = torch.randn(10, dtype=dtype)
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = (x**2).sum()
        assert result.dtype == dtype
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_dtypes(self, dtype):
        """Test with float16 and bfloat16."""
        x = torch.randn(10, dtype=dtype)
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = (x**2).sum()
        assert result.dtype == dtype
        # Larger tolerance for half precision
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


class TestSumSquaresMetaTensor:
    """Test sum_squares with meta tensors for shape inference."""

    def test_meta_all_dims(self):
        """Test shape inference for all dims reduction."""
        x = torch.randn(3, 4, 5, device="meta")
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        assert result.shape == ()

    def test_meta_single_dim(self):
        """Test shape inference for single dim reduction."""
        x = torch.randn(3, 4, 5, device="meta")
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], False)
        assert result.shape == (3, 5)

    def test_meta_keepdim(self):
        """Test shape inference with keepdim."""
        x = torch.randn(3, 4, 5, device="meta")
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], True)
        assert result.shape == (3, 1, 5)

    def test_meta_multiple_dims(self):
        """Test shape inference for multiple dims."""
        x = torch.randn(2, 3, 4, 5, device="meta")
        op = torch.ops.torchscience.sum_squares
        result = op(x, [0, 2], False)
        assert result.shape == (3, 5)
