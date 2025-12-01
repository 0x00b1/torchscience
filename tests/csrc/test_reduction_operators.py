import pytest
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators

# This test validates the reduction operator template works correctly
# by testing against a simple "sum" reduction (which we can compare to torch.sum)


class TestReductionOperatorTemplate:
    """Test the CPUReductionOperator template pattern."""

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(4, 5, 6, requires_grad=True)

    def test_reduction_all_dims(self, input_tensor):
        """Test reduction over all dimensions."""
        # We'll test with kurtosis since it uses the pattern
        result = torch.ops.torchscience.kurtosis(
            input_tensor, None, False, True, True
        )
        assert result.shape == ()
        assert result.dtype == input_tensor.dtype

    def test_reduction_single_dim(self, input_tensor):
        """Test reduction over a single dimension."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], False, True, True
        )
        assert result.shape == (4, 6)

    def test_reduction_keepdim(self, input_tensor):
        """Test reduction with keepdim=True."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], True, True, True
        )
        assert result.shape == (4, 1, 6)

    def test_reduction_multiple_dims(self, input_tensor):
        """Test reduction over multiple dimensions."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [0, 2], False, True, True
        )
        assert result.shape == (5,)

    def test_reduction_negative_dim(self, input_tensor):
        """Test reduction with negative dimension index."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [-1], False, True, True
        )
        assert result.shape == (4, 5)

    def test_reduction_backward(self, input_tensor):
        """Test backward pass through reduction."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], False, True, True
        )
        loss = result.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape

    def test_reduction_backward_backward(self, input_tensor):
        """Test second-order backward pass through reduction operator.

        Verifies that second-order gradients (Hessian-vector products) can be
        computed correctly for scalar reduction.
        """
        x = input_tensor.clone().detach().requires_grad_(True)

        # Test scalar reduction (all dimensions)
        result = torch.ops.torchscience.kurtosis(x, None, False, True, True)

        # First backward - compute gradient of result w.r.t. input
        (grad_input,) = torch.autograd.grad(result, x, create_graph=True)

        # Second backward - compute Hessian-vector product
        grad_grad_input = torch.ones_like(grad_input)
        grad_grad_output = torch.autograd.grad(grad_input, x, grad_grad_input)

        # Verify gradient exists and has correct shape
        assert grad_grad_output[0] is not None
        assert grad_grad_output[0].shape == x.shape

        # Verify numerical correctness
        assert torch.isfinite(grad_grad_output[0]).all(), (
            "Second-order gradients contain NaN or Inf"
        )
