"""Comprehensive tests for conditional entropy."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.information import (
    conditional_entropy,
    joint_entropy,
    shannon_entropy,
)


class TestConditionalEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for 2D joint distribution."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        result = conditional_entropy(joint)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched joint distributions."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = conditional_entropy(joint)
        assert result.shape == torch.Size([10])

    def test_output_shape_4d_batch(self):
        """Returns 2D tensor for 2D batched joint distributions."""
        joint = torch.softmax(
            torch.randn(8, 10, 4, 5).flatten(-2), dim=-1
        ).view(8, 10, 4, 5)
        result = conditional_entropy(joint)
        assert result.shape == torch.Size([8, 10])

    def test_reduction_mean(self):
        """Mean reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = conditional_entropy(joint, reduction="mean")
        assert result.shape == torch.Size([])

    def test_reduction_sum(self):
        """Sum reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = conditional_entropy(joint, reduction="sum")
        assert result.shape == torch.Size([])


class TestConditionalEntropyCorrectness:
    """Numerical correctness tests."""

    def test_chain_rule(self):
        """Verify H(Y|X) = H(X,Y) - H(X)."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        p_x = joint.sum(dim=1)

        h_y_given_x = conditional_entropy(joint)
        h_xy = joint_entropy(joint)
        h_x = shannon_entropy(p_x)

        assert torch.isclose(h_y_given_x, h_xy - h_x, rtol=1e-4)

    def test_independent_variables(self):
        """H(Y|X) = H(Y) for independent X, Y."""
        p_x = torch.softmax(torch.randn(4), dim=-1)
        p_y = torch.softmax(torch.randn(5), dim=-1)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        h_y_given_x = conditional_entropy(joint)
        h_y = shannon_entropy(p_y)

        assert torch.isclose(h_y_given_x, h_y, rtol=1e-4)

    def test_conditioning_reduces_entropy(self):
        """H(Y|X) <= H(Y) always."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            p_y = joint.sum(dim=0)

            h_y_given_x = conditional_entropy(joint)
            h_y = shannon_entropy(p_y)

            assert h_y_given_x <= h_y + 1e-5

    def test_non_negative(self):
        """Conditional entropy is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            h_y_given_x = conditional_entropy(joint)
            assert h_y_given_x >= -1e-6

    def test_deterministic_relationship(self):
        """When Y is determined by X, H(Y|X) = 0."""
        # Y = X (deterministic mapping)
        joint = torch.zeros(4, 4)
        for i in range(4):
            joint[i, i] = 0.25

        h_y_given_x = conditional_entropy(joint)
        assert torch.isclose(h_y_given_x, torch.tensor(0.0), atol=1e-5)

    def test_symmetric_conditioning(self):
        """Test H(X|Y) using swapped dims."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        # H(Y|X) with default dims
        h_y_given_x = conditional_entropy(
            joint, condition_dim=-2, target_dim=-1
        )

        # H(X|Y) with swapped dims
        h_x_given_y = conditional_entropy(
            joint, condition_dim=-1, target_dim=-2
        )

        # Chain rule: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
        h_xy = joint_entropy(joint)
        h_x = shannon_entropy(p_x)
        h_y = shannon_entropy(p_y)

        assert torch.isclose(h_xy, h_x + h_y_given_x, rtol=1e-4)
        assert torch.isclose(h_xy, h_y + h_x_given_y, rtol=1e-4)

    def test_base_2(self):
        """Base-2 logarithm returns bits."""
        # Y is uniform 4 outcomes given any X
        joint = torch.ones(2, 4) / 8.0
        h_bits = conditional_entropy(joint, base=2.0)
        # H(Y|X) = log2(4) = 2 bits
        assert torch.isclose(h_bits, torch.tensor(2.0), rtol=1e-5)


class TestConditionalEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.softmax(
            torch.randn(3, 4, dtype=torch.float64).flatten(), dim=-1
        ).view(3, 4)
        joint.requires_grad_(True)

        def func(j):
            return conditional_entropy(j)

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """First-order gradients are correct for batched input."""
        joint = torch.softmax(
            torch.randn(5, 3, 4, dtype=torch.float64).flatten(-2), dim=-1
        ).view(5, 3, 4)
        joint.requires_grad_(True)

        def func(j):
            return conditional_entropy(j).sum()

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_shape(self):
        """Gradient has same shape as input."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        joint.requires_grad_(True)

        h = conditional_entropy(joint)
        h.backward()

        assert joint.grad is not None
        assert joint.grad.shape == joint.shape

    def test_gradgradcheck(self):
        """Second-order gradients are correct."""
        joint = torch.softmax(
            torch.randn(3, 4, dtype=torch.float64).flatten(), dim=-1
        ).view(3, 4)
        joint.requires_grad_(True)

        def func(j):
            return conditional_entropy(j)

        assert gradgradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_batched(self):
        """Second-order gradients are correct for batched input."""
        joint = torch.softmax(
            torch.randn(5, 3, 4, dtype=torch.float64).flatten(-2), dim=-1
        ).view(5, 3, 4)
        joint.requires_grad_(True)

        def func(j):
            return conditional_entropy(j).sum()

        assert gradgradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestConditionalEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.randn(4, 5, device="meta")
        result = conditional_entropy(joint)
        assert result.shape == torch.Size([])
        assert result.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.randn(10, 4, 5, device="meta")
        result = conditional_entropy(joint)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"


class TestConditionalEntropyDtypes:
    """Dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input dtype."""
        joint = torch.softmax(
            torch.randn(4, 5, dtype=dtype).flatten(), dim=-1
        ).view(4, 5)
        result = conditional_entropy(joint)
        assert result.dtype == dtype


class TestConditionalEntropyErrors:
    """Error handling tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="input_type"):
            conditional_entropy(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="reduction"):
            conditional_entropy(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="base"):
            conditional_entropy(joint, base=1.0)

    def test_same_dims(self):
        """Same condition and target dims raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="different"):
            conditional_entropy(joint, condition_dim=-1, target_dim=-1)

    def test_dim_out_of_range(self):
        """Dim out of range raises IndexError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(IndexError, match="out of range"):
            conditional_entropy(joint, condition_dim=5)

    def test_not_tensor(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            conditional_entropy([[0.25, 0.25], [0.25, 0.25]])

    def test_1d_input(self):
        """1D input raises ValueError."""
        p = torch.softmax(torch.randn(5), dim=-1)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            conditional_entropy(p)
