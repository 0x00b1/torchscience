"""Comprehensive tests for mutual information."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.information import (
    conditional_entropy,
    joint_entropy,
    mutual_information,
    shannon_entropy,
)


class TestMutualInformationBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for 2D joint distribution."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        result = mutual_information(joint)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched joint distributions."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = mutual_information(joint)
        assert result.shape == torch.Size([10])

    def test_output_shape_4d_batch(self):
        """Returns 2D tensor for 2D batched joint distributions."""
        joint = torch.softmax(
            torch.randn(8, 10, 4, 5).flatten(-2), dim=-1
        ).view(8, 10, 4, 5)
        result = mutual_information(joint)
        assert result.shape == torch.Size([8, 10])

    def test_reduction_mean(self):
        """Mean reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = mutual_information(joint, reduction="mean")
        assert result.shape == torch.Size([])

    def test_reduction_sum(self):
        """Sum reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = mutual_information(joint, reduction="sum")
        assert result.shape == torch.Size([])


class TestMutualInformationCorrectness:
    """Numerical correctness tests."""

    def test_independent_variables_zero(self):
        """I(X;Y) = 0 for independent X, Y."""
        p_x = torch.softmax(torch.randn(4), dim=-1)
        p_y = torch.softmax(torch.randn(5), dim=-1)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        mi = mutual_information(joint)
        assert torch.isclose(mi, torch.tensor(0.0), atol=1e-5)

    def test_perfect_correlation(self):
        """I(X;Y) = H(X) = H(Y) when Y = X (deterministic)."""
        # Y = X (diagonal joint distribution)
        joint = torch.zeros(4, 4)
        for i in range(4):
            joint[i, i] = 0.25

        mi = mutual_information(joint)
        h_x = shannon_entropy(joint.sum(dim=1))
        h_y = shannon_entropy(joint.sum(dim=0))

        assert torch.isclose(mi, h_x, rtol=1e-4)
        assert torch.isclose(mi, h_y, rtol=1e-4)

    def test_formula_h_x_plus_h_y_minus_h_xy(self):
        """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        torch.manual_seed(42)
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)

        mi = mutual_information(joint)
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)
        h_x = shannon_entropy(p_x)
        h_y = shannon_entropy(p_y)
        h_xy = joint_entropy(joint)

        expected = h_x + h_y - h_xy
        assert torch.isclose(mi, expected, rtol=1e-4)

    def test_formula_h_y_minus_h_y_given_x(self):
        """I(X;Y) = H(Y) - H(Y|X)."""
        torch.manual_seed(42)
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)

        mi = mutual_information(joint)
        p_y = joint.sum(dim=0)
        h_y = shannon_entropy(p_y)
        h_y_given_x = conditional_entropy(joint)

        expected = h_y - h_y_given_x
        assert torch.isclose(mi, expected, rtol=1e-4)

    def test_symmetry(self):
        """I(X;Y) = I(Y;X)."""
        torch.manual_seed(42)
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)

        mi_xy = mutual_information(joint, dims=(-2, -1))
        mi_yx = mutual_information(joint, dims=(-1, -2))

        assert torch.isclose(mi_xy, mi_yx, rtol=1e-4)

    def test_non_negative(self):
        """Mutual information is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            mi = mutual_information(joint)
            assert mi >= -1e-6

    def test_bounded_by_marginal_entropies(self):
        """I(X;Y) <= min(H(X), H(Y))."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            mi = mutual_information(joint)
            h_x = shannon_entropy(joint.sum(dim=1))
            h_y = shannon_entropy(joint.sum(dim=0))

            min_entropy = torch.min(h_x, h_y)
            assert mi <= min_entropy + 1e-5

    def test_base_2(self):
        """Base-2 logarithm returns bits."""
        # Perfect correlation with uniform marginal
        joint = torch.zeros(4, 4)
        for i in range(4):
            joint[i, i] = 0.25

        mi_bits = mutual_information(joint, base=2.0)
        # I(X;Y) = H(X) = log2(4) = 2 bits
        assert torch.isclose(mi_bits, torch.tensor(2.0), rtol=1e-5)


class TestMutualInformationGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.softmax(
            torch.randn(3, 4, dtype=torch.float64).flatten(), dim=-1
        ).view(3, 4)
        joint.requires_grad_(True)

        def func(j):
            return mutual_information(j)

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """First-order gradients are correct for batched input."""
        joint = torch.softmax(
            torch.randn(5, 3, 4, dtype=torch.float64).flatten(-2), dim=-1
        ).view(5, 3, 4)
        joint.requires_grad_(True)

        def func(j):
            return mutual_information(j).sum()

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_shape(self):
        """Gradient has same shape as input."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        joint.requires_grad_(True)

        mi = mutual_information(joint)
        mi.backward()

        assert joint.grad is not None
        assert joint.grad.shape == joint.shape

    def test_gradgradcheck(self):
        """Second-order gradients are correct."""
        joint = torch.softmax(
            torch.randn(3, 4, dtype=torch.float64).flatten(), dim=-1
        ).view(3, 4)
        joint.requires_grad_(True)

        def func(j):
            return mutual_information(j)

        assert gradgradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_batched(self):
        """Second-order gradients are correct for batched input."""
        joint = torch.softmax(
            torch.randn(5, 3, 4, dtype=torch.float64).flatten(-2), dim=-1
        ).view(5, 3, 4)
        joint.requires_grad_(True)

        def func(j):
            return mutual_information(j).sum()

        assert gradgradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestMutualInformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.randn(4, 5, device="meta")
        result = mutual_information(joint)
        assert result.shape == torch.Size([])
        assert result.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.randn(10, 4, 5, device="meta")
        result = mutual_information(joint)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"


class TestMutualInformationDtypes:
    """Dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input dtype."""
        joint = torch.softmax(
            torch.randn(4, 5, dtype=dtype).flatten(), dim=-1
        ).view(4, 5)
        result = mutual_information(joint)
        assert result.dtype == dtype


class TestMutualInformationErrors:
    """Error handling tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="input_type"):
            mutual_information(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="reduction"):
            mutual_information(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="base"):
            mutual_information(joint, base=1.0)

    def test_wrong_dims_length(self):
        """Wrong dims length raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="2 elements"):
            mutual_information(joint, dims=(-1,))

    def test_same_dims(self):
        """Same dims raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="different"):
            mutual_information(joint, dims=(-1, -1))

    def test_dim_out_of_range(self):
        """Dim out of range raises IndexError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(IndexError, match="out of range"):
            mutual_information(joint, dims=(5, 0))

    def test_not_tensor(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            mutual_information([[0.25, 0.25], [0.25, 0.25]])

    def test_1d_input(self):
        """1D input raises ValueError."""
        p = torch.softmax(torch.randn(5), dim=-1)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            mutual_information(p)
