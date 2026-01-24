"""Comprehensive tests for joint entropy."""

import pytest
import scipy.stats
import torch
from torch.autograd import gradcheck

from torchscience.information import joint_entropy, shannon_entropy


class TestJointEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for 2D joint distribution."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        result = joint_entropy(joint)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched joint distributions."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = joint_entropy(joint)
        assert result.shape == torch.Size([10])

    def test_output_shape_4d_batch(self):
        """Returns 2D tensor for 2D batched joint distributions."""
        joint = torch.softmax(
            torch.randn(8, 10, 4, 5).flatten(-2), dim=-1
        ).view(8, 10, 4, 5)
        result = joint_entropy(joint)
        assert result.shape == torch.Size([8, 10])

    def test_reduction_mean(self):
        """Mean reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = joint_entropy(joint, reduction="mean")
        assert result.shape == torch.Size([])

    def test_reduction_sum(self):
        """Sum reduction returns scalar."""
        joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1).view(
            10, 4, 5
        )
        result = joint_entropy(joint, reduction="sum")
        assert result.shape == torch.Size([])


class TestJointEntropyCorrectness:
    """Numerical correctness tests."""

    def test_independent_variables(self):
        """H(X,Y) = H(X) + H(Y) for independent variables."""
        # Create independent joint distribution: p(x,y) = p(x) * p(y)
        p_x = torch.softmax(torch.randn(4), dim=-1)
        p_y = torch.softmax(torch.randn(5), dim=-1)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)  # Outer product

        h_xy = joint_entropy(joint)
        h_x = shannon_entropy(p_x)
        h_y = shannon_entropy(p_y)

        assert torch.isclose(h_xy, h_x + h_y, rtol=1e-5)

    def test_subadditivity(self):
        """H(X,Y) <= H(X) + H(Y) always."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            p_x = joint.sum(dim=1)  # Marginal P(X)
            p_y = joint.sum(dim=0)  # Marginal P(Y)

            h_xy = joint_entropy(joint)
            h_x = shannon_entropy(p_x)
            h_y = shannon_entropy(p_y)

            assert h_xy <= h_x + h_y + 1e-5  # Small tolerance

    def test_matches_scipy(self):
        """Result matches scipy.stats.entropy on flattened joint."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        result = joint_entropy(joint)
        expected = scipy.stats.entropy(joint.flatten().numpy())
        assert torch.isclose(
            result, torch.tensor(expected, dtype=joint.dtype), rtol=1e-5
        )

    def test_non_negative(self):
        """Joint entropy is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(
                4, 5
            )
            h_xy = joint_entropy(joint)
            assert h_xy >= -1e-6  # Small tolerance for numerical errors

    def test_uniform_distribution_maximum(self):
        """Uniform distribution has maximum entropy."""
        # Uniform 4x5 joint distribution
        joint = torch.ones(4, 5) / 20.0
        h_xy = joint_entropy(joint)
        # Maximum entropy is log(20) for 20 outcomes
        expected = torch.log(torch.tensor(20.0))
        assert torch.isclose(h_xy, expected, rtol=1e-5)

    def test_delta_distribution_zero(self):
        """Delta distribution has zero entropy."""
        joint = torch.zeros(4, 5)
        joint[1, 2] = 1.0
        h_xy = joint_entropy(joint)
        assert torch.isclose(h_xy, torch.tensor(0.0), atol=1e-5)

    def test_base_2(self):
        """Base-2 logarithm returns bits."""
        joint = torch.ones(4, 4) / 16.0  # Uniform over 16 outcomes
        h_bits = joint_entropy(joint, base=2.0)
        # log2(16) = 4 bits
        assert torch.isclose(h_bits, torch.tensor(4.0), rtol=1e-5)

    def test_log_probability_input(self):
        """Log-probability input type works correctly."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        log_joint = torch.log(joint)

        h_prob = joint_entropy(joint, input_type="probability")
        h_log = joint_entropy(log_joint, input_type="log_probability")

        assert torch.isclose(h_prob, h_log, rtol=1e-4)


class TestJointEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.softmax(
            torch.randn(3, 4, dtype=torch.float64).flatten(), dim=-1
        ).view(3, 4)
        joint.requires_grad_(True)

        def func(j):
            return joint_entropy(j)

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """First-order gradients are correct for batched input."""
        joint = torch.softmax(
            torch.randn(5, 3, 4, dtype=torch.float64).flatten(-2), dim=-1
        ).view(5, 3, 4)
        joint.requires_grad_(True)

        def func(j):
            return joint_entropy(j).sum()

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_shape(self):
        """Gradient has same shape as input."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        joint.requires_grad_(True)

        h = joint_entropy(joint)
        h.backward()

        assert joint.grad is not None
        assert joint.grad.shape == joint.shape


class TestJointEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.randn(4, 5, device="meta")
        result = joint_entropy(joint)
        assert result.shape == torch.Size([])
        assert result.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.randn(10, 4, 5, device="meta")
        result = joint_entropy(joint)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"


class TestJointEntropyEdgeCases:
    """Edge case tests."""

    def test_single_element(self):
        """Single element joint distribution has zero entropy."""
        joint = torch.tensor([[1.0]])
        h = joint_entropy(joint)
        assert torch.isclose(h, torch.tensor(0.0), atol=1e-5)

    def test_very_small_probabilities(self):
        """Very small probabilities are handled correctly."""
        joint = torch.zeros(4, 5)
        joint[0, 0] = 1.0 - 1e-10
        joint[1, 1] = 1e-10
        h = joint_entropy(joint)
        # Should be close to 0 (nearly delta)
        assert h < 1e-3


class TestJointEntropyDtypes:
    """Dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input dtype."""
        joint = torch.softmax(
            torch.randn(4, 5, dtype=dtype).flatten(), dim=-1
        ).view(4, 5)
        result = joint_entropy(joint)
        assert result.dtype == dtype


class TestJointEntropyErrors:
    """Error handling tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="input_type"):
            joint_entropy(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="reduction"):
            joint_entropy(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(ValueError, match="base"):
            joint_entropy(joint, base=1.0)  # base=1 is invalid
        with pytest.raises(ValueError, match="base"):
            joint_entropy(joint, base=-1.0)  # negative base is invalid

    def test_dim_out_of_range(self):
        """Dim out of range raises IndexError."""
        joint = torch.softmax(torch.randn(4, 5).flatten(), dim=-1).view(4, 5)
        with pytest.raises(IndexError, match="out of range"):
            joint_entropy(joint, dims=(5,))

    def test_not_tensor(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            joint_entropy([[0.25, 0.25], [0.25, 0.25]])
