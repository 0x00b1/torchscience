"""Tests for causally conditioned entropy."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import (
    causally_conditioned_entropy,
    conditional_entropy,
    shannon_entropy,
)


class TestCausallyConditionedEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()
        result = causally_conditioned_entropy(joint)
        assert result.dim() == 0

    def test_output_shape_batched(self):
        """Output has batch dimensions for >3D input."""
        joint = torch.rand(2, 3, 4, 5, 6)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = causally_conditioned_entropy(joint)
        assert result.shape == (2, 3)

    def test_non_negative(self):
        """Causally conditioned entropy is non-negative."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = causally_conditioned_entropy(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = causally_conditioned_entropy(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = causally_conditioned_entropy(
            log_joint, input_type="log_probability"
        )
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = causally_conditioned_entropy(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = causally_conditioned_entropy(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2, 2)
        joint = joint / joint.sum()
        result_nats = causally_conditioned_entropy(joint, base=None)
        result_bits = causally_conditioned_entropy(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestCausallyConditionedEntropyCorrectness:
    """Correctness tests."""

    def test_deterministic_zero(self):
        """H(Y||X) = 0 when Y_t is deterministic given (Y_{t-1}, X_t)."""
        # Create a distribution where y_t is determined by (y_prev, x_t)
        # For each (y_prev, x_t) pair, y_t is a specific value
        joint = torch.zeros(2, 2, 2)
        # When (y_prev=0, x_t=0) -> y_t=0
        joint[0, 0, 0] = 0.125
        # When (y_prev=0, x_t=1) -> y_t=1
        joint[1, 0, 1] = 0.125
        # When (y_prev=1, x_t=0) -> y_t=1
        joint[1, 1, 0] = 0.125
        # When (y_prev=1, x_t=1) -> y_t=0
        joint[0, 1, 1] = 0.125
        # Fill remaining to make valid distribution
        joint[0, 0, 1] = 0.125
        joint[1, 0, 0] = 0.125
        joint[0, 1, 0] = 0.125
        joint[1, 1, 1] = 0.125

        # Actually make it deterministic by zeroing alternatives
        joint_det = torch.zeros(2, 2, 2)
        # For each (y_prev, x_t), only one y_t has probability
        joint_det[0, 0, 0] = 0.25  # y_t=0 when (y_prev=0, x_t=0)
        joint_det[1, 0, 1] = 0.25  # y_t=1 when (y_prev=0, x_t=1)
        joint_det[1, 1, 0] = 0.25  # y_t=1 when (y_prev=1, x_t=0)
        joint_det[0, 1, 1] = 0.25  # y_t=0 when (y_prev=1, x_t=1)

        result = causally_conditioned_entropy(joint_det)
        assert torch.abs(result) < 1e-5

    def test_uniform_conditional_max_entropy(self):
        """H(Y||X) = log(|Y|) when Y_t is uniform given any (Y_{t-1}, X_t)."""
        # When Y_t is uniform given (Y_{t-1}, X_t), conditional entropy is log(|Y|)
        num_y = 4
        num_yprev = 2
        num_x = 2

        # Create uniform conditional: p(y_t | y_prev, x_t) = 1/num_y for all
        joint = torch.ones(num_y, num_yprev, num_x)
        joint = joint / joint.sum()

        result = causally_conditioned_entropy(joint)
        expected = torch.log(torch.tensor(float(num_y)))
        assert torch.isclose(result, expected, rtol=1e-4)

    def test_bounded_by_marginal_entropy(self):
        """H(Y||X) <= H(Y_t) (conditioning reduces entropy)."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.rand(3, 4, 5)
            joint = joint / joint.sum()

            h_cond = causally_conditioned_entropy(joint)
            # Marginal p(y_t)
            p_yt = joint.sum(dim=(-2, -1))
            h_yt = shannon_entropy(p_yt)

            assert h_cond <= h_yt + 1e-5

    def test_independent_x_equals_conditional_on_yprev(self):
        """When X is independent, H(Y||X) = H(Y_t | Y_{t-1})."""
        # p(y_t, y_prev, x_t) = p(y_t, y_prev) * p(x_t) means X is independent
        # Create joint that factors this way
        p_yt_yprev = torch.rand(3, 4)
        p_yt_yprev = p_yt_yprev / p_yt_yprev.sum()
        p_x = torch.tensor([0.3, 0.7])

        # joint[y_t, y_prev, x_t] = p(y_t, y_prev) * p(x_t)
        joint = p_yt_yprev.unsqueeze(-1) * p_x.unsqueeze(0).unsqueeze(0)

        # H(Y_t | Y_{t-1}, X_t) should equal H(Y_t | Y_{t-1}) when X is independent
        h_cce = causally_conditioned_entropy(joint)
        h_cond = conditional_entropy(
            p_yt_yprev, condition_dim=-1, target_dim=-2
        )

        assert torch.isclose(h_cce, h_cond, rtol=1e-4)

    def test_relation_to_conditional_entropy(self):
        """H(Y_t | Y_{t-1}, X_t) is conditional entropy on the joint."""
        # This is just H(Y_t | Y_{t-1}, X_t) computed differently
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        # Our function
        h_cce = causally_conditioned_entropy(joint)

        # Manual computation using conditional entropy formula
        # H(Y_t | Y_{t-1}, X_t) = H(Y_t, Y_{t-1}, X_t) - H(Y_{t-1}, X_t)
        eps = 1e-10
        h_joint = -torch.sum(joint * torch.log(joint + eps))
        p_yprev_xt = joint.sum(dim=0)  # Sum over y_t
        h_yprev_xt = -torch.sum(p_yprev_xt * torch.log(p_yprev_xt + eps))
        h_manual = h_joint - h_yprev_xt

        assert torch.isclose(h_cce, h_manual, rtol=1e-4)


class TestCausallyConditionedEntropyGradients:
    """Gradient tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.rand(3, 3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        # Add small offset to avoid zeros
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return causally_conditioned_entropy(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = causally_conditioned_entropy(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = causally_conditioned_entropy(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape

    def test_backward_batched(self):
        """Backward pass works for batched input."""
        joint = torch.rand(2, 3, 4, 5, requires_grad=True)
        joint_norm = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = causally_conditioned_entropy(joint_norm, reduction="sum")
        result.backward()
        assert joint.grad is not None
        assert joint.grad.shape == joint.shape


class TestCausallyConditionedEntropyValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="input_type"):
            causally_conditioned_entropy(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="reduction"):
            causally_conditioned_entropy(joint, reduction="invalid")

    def test_invalid_base_negative(self):
        """Negative base raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            causally_conditioned_entropy(joint, base=-1.0)

    def test_invalid_base_one(self):
        """Base=1 raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            causally_conditioned_entropy(joint, base=1.0)

    def test_too_few_dimensions(self):
        """Tensor with <3 dimensions raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            causally_conditioned_entropy(joint)

    def test_non_tensor_input(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Tensor"):
            causally_conditioned_entropy(
                [[[0.1, 0.2], [0.3, 0.4]], [[0.0, 0.0], [0.0, 0.0]]]
            )


class TestCausallyConditionedEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape_3d(self):
        """Meta tensor produces correct output shape for 3D input."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = causally_conditioned_entropy(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output for 3D input

    def test_meta_tensor_shape_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.rand(2, 3, 4, 5, 6, device="meta")
        result = causally_conditioned_entropy(joint)
        assert result.device.type == "meta"
        assert result.shape == (2, 3)

    def test_meta_tensor_reduction(self):
        """Meta tensor with reduction produces scalar."""
        joint = torch.rand(2, 3, 4, 5, device="meta")
        result = causally_conditioned_entropy(joint, reduction="mean")
        assert result.device.type == "meta"
        assert result.dim() == 0


class TestCausallyConditionedEntropyDtypes:
    """Dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input dtype."""
        joint = torch.rand(4, 4, 4, dtype=dtype)
        joint = joint / joint.sum()
        result = causally_conditioned_entropy(joint)
        assert result.dtype == dtype
