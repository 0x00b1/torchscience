"""Tests for total correlation (multi-information)."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import total_correlation


class TestTotalCorrelationBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Output is scalar for 2D input."""
        joint = torch.rand(4, 4)
        joint = joint / joint.sum()
        result = total_correlation(joint)
        assert result.dim() == 0

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = total_correlation(joint)
        assert result.dim() == 0

    def test_non_negative(self):
        """Total correlation is non-negative."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()
        result = total_correlation(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = total_correlation(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = total_correlation(log_joint, input_type="log_probability")
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3)
        result = total_correlation(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = total_correlation(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = total_correlation(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2)
        joint = joint / joint.sum()
        result_nats = total_correlation(joint, base=None)
        result_bits = total_correlation(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestTotalCorrelationCorrectness:
    """Correctness tests."""

    def test_independent_variables_2d(self):
        """TC = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y) = p(x) * p(y)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        joint = torch.outer(p_x, p_y)

        result = total_correlation(joint)
        assert torch.abs(result) < 1e-5

    def test_independent_variables_3d(self):
        """TC = 0 for independent 3D joint."""
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        # p(x,y,z) = p(x) * p(y) * p(z)
        joint = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)

        result = total_correlation(joint)
        assert torch.abs(result) < 1e-5

    def test_perfect_correlation_2d(self):
        """TC > 0 for perfectly correlated variables."""
        # X = Y with probability 1
        joint = torch.zeros(2, 2)
        joint[0, 0] = 0.5
        joint[1, 1] = 0.5

        result = total_correlation(joint)
        # TC = H(X) + H(Y) - H(X,Y) = log(2) + log(2) - log(2) = log(2)
        expected = torch.log(torch.tensor(2.0))
        assert torch.abs(result - expected) < 1e-5

    def test_equals_mutual_information_for_2d(self):
        """For 2D, TC equals mutual information I(X;Y)."""
        joint = torch.rand(4, 5)
        joint = joint / joint.sum()

        # TC = H(X) + H(Y) - H(X,Y)
        tc = total_correlation(joint)

        # Compute I(X;Y) = H(X) + H(Y) - H(X,Y) manually
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        eps = 1e-10
        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_joint = -(joint * torch.log(joint + eps)).sum()
        mi = h_x + h_y - h_joint

        assert torch.allclose(tc, mi, rtol=1e-4)

    def test_tc_formula_3d(self):
        """Verify TC = sum_i H(X_i) - H(joint) for 3D case."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        tc = total_correlation(joint)

        # Compute manually
        eps = 1e-10

        # Marginals
        p_x = joint.sum(dim=(1, 2))  # Sum over Y, Z
        p_y = joint.sum(dim=(0, 2))  # Sum over X, Z
        p_z = joint.sum(dim=(0, 1))  # Sum over X, Y

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_z = -(p_z * torch.log(p_z + eps)).sum()
        h_joint = -(joint * torch.log(joint + eps)).sum()

        expected = h_x + h_y + h_z - h_joint

        assert torch.allclose(tc, expected, rtol=1e-4)

    def test_kl_divergence_form(self):
        """Verify TC = KL(joint || product of marginals)."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()

        tc = total_correlation(joint)

        # Compute KL(joint || prod marginals)
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)
        product = torch.outer(p_x, p_y)

        eps = 1e-10
        kl = (
            joint * (torch.log(joint + eps) - torch.log(product + eps))
        ).sum()

        assert torch.allclose(tc, kl, rtol=1e-4)


class TestTotalCorrelationGradients:
    """Gradient tests."""

    def test_gradcheck_2d(self):
        """First-order gradients are correct for 2D."""
        joint = torch.rand(3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        # Add small offset to avoid zeros for numerical stability
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return total_correlation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_3d(self):
        """First-order gradients are correct for 3D."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return total_correlation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 4, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = total_correlation(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = total_correlation(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestTotalCorrelationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="input_type"):
            total_correlation(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="reduction"):
            total_correlation(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            total_correlation(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <2 dimensions raises ValueError."""
        joint = torch.rand(5)
        with pytest.raises(ValueError, match="2 dimensions"):
            total_correlation(joint)


class TestTotalCorrelationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.rand(3, 4, device="meta")
        result = total_correlation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output

    def test_meta_tensor_3d(self):
        """Meta tensor works for 3D input."""
        joint = torch.rand(2, 3, 4, device="meta")
        result = total_correlation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0
