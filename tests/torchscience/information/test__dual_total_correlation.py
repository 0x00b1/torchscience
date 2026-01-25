"""Tests for dual total correlation (binding information)."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import dual_total_correlation


class TestDualTotalCorrelationBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Output is scalar for 2D input."""
        joint = torch.rand(4, 4)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint)
        assert result.dim() == 0

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint)
        assert result.dim() == 0

    def test_non_negative(self):
        """Dual total correlation is non-negative."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = dual_total_correlation(
            log_joint, input_type="log_probability"
        )
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3)
        result = dual_total_correlation(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = dual_total_correlation(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2)
        joint = joint / joint.sum()
        result_nats = dual_total_correlation(joint, base=None)
        result_bits = dual_total_correlation(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestDualTotalCorrelationCorrectness:
    """Correctness tests."""

    def test_independent_variables_2d(self):
        """DTC = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y) = p(x) * p(y)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        joint = torch.outer(p_x, p_y)

        result = dual_total_correlation(joint)
        assert torch.abs(result) < 1e-5

    def test_independent_variables_3d(self):
        """DTC = 0 for independent 3D joint."""
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        # p(x,y,z) = p(x) * p(y) * p(z)
        joint = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)

        result = dual_total_correlation(joint)
        assert torch.abs(result) < 1e-5

    def test_perfect_correlation_2d(self):
        """DTC > 0 for perfectly correlated variables."""
        # X = Y with probability 1
        joint = torch.zeros(2, 2)
        joint[0, 0] = 0.5
        joint[1, 1] = 0.5

        result = dual_total_correlation(joint)
        # For n=2 case, DTC = I(X;Y) = H(X) + H(Y) - H(X,Y) = log(2)
        expected = torch.log(torch.tensor(2.0))
        assert torch.abs(result - expected) < 1e-5

    def test_equals_mutual_information_for_2d(self):
        """For 2D, DTC equals mutual information I(X;Y)."""
        joint = torch.rand(4, 5)
        joint = joint / joint.sum()

        # DTC for bivariate case = I(X;Y)
        dtc = dual_total_correlation(joint)

        # Compute I(X;Y) = H(X) + H(Y) - H(X,Y) manually
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        eps = 1e-10
        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_joint = -(joint * torch.log(joint + eps)).sum()
        mi = h_x + h_y - h_joint

        assert torch.allclose(dtc, mi, rtol=1e-4)

    def test_dtc_formula_3d(self):
        """Verify DTC formula for 3D case.

        DTC = H(X,Y,Z) - sum_i H(X_i | X_{-i})
            = H(joint) - [H(X|Y,Z) + H(Y|X,Z) + H(Z|X,Y)]
            = (1-n) * H(joint) + sum_i H(X_{-i})

        where H(X_{-i}) is the entropy of the marginal over all variables except X_i.
        """
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        dtc = dual_total_correlation(joint)

        # Compute manually using the formula:
        # DTC = (1-n) * H(joint) + sum_i H(X_{-i})
        eps = 1e-10
        n = 3

        # Joint entropy
        h_joint = -(joint * torch.log(joint + eps)).sum()

        # Complementary marginals (marginals over all-but-one dimension)
        # For X: marginal over Y,Z
        p_yz = joint.sum(dim=0)  # Shape: (4, 5)
        h_yz = -(p_yz * torch.log(p_yz + eps)).sum()

        # For Y: marginal over X,Z
        p_xz = joint.sum(dim=1)  # Shape: (3, 5)
        h_xz = -(p_xz * torch.log(p_xz + eps)).sum()

        # For Z: marginal over X,Y
        p_xy = joint.sum(dim=2)  # Shape: (3, 4)
        h_xy = -(p_xy * torch.log(p_xy + eps)).sum()

        expected = (1 - n) * h_joint + h_yz + h_xz + h_xy

        assert torch.allclose(dtc, expected, rtol=1e-4)

    def test_dtc_vs_tc_relationship(self):
        """Verify the relationship between DTC and TC.

        For n variables:
        TC = sum_i H(X_i) - H(joint)
        DTC = H(joint) - sum_i H(X_i | X_{-i})

        Both should be >= 0 and equal for n=2.
        """
        from torchscience.information import total_correlation

        # For 2D case, TC = DTC = I(X;Y)
        joint_2d = torch.rand(4, 5)
        joint_2d = joint_2d / joint_2d.sum()
        tc_2d = total_correlation(joint_2d)
        dtc_2d = dual_total_correlation(joint_2d)
        assert torch.allclose(tc_2d, dtc_2d, rtol=1e-4)

        # For higher dimensions, they are generally different
        joint_3d = torch.rand(3, 4, 5)
        joint_3d = joint_3d / joint_3d.sum()
        tc_3d = total_correlation(joint_3d)
        dtc_3d = dual_total_correlation(joint_3d)

        # Both should be non-negative
        assert tc_3d >= -1e-6
        assert dtc_3d >= -1e-6

    def test_conditional_entropy_interpretation(self):
        """Verify DTC using conditional entropy interpretation.

        DTC = H(joint) - sum_i H(X_i | X_{-i})
            = H(joint) - sum_i [H(joint) - H(X_{-i})]
            = (1-n) * H(joint) + sum_i H(X_{-i})
        """
        joint = torch.rand(2, 3)
        joint = joint / joint.sum()

        dtc = dual_total_correlation(joint)

        eps = 1e-10
        n = 2

        # Joint entropy
        h_joint = -(joint * torch.log(joint + eps)).sum()

        # Marginals
        p_y = joint.sum(dim=0)  # Marginal over X (gives Y)
        p_x = joint.sum(dim=1)  # Marginal over Y (gives X)

        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_x = -(p_x * torch.log(p_x + eps)).sum()

        # Conditional entropies
        h_x_given_y = h_joint - h_y  # H(X|Y)
        h_y_given_x = h_joint - h_x  # H(Y|X)

        # DTC = H(joint) - [H(X|Y) + H(Y|X)]
        expected = h_joint - (h_x_given_y + h_y_given_x)

        assert torch.allclose(dtc, expected, rtol=1e-4)


class TestDualTotalCorrelationGradients:
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
            return dual_total_correlation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_3d(self):
        """First-order gradients are correct for 3D."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return dual_total_correlation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 4, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = dual_total_correlation(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = dual_total_correlation(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestDualTotalCorrelationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="input_type"):
            dual_total_correlation(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="reduction"):
            dual_total_correlation(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            dual_total_correlation(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <2 dimensions raises ValueError."""
        joint = torch.rand(5)
        with pytest.raises(ValueError, match="2 dimensions"):
            dual_total_correlation(joint)


class TestDualTotalCorrelationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.rand(3, 4, device="meta")
        result = dual_total_correlation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output

    def test_meta_tensor_3d(self):
        """Meta tensor works for 3D input."""
        joint = torch.rand(2, 3, 4, device="meta")
        result = dual_total_correlation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0
