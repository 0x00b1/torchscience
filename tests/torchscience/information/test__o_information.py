"""Tests for O-information."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import (
    dual_total_correlation,
    o_information,
    total_correlation,
)


class TestOInformationBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Output is scalar for 2D input."""
        joint = torch.rand(4, 4)
        joint = joint / joint.sum()
        result = o_information(joint)
        assert result.dim() == 0

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = o_information(joint)
        assert result.dim() == 0

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = o_information(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = o_information(log_joint, input_type="log_probability")
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3)
        result = o_information(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = o_information(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = o_information(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result_nats = o_information(joint, base=None)
        result_bits = o_information(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestOInformationCorrectness:
    """Correctness tests."""

    def test_independent_variables_2d(self):
        """O-information = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y) = p(x) * p(y)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        joint = torch.outer(p_x, p_y)

        result = o_information(joint)
        # For independent variables, TC = 0 and DTC = 0, so O = 0
        assert torch.abs(result) < 1e-5

    def test_independent_variables_3d(self):
        """O-information = 0 for independent 3D joint."""
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        # p(x,y,z) = p(x) * p(y) * p(z)
        joint = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)

        result = o_information(joint)
        # For independent variables, TC = 0 and DTC = 0, so O = 0
        assert torch.abs(result) < 1e-5

    def test_bivariate_is_zero(self):
        """For 2D, O-information = 0 since TC = DTC = I(X;Y)."""
        joint = torch.rand(4, 5)
        joint = joint / joint.sum()

        result = o_information(joint)
        # For n=2, TC = DTC = I(X;Y), so O-information = 0
        assert torch.abs(result) < 1e-5

    def test_equals_tc_minus_dtc(self):
        """Verify O-information = TC - DTC."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        o_info = o_information(joint)
        tc = total_correlation(joint)
        dtc = dual_total_correlation(joint)

        expected = tc - dtc
        assert torch.allclose(o_info, expected, rtol=1e-5)

    def test_redundant_system(self):
        """O-information > 0 for redundancy-dominated system.

        A common source structure: X -> Z, Y -> Z independent, with X = Y.
        When both inputs carry the same information about Z, we have redundancy.
        """
        # Create a joint distribution where X and Y are perfectly correlated
        # This gives high redundancy: O > 0
        # P(X=0,Y=0,Z=0) = P(X=1,Y=1,Z=1) = 0.5
        joint = torch.zeros(2, 2, 2)
        joint[0, 0, 0] = 0.5
        joint[1, 1, 1] = 0.5

        result = o_information(joint)
        # With perfectly correlated variables, we expect positive O-information
        # TC captures all pairwise + higher correlations
        # DTC captures binding across variables
        # For redundant systems, TC > DTC
        assert result > 0

    def test_xor_like_synergy(self):
        """O-information < 0 for synergy-dominated system (XOR-like).

        For XOR: Z = X XOR Y
        - Knowing X alone tells nothing about Z
        - Knowing Y alone tells nothing about Z
        - Knowing both X and Y determines Z completely

        This is the classic synergy example.
        """
        # XOR joint distribution: P(X, Y, Z)
        # Z = X XOR Y with uniform X, Y
        joint = torch.zeros(2, 2, 2)
        # P(X=0, Y=0, Z=0) = 0.25
        joint[0, 0, 0] = 0.25
        # P(X=0, Y=1, Z=1) = 0.25
        joint[0, 1, 1] = 0.25
        # P(X=1, Y=0, Z=1) = 0.25
        joint[1, 0, 1] = 0.25
        # P(X=1, Y=1, Z=0) = 0.25
        joint[1, 1, 0] = 0.25

        result = o_information(joint)
        # For XOR, synergy dominates: O-information < 0
        # TC = sum of marginal entropies - joint entropy
        # DTC = joint entropy - sum of conditional entropies
        # For XOR-like distributions, DTC > TC
        assert result < 0

    def test_sign_interpretation(self):
        """Verify O-information sign interpretation for different distributions."""
        # Test that O-information captures the redundancy/synergy balance

        # Case 1: Bivariate - always 0
        joint_2d = torch.rand(3, 3)
        joint_2d = joint_2d / joint_2d.sum()
        o_2d = o_information(joint_2d)
        assert torch.abs(o_2d) < 1e-5

        # Case 2: Random 3D - can be positive or negative
        joint_3d = torch.rand(3, 3, 3)
        joint_3d = joint_3d / joint_3d.sum()
        o_3d = o_information(joint_3d)
        assert torch.isfinite(o_3d)


class TestOInformationFormula:
    """Tests for the O-information formula."""

    def test_formula_alternative_form(self):
        """Verify the alternative form of O-information.

        O = TC - DTC
          = [sum_i H(X_i) - H(joint)] - [H(joint) - sum_i H(X_i|X_{-i})]
          = sum_i H(X_i) - 2*H(joint) + sum_i H(X_i|X_{-i})
          = sum_i H(X_i) - 2*H(joint) + sum_i [H(joint) - H(X_{-i})]
          = sum_i H(X_i) + (n-2)*H(joint) - sum_i H(X_{-i})
        """
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        o_info = o_information(joint)

        # Compute using the expanded formula
        eps = 1e-10
        n = 3

        # Joint entropy
        h_joint = -(joint * torch.log(joint + eps)).sum()

        # Marginal entropies
        p_x = joint.sum(dim=(1, 2))
        p_y = joint.sum(dim=(0, 2))
        p_z = joint.sum(dim=(0, 1))

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_z = -(p_z * torch.log(p_z + eps)).sum()

        # Complementary marginal entropies
        p_yz = joint.sum(dim=0)
        p_xz = joint.sum(dim=1)
        p_xy = joint.sum(dim=2)

        h_yz = -(p_yz * torch.log(p_yz + eps)).sum()
        h_xz = -(p_xz * torch.log(p_xz + eps)).sum()
        h_xy = -(p_xy * torch.log(p_xy + eps)).sum()

        # O = sum_i H(X_i) + (n-2)*H(joint) - sum_i H(X_{-i})
        expected = (h_x + h_y + h_z) + (n - 2) * h_joint - (h_yz + h_xz + h_xy)

        assert torch.allclose(o_info, expected, rtol=1e-4)

    def test_trivariate_explicit(self):
        """Verify O-information for trivariate case with explicit formulas.

        For n=3:
        TC = H(X) + H(Y) + H(Z) - H(X,Y,Z)
        DTC = H(X,Y,Z) - H(X|Y,Z) - H(Y|X,Z) - H(Z|X,Y)
            = -2*H(X,Y,Z) + H(X,Y) + H(X,Z) + H(Y,Z)

        O = TC - DTC
          = H(X) + H(Y) + H(Z) - H(X,Y,Z) - (-2*H(X,Y,Z) + H(X,Y) + H(X,Z) + H(Y,Z))
          = H(X) + H(Y) + H(Z) + H(X,Y,Z) - H(X,Y) - H(X,Z) - H(Y,Z)
          = -I(X;Y;Z)  (negative interaction information for n=3)
        """
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        o_info = o_information(joint)

        # Compute interaction information
        eps = 1e-10

        # Individual entropies
        p_x = joint.sum(dim=(1, 2))
        p_y = joint.sum(dim=(0, 2))
        p_z = joint.sum(dim=(0, 1))

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_z = -(p_z * torch.log(p_z + eps)).sum()

        # Pairwise joint entropies
        p_xy = joint.sum(dim=2)
        p_xz = joint.sum(dim=1)
        p_yz = joint.sum(dim=0)

        h_xy = -(p_xy * torch.log(p_xy + eps)).sum()
        h_xz = -(p_xz * torch.log(p_xz + eps)).sum()
        h_yz = -(p_yz * torch.log(p_yz + eps)).sum()

        # Joint entropy
        h_xyz = -(joint * torch.log(joint + eps)).sum()

        # Interaction information I(X;Y;Z)
        # I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
        #          = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
        interaction_info = h_x + h_y + h_z - h_xy - h_xz - h_yz + h_xyz

        # O-information = -I(X;Y;Z) for n=3
        # (Note: this relationship depends on sign conventions)
        # Actually: O = (n-2)*I(X_1;...;X_n) for n >= 2
        # For n=3: O = I(X;Y;Z)
        expected = interaction_info

        assert torch.allclose(o_info, expected, rtol=1e-4)


class TestOInformationGradients:
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
            return o_information(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_3d(self):
        """First-order gradients are correct for 3D."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return o_information(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 4, 5, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = o_information(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = o_information(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestOInformationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="input_type"):
            o_information(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="reduction"):
            o_information(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            o_information(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <2 dimensions raises ValueError."""
        joint = torch.rand(5)
        with pytest.raises(ValueError, match="2 dimensions"):
            o_information(joint)

    def test_non_tensor_input(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            o_information([0.5, 0.5])


class TestOInformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.rand(3, 4, device="meta")
        result = o_information(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output

    def test_meta_tensor_3d(self):
        """Meta tensor works for 3D input."""
        joint = torch.rand(2, 3, 4, device="meta")
        result = o_information(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0
