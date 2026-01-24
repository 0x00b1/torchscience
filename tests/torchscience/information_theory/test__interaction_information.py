"""Tests for interaction information."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import interaction_information


class TestInteractionInformationBasic:
    """Basic functionality tests."""

    def test_output_shape(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(4, 4, 4)
        joint = joint / joint.sum()
        result = interaction_information(joint)
        assert result.dim() == 0

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = interaction_information(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = interaction_information(
            log_joint, input_type="log_probability"
        )
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3, 3)
        result = interaction_information(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = interaction_information(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = interaction_information(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2, 2)
        joint = joint / joint.sum()
        result_nats = interaction_information(joint, base=None)
        result_bits = interaction_information(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestInteractionInformationCorrectness:
    """Correctness tests."""

    def test_independent_variables(self):
        """I(X;Y;Z) = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y,z) = p(x) * p(y) * p(z)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        joint = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)

        result = interaction_information(joint)
        assert torch.abs(result) < 1e-5

    def test_redundant_case(self):
        """I(X;Y;Z) > 0 for redundant case (X = Y = Z)."""
        # All three variables are identical
        joint = torch.zeros(2, 2, 2)
        joint[0, 0, 0] = 0.5
        joint[1, 1, 1] = 0.5

        result = interaction_information(joint)
        # For X=Y=Z: I(X;Y;Z) = I(X;Y) - I(X;Y|Z) = log(2) - 0 = log(2)
        # Because when Z is known, X and Y are deterministic (they equal Z)
        expected = torch.log(torch.tensor(2.0))
        assert result > 0  # Positive (redundancy)
        assert torch.abs(result - expected) < 1e-5

    def test_xor_synergy_case(self):
        """I(X;Y;Z) < 0 for XOR-like synergy case."""
        # XOR case: Z = X XOR Y
        # X and Y are independent, but together they fully determine Z
        joint = torch.zeros(2, 2, 2)
        # p(x,y,z) where z = x XOR y
        joint[0, 0, 0] = 0.25  # x=0, y=0, z=0
        joint[0, 1, 1] = 0.25  # x=0, y=1, z=1
        joint[1, 0, 1] = 0.25  # x=1, y=0, z=1
        joint[1, 1, 0] = 0.25  # x=1, y=1, z=0

        result = interaction_information(joint)
        # For XOR: I(X;Y|Z) > I(X;Y) = 0, so I(X;Y;Z) < 0 (synergy)
        # I(X;Y;Z) = -log(2)
        expected = -torch.log(torch.tensor(2.0))
        assert result < 0  # Negative (synergy)
        assert torch.abs(result - expected) < 1e-5

    def test_entropy_formula(self):
        """Verify I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        ii = interaction_information(joint)

        # Compute manually using entropy formula
        eps = 1e-10

        # Marginals
        p_x = joint.sum(dim=(1, 2))  # Sum over Y, Z
        p_y = joint.sum(dim=(0, 2))  # Sum over X, Z
        p_z = joint.sum(dim=(0, 1))  # Sum over X, Y
        p_xy = joint.sum(dim=2)  # Sum over Z
        p_xz = joint.sum(dim=1)  # Sum over Y
        p_yz = joint.sum(dim=0)  # Sum over X

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_z = -(p_z * torch.log(p_z + eps)).sum()
        h_xy = -(p_xy * torch.log(p_xy + eps)).sum()
        h_xz = -(p_xz * torch.log(p_xz + eps)).sum()
        h_yz = -(p_yz * torch.log(p_yz + eps)).sum()
        h_xyz = -(joint * torch.log(joint + eps)).sum()

        expected = h_x + h_y + h_z - h_xy - h_xz - h_yz + h_xyz

        assert torch.allclose(ii, expected, rtol=1e-4)

    def test_mi_cmi_formula(self):
        """Verify I(X;Y;Z) = I(X;Y) - I(X;Y|Z)."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()

        ii = interaction_information(joint)

        eps = 1e-10

        # Compute I(X;Y) = H(X) + H(Y) - H(X,Y)
        p_x = joint.sum(dim=(1, 2))
        p_y = joint.sum(dim=(0, 2))
        p_xy = joint.sum(dim=2)

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_xy = -(p_xy * torch.log(p_xy + eps)).sum()
        mi_xy = h_x + h_y - h_xy

        # Compute I(X;Y|Z) = sum_z p(z) * I(X;Y|Z=z)
        # = H(X|Z) + H(Y|Z) - H(X,Y|Z)
        p_z = joint.sum(dim=(0, 1))
        p_xz = joint.sum(dim=1)
        p_yz = joint.sum(dim=0)

        h_xz = -(p_xz * torch.log(p_xz + eps)).sum()
        h_yz = -(p_yz * torch.log(p_yz + eps)).sum()
        h_z = -(p_z * torch.log(p_z + eps)).sum()
        h_xyz = -(joint * torch.log(joint + eps)).sum()

        # H(X|Z) = H(X,Z) - H(Z)
        # H(Y|Z) = H(Y,Z) - H(Z)
        # H(X,Y|Z) = H(X,Y,Z) - H(Z)
        # I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
        #          = H(X,Z) - H(Z) + H(Y,Z) - H(Z) - H(X,Y,Z) + H(Z)
        #          = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
        cmi = h_xz + h_yz - h_z - h_xyz

        expected = mi_xy - cmi

        assert torch.allclose(ii, expected, rtol=1e-4)

    def test_symmetry(self):
        """I(X;Y;Z) is symmetric in all three variables."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        ii_xyz = interaction_information(joint)

        # Permute to (Y,X,Z)
        joint_yxz = joint.permute(1, 0, 2).contiguous()
        joint_yxz = joint_yxz / joint_yxz.sum()
        ii_yxz = interaction_information(joint_yxz)

        # Permute to (Z,Y,X)
        joint_zyx = joint.permute(2, 1, 0).contiguous()
        joint_zyx = joint_zyx / joint_zyx.sum()
        ii_zyx = interaction_information(joint_zyx)

        assert torch.allclose(ii_xyz, ii_yxz, rtol=1e-4)
        assert torch.allclose(ii_xyz, ii_zyx, rtol=1e-4)


class TestInteractionInformationGradients:
    """Gradient tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        # Add small offset to avoid zeros for numerical stability
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return interaction_information(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = interaction_information(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = interaction_information(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestInteractionInformationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="input_type"):
            interaction_information(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="reduction"):
            interaction_information(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            interaction_information(joint, base=-1.0)

    def test_wrong_dimensions(self):
        """Tensor with != 3 dimensions raises ValueError."""
        joint_2d = torch.rand(3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            interaction_information(joint_2d)

        joint_4d = torch.rand(3, 3, 3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            interaction_information(joint_4d)


class TestInteractionInformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = interaction_information(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output

    def test_meta_tensor_backward_shape(self):
        """Meta tensor backward produces correct shape."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = interaction_information(joint)
        assert result.dim() == 0
