"""Tests for coinformation."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import coinformation


class TestCoinformationBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Output is scalar for 2D input."""
        joint = torch.rand(4, 4)
        joint = joint / joint.sum()
        result = coinformation(joint)
        assert result.dim() == 0

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(4, 4, 4)
        joint = joint / joint.sum()
        result = coinformation(joint)
        assert result.dim() == 0

    def test_output_shape_4d(self):
        """Output is scalar for 4D input."""
        joint = torch.rand(3, 3, 3, 3)
        joint = joint / joint.sum()
        result = coinformation(joint)
        assert result.dim() == 0

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = coinformation(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = coinformation(log_joint, input_type="log_probability")
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3)
        result = coinformation(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = coinformation(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = coinformation(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2)
        joint = joint / joint.sum()
        result_nats = coinformation(joint, base=None)
        result_bits = coinformation(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestCoinformationCorrectness:
    """Correctness tests."""

    def test_2d_equals_mutual_information(self):
        """For 2 variables, coinformation equals mutual information I(X;Y)."""
        joint = torch.rand(3, 4)
        joint = joint / joint.sum()

        ci = coinformation(joint)

        # Compute I(X;Y) = H(X) + H(Y) - H(X,Y) manually
        eps = 1e-10
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_x = -(p_x * torch.log(p_x + eps)).sum()
        h_y = -(p_y * torch.log(p_y + eps)).sum()
        h_xy = -(joint * torch.log(joint + eps)).sum()

        mi = h_x + h_y - h_xy

        assert torch.allclose(ci, mi, rtol=1e-4)

    def test_3d_equals_interaction_information(self):
        """For 3 variables, coinformation equals interaction information I(X;Y;Z)."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()

        ci = coinformation(joint)

        # Compute I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
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

        ii = h_x + h_y + h_z - h_xy - h_xz - h_yz + h_xyz

        assert torch.allclose(ci, ii, rtol=1e-4)

    def test_independent_2d(self):
        """CI(X;Y) = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y) = p(x) * p(y)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        joint = torch.outer(p_x, p_y)

        result = coinformation(joint)
        assert torch.abs(result) < 1e-5

    def test_independent_3d(self):
        """CI(X;Y;Z) = 0 for independent variables (product distribution)."""
        # Create independent joint: p(x,y,z) = p(x) * p(y) * p(z)
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        joint = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)

        result = coinformation(joint)
        assert torch.abs(result) < 1e-5

    def test_independent_4d(self):
        """CI(X;Y;Z;W) = 0 for independent variables."""
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        p_z = torch.tensor([0.5, 0.5])
        p_w = torch.tensor([0.2, 0.8])
        joint = torch.einsum("i,j,k,l->ijkl", p_x, p_y, p_z, p_w)

        result = coinformation(joint)
        assert torch.abs(result) < 1e-4

    def test_perfectly_correlated_2d(self):
        """CI = log(n) for perfectly correlated 2D case (X = Y)."""
        joint = torch.zeros(2, 2)
        joint[0, 0] = joint[1, 1] = 0.5

        result = coinformation(joint)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(result, expected, rtol=1e-4)

    def test_redundant_3d(self):
        """CI > 0 for redundant 3D case (X = Y = Z)."""
        # All three variables are identical
        joint = torch.zeros(2, 2, 2)
        joint[0, 0, 0] = 0.5
        joint[1, 1, 1] = 0.5

        result = coinformation(joint)
        # For X=Y=Z: I(X;Y;Z) = log(2)
        expected = torch.log(torch.tensor(2.0))
        assert result > 0  # Positive (redundancy)
        assert torch.allclose(result, expected, rtol=1e-4)

    def test_xor_synergy_case(self):
        """CI < 0 for XOR-like synergy case."""
        # XOR case: Z = X XOR Y
        # X and Y are independent, but together they fully determine Z
        joint = torch.zeros(2, 2, 2)
        # p(x,y,z) where z = x XOR y
        joint[0, 0, 0] = 0.25  # x=0, y=0, z=0
        joint[0, 1, 1] = 0.25  # x=0, y=1, z=1
        joint[1, 0, 1] = 0.25  # x=1, y=0, z=1
        joint[1, 1, 0] = 0.25  # x=1, y=1, z=0

        result = coinformation(joint)
        # For XOR: I(X;Y;Z) = -log(2)
        expected = -torch.log(torch.tensor(2.0))
        assert result < 0  # Negative (synergy)
        assert torch.allclose(result, expected, rtol=1e-4)

    def test_symmetry_2d(self):
        """CI(X;Y) is symmetric."""
        joint = torch.rand(3, 4)
        joint = joint / joint.sum()

        ci_xy = coinformation(joint)

        # Transpose
        joint_yx = joint.T.contiguous()
        ci_yx = coinformation(joint_yx)

        assert torch.allclose(ci_xy, ci_yx, rtol=1e-4)

    def test_symmetry_3d(self):
        """CI(X;Y;Z) is symmetric in all three variables."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        ci_xyz = coinformation(joint)

        # Permute to (Y,X,Z)
        joint_yxz = joint.permute(1, 0, 2).contiguous()
        joint_yxz = joint_yxz / joint_yxz.sum()
        ci_yxz = coinformation(joint_yxz)

        # Permute to (Z,Y,X)
        joint_zyx = joint.permute(2, 1, 0).contiguous()
        joint_zyx = joint_zyx / joint_zyx.sum()
        ci_zyx = coinformation(joint_zyx)

        assert torch.allclose(ci_xyz, ci_yxz, rtol=1e-4)
        assert torch.allclose(ci_xyz, ci_zyx, rtol=1e-4)


class TestCoinformationGradients:
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
            return coinformation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_3d(self):
        """First-order gradients are correct for 3D."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return coinformation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_4d(self):
        """First-order gradients are correct for 4D."""
        joint = torch.rand(2, 2, 2, 2, dtype=torch.float64)
        joint = joint / joint.sum()
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return coinformation(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = coinformation(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = coinformation(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestCoinformationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="input_type"):
            coinformation(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="reduction"):
            coinformation(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            coinformation(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with < 2 dimensions raises ValueError."""
        joint_1d = torch.rand(3)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            coinformation(joint_1d)

    def test_too_many_dimensions(self):
        """Tensor with > 10 dimensions raises ValueError."""
        joint_11d = torch.rand(*([2] * 11))
        with pytest.raises(ValueError, match="at most 10 dimensions"):
            coinformation(joint_11d)


class TestCoinformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape_2d(self):
        """Meta tensor produces correct output shape for 2D."""
        joint = torch.rand(3, 4, device="meta")
        result = coinformation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output

    def test_meta_tensor_shape_3d(self):
        """Meta tensor produces correct output shape for 3D."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = coinformation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0

    def test_meta_tensor_shape_4d(self):
        """Meta tensor produces correct output shape for 4D."""
        joint = torch.rand(3, 4, 5, 2, device="meta")
        result = coinformation(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0
