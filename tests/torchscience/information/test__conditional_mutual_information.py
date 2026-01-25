"""Tests for conditional mutual information."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import conditional_mutual_information


class TestConditionalMutualInformationBasic:
    """Basic functionality tests."""

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(4, 4, 4)
        joint = joint / joint.sum()
        result = conditional_mutual_information(joint)
        assert result.dim() == 0

    def test_non_negative(self):
        """Conditional mutual information is non-negative."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = conditional_mutual_information(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = conditional_mutual_information(
            joint, input_type="probability"
        )
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = conditional_mutual_information(
            log_joint, input_type="log_probability"
        )
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3, 3)
        result = conditional_mutual_information(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = conditional_mutual_information(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = conditional_mutual_information(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2, 2)
        joint = joint / joint.sum()
        result_nats = conditional_mutual_information(joint, base=None)
        result_bits = conditional_mutual_information(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestConditionalMutualInformationCorrectness:
    """Correctness tests."""

    def test_independent_given_z(self):
        """I(X;Y|Z) = 0 when X and Y are independent given Z."""
        # Create joint where X and Y are independent conditioned on Z
        # p(x,y|z) = p(x|z) * p(y|z)
        joint = torch.zeros(2, 2, 2)
        # For z=0: p(x|z=0) = [0.5, 0.5], p(y|z=0) = [0.5, 0.5]
        # For z=1: p(x|z=1) = [0.5, 0.5], p(y|z=1) = [0.5, 0.5]
        # p(z) = [0.5, 0.5]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    joint[x, y, z] = 0.5 * 0.5 * 0.5  # p(x|z)*p(y|z)*p(z)

        result = conditional_mutual_information(joint)
        assert torch.abs(result) < 1e-5

    def test_perfect_correlation_given_z(self):
        """I(X;Y|Z) > 0 when X and Y are correlated given Z."""
        joint = torch.zeros(2, 2, 2)
        # X = Y conditioned on Z (perfect correlation)
        joint[0, 0, 0] = 0.25
        joint[1, 1, 0] = 0.25
        joint[0, 0, 1] = 0.25
        joint[1, 1, 1] = 0.25

        result = conditional_mutual_information(joint)
        # Should equal H(X|Z) = H(Y|Z) = log(2) for binary uniform marginals
        expected = torch.log(torch.tensor(2.0))
        assert torch.abs(result - expected) < 1e-5

    def test_chain_rule(self):
        """Verify I(X;Y|Z) = H(X|Z) - H(X|Y,Z)."""
        # Create a non-trivial joint distribution
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()

        # Compute CMI directly
        cmi = conditional_mutual_information(joint)

        # Compute via chain rule: I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
        # H(X|Z) = -sum_{x,z} p(x,z) log p(x|z)
        p_xz = joint.sum(dim=1)  # Sum over Y
        p_z = p_xz.sum(dim=0)  # Sum over X

        eps = 1e-10
        # H(X|Z) = -sum_{x,z} p(x,z) log p(x|z) = -sum_{x,z} p(x,z) log(p(x,z)/p(z))
        h_x_given_z = 0.0
        for x in range(3):
            for z in range(3):
                if p_xz[x, z] > eps and p_z[z] > eps:
                    h_x_given_z -= p_xz[x, z] * torch.log(p_xz[x, z] / p_z[z])

        # H(X|Y,Z) = -sum_{x,y,z} p(x,y,z) log p(x|y,z) = -sum_{x,y,z} p(x,y,z) log(p(x,y,z)/p(y,z))
        p_yz = joint.sum(dim=0)  # Sum over X
        h_x_given_yz = 0.0
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if joint[x, y, z] > eps and p_yz[y, z] > eps:
                        h_x_given_yz -= joint[x, y, z] * torch.log(
                            joint[x, y, z] / p_yz[y, z]
                        )

        expected_cmi = h_x_given_z - h_x_given_yz
        assert torch.abs(cmi - expected_cmi) < 1e-4

    def test_symmetry_xy(self):
        """I(X;Y|Z) = I(Y;X|Z) - symmetry in X and Y."""
        joint = torch.rand(3, 4, 5)
        joint = joint / joint.sum()

        # I(X;Y|Z) with dims_x=(0,), dims_y=(1,)
        cmi_xy = conditional_mutual_information(
            joint, dims_x=(0,), dims_y=(1,), dims_z=(2,)
        )

        # I(Y;X|Z) - swap X and Y by permuting joint and swapping dims
        joint_permuted = joint.permute(1, 0, 2)
        cmi_yx = conditional_mutual_information(
            joint_permuted, dims_x=(0,), dims_y=(1,), dims_z=(2,)
        )

        assert torch.allclose(cmi_xy, cmi_yx, rtol=1e-4)


class TestConditionalMutualInformationGradients:
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
            return conditional_mutual_information(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = conditional_mutual_information(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = conditional_mutual_information(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape


class TestConditionalMutualInformationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="input_type"):
            conditional_mutual_information(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="reduction"):
            conditional_mutual_information(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            conditional_mutual_information(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <3 dimensions raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            conditional_mutual_information(joint)

    def test_overlapping_dims(self):
        """Overlapping dimension specs raise ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="overlap"):
            conditional_mutual_information(
                joint, dims_x=(0,), dims_y=(0,), dims_z=(1,)
            )


class TestConditionalMutualInformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor produces correct output shape."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = conditional_mutual_information(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output for 3D input
