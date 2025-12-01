"""Tests for mathematical relationships between information-theoretic operators.

These tests verify that the implementations satisfy the fundamental identities
from information theory:

1. I(X;Y) = H(X) + H(Y) - H(X,Y)           (MI via joint entropy)
2. I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)  (MI via conditional entropy)
3. H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)  (chain rule)
4. E[PMI(X,Y)] = I(X;Y)                    (expected PMI = MI)
5. I(X;Y) >= 0                             (non-negativity)
6. I(X;Y) = 0 for independent X and Y
"""

import torch

from torchscience.information_theory import (
    conditional_entropy,
    joint_entropy,
    mutual_information,
    pointwise_mutual_information,
    shannon_entropy,
)


class TestMutualInformationViaJointEntropy:
    """Test I(X;Y) = H(X) + H(Y) - H(X,Y)."""

    def test_identity_2x2(self) -> None:
        """Test MI identity for 2x2 distribution."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_x = joint.sum(dim=1)  # marginal p(x)
        p_y = joint.sum(dim=0)  # marginal p(y)

        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        h_xy = joint_entropy(joint)
        mi = mutual_information(joint)

        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        expected_mi = h_x + h_y - h_xy

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_identity_3x4(self) -> None:
        """Test MI identity for 3x4 distribution."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        h_xy = joint_entropy(joint)
        mi = mutual_information(joint)

        expected_mi = h_x + h_y - h_xy

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_identity_batched(self) -> None:
        """Test MI identity for batched distributions."""
        batch = 5
        joint = torch.rand(batch, 3, 4, dtype=torch.float64)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)
        p_x = joint.sum(dim=-1)  # (batch, 3)
        p_y = joint.sum(dim=-2)  # (batch, 4)

        h_x = shannon_entropy(p_x, dim=-1)
        h_y = shannon_entropy(p_y, dim=-1)
        h_xy = joint_entropy(joint, dims=(-2, -1))
        mi = mutual_information(joint, dims=(-2, -1))

        expected_mi = h_x + h_y - h_xy

        assert torch.allclose(mi, expected_mi, atol=1e-6)


class TestMutualInformationViaConditionalEntropy:
    """Test I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)."""

    def test_mi_equals_h_x_minus_h_x_given_y(self) -> None:
        """Test I(X;Y) = H(X) - H(X|Y)."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_x = joint.sum(dim=1)

        h_x = shannon_entropy(p_x, dim=0)
        h_x_given_y = conditional_entropy(joint, condition_dim=1, target_dim=0)
        mi = mutual_information(joint)

        expected_mi = h_x - h_x_given_y

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_mi_equals_h_y_minus_h_y_given_x(self) -> None:
        """Test I(X;Y) = H(Y) - H(Y|X)."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_y = joint.sum(dim=0)

        h_y = shannon_entropy(p_y, dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)
        mi = mutual_information(joint)

        expected_mi = h_y - h_y_given_x

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_both_forms_equivalent(self) -> None:
        """Test H(X) - H(X|Y) = H(Y) - H(Y|X)."""
        joint = torch.rand(4, 5, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        h_x_given_y = conditional_entropy(joint, condition_dim=1, target_dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)

        mi_form1 = h_x - h_x_given_y
        mi_form2 = h_y - h_y_given_x

        assert torch.allclose(mi_form1, mi_form2, atol=1e-6)


class TestChainRule:
    """Test H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)."""

    def test_chain_rule_x_then_y(self) -> None:
        """Test H(X,Y) = H(X) + H(Y|X)."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_x = joint.sum(dim=1)

        h_xy = joint_entropy(joint)
        h_x = shannon_entropy(p_x, dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)

        expected_h_xy = h_x + h_y_given_x

        assert torch.allclose(h_xy, expected_h_xy, atol=1e-6)

    def test_chain_rule_y_then_x(self) -> None:
        """Test H(X,Y) = H(Y) + H(X|Y)."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_y = joint.sum(dim=0)

        h_xy = joint_entropy(joint)
        h_y = shannon_entropy(p_y, dim=0)
        h_x_given_y = conditional_entropy(joint, condition_dim=1, target_dim=0)

        expected_h_xy = h_y + h_x_given_y

        assert torch.allclose(h_xy, expected_h_xy, atol=1e-6)

    def test_chain_rule_random_distribution(self) -> None:
        """Test chain rule for random distribution."""
        joint = torch.rand(5, 6, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_xy = joint_entropy(joint)
        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        h_x_given_y = conditional_entropy(joint, condition_dim=1, target_dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)

        # Both forms should equal H(X,Y)
        assert torch.allclose(h_xy, h_x + h_y_given_x, atol=1e-6)
        assert torch.allclose(h_xy, h_y + h_x_given_y, atol=1e-6)


class TestExpectedPMIEqualsMI:
    """Test E[PMI(X,Y)] = I(X;Y)."""

    def test_expected_pmi_2x2(self) -> None:
        """Test expected PMI equals MI for 2x2 distribution."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)

        pmi = pointwise_mutual_information(joint)
        mi = mutual_information(joint)

        # E[PMI] = sum p(x,y) * PMI(x,y)
        expected_mi = (joint * pmi).sum()

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_expected_pmi_random(self) -> None:
        """Test expected PMI equals MI for random distribution."""
        joint = torch.rand(4, 5, dtype=torch.float64)
        joint = joint / joint.sum()

        pmi = pointwise_mutual_information(joint)
        mi = mutual_information(joint)

        expected_mi = (joint * pmi).sum()

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_expected_pmi_batched(self) -> None:
        """Test expected PMI equals MI for batched distributions."""
        batch = 3
        joint = torch.rand(batch, 4, 5, dtype=torch.float64)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)

        pmi = pointwise_mutual_information(joint, dims=(-2, -1))
        mi = mutual_information(joint, dims=(-2, -1))

        expected_mi = (joint * pmi).sum(dim=(-2, -1))

        assert torch.allclose(mi, expected_mi, atol=1e-6)


class TestNonNegativity:
    """Test I(X;Y) >= 0."""

    def test_mi_non_negative_random(self) -> None:
        """Test MI is non-negative for random distributions."""
        for _ in range(10):
            joint = torch.rand(3, 4, dtype=torch.float64)
            joint = joint / joint.sum()

            mi = mutual_information(joint)

            assert mi >= -1e-10  # Allow small numerical error

    def test_mi_non_negative_near_deterministic(self) -> None:
        """Test MI is non-negative for near-deterministic distributions."""
        # Almost deterministic X=Y
        joint = torch.tensor([[0.49, 0.01], [0.01, 0.49]], dtype=torch.float64)

        mi = mutual_information(joint)

        assert mi >= 0

    def test_conditional_entropy_non_negative(self) -> None:
        """Test H(Y|X) >= 0."""
        for _ in range(10):
            joint = torch.rand(3, 4, dtype=torch.float64)
            joint = joint / joint.sum()

            h_y_given_x = conditional_entropy(
                joint, condition_dim=0, target_dim=1
            )

            assert h_y_given_x >= -1e-10


class TestIndependence:
    """Test I(X;Y) = 0 for independent X and Y."""

    def test_mi_zero_for_independent(self) -> None:
        """Test MI is zero for independent distributions."""
        p_x = torch.tensor([0.3, 0.7], dtype=torch.float64)
        p_y = torch.tensor([0.4, 0.6], dtype=torch.float64)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        mi = mutual_information(joint)

        assert torch.allclose(
            mi, torch.tensor(0.0, dtype=torch.float64), atol=1e-6
        )

    def test_pmi_zero_for_independent(self) -> None:
        """Test PMI is zero everywhere for independent distributions."""
        p_x = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        p_y = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float64)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        pmi = pointwise_mutual_information(joint)

        assert torch.allclose(pmi, torch.zeros_like(pmi), atol=1e-5)

    def test_conditional_entropy_equals_marginal_for_independent(self) -> None:
        """Test H(Y|X) = H(Y) for independent X and Y."""
        p_x = torch.tensor([0.3, 0.7], dtype=torch.float64)
        p_y = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        h_y = shannon_entropy(p_y, dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)

        assert torch.allclose(h_y, h_y_given_x, atol=1e-6)


class TestSymmetry:
    """Test symmetry properties."""

    def test_mi_symmetric(self) -> None:
        """Test I(X;Y) = I(Y;X)."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()

        mi_xy = mutual_information(joint, dims=(0, 1))
        mi_yx = mutual_information(joint.T, dims=(0, 1))

        assert torch.allclose(mi_xy, mi_yx, atol=1e-6)

    def test_joint_entropy_symmetric(self) -> None:
        """Test H(X,Y) = H(Y,X)."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()

        h_xy = joint_entropy(joint, dims=(0, 1))
        h_yx = joint_entropy(joint.T, dims=(0, 1))

        assert torch.allclose(h_xy, h_yx, atol=1e-6)


class TestBounds:
    """Test entropy and MI bounds."""

    def test_mi_bounded_by_min_entropy(self) -> None:
        """Test I(X;Y) <= min(H(X), H(Y))."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        mi = mutual_information(joint)

        assert mi <= min(h_x.item(), h_y.item()) + 1e-6

    def test_conditional_entropy_bounded_by_joint(self) -> None:
        """Test H(Y|X) <= H(Y) and H(X|Y) <= H(X)."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        h_x = shannon_entropy(p_x, dim=0)
        h_y = shannon_entropy(p_y, dim=0)
        h_x_given_y = conditional_entropy(joint, condition_dim=1, target_dim=0)
        h_y_given_x = conditional_entropy(joint, condition_dim=0, target_dim=1)

        assert h_y_given_x <= h_y + 1e-6
        assert h_x_given_y <= h_x + 1e-6


class TestLogarithmBase:
    """Test consistency across logarithm bases."""

    def test_mi_base_conversion(self) -> None:
        """Test MI scales correctly with log base."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()

        mi_nats = mutual_information(joint, base=None)
        mi_bits = mutual_information(joint, base=2.0)

        # bits = nats / log(2)
        import math

        assert torch.allclose(mi_bits, mi_nats / math.log(2), atol=1e-6)

    def test_all_operators_consistent_base(self) -> None:
        """Test all operators use consistent base conversion."""

        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        p_x = joint.sum(dim=1)

        # Test in bits (base 2)
        h_xy_bits = joint_entropy(joint, base=2.0)
        h_x_bits = shannon_entropy(p_x, dim=0, base=2.0)
        h_y_given_x_bits = conditional_entropy(
            joint, condition_dim=0, target_dim=1, base=2.0
        )
        mi_bits = mutual_information(joint, base=2.0)

        # Chain rule should still hold
        assert torch.allclose(
            h_xy_bits, h_x_bits + h_y_given_x_bits, atol=1e-6
        )

        # MI identity should still hold
        h_xy_nats = joint_entropy(joint, base=None)
        h_x_nats = shannon_entropy(p_x, dim=0, base=None)
        p_y = joint.sum(dim=0)
        h_y_nats = shannon_entropy(p_y, dim=0, base=None)
        mi_nats = mutual_information(joint, base=None)

        assert torch.allclose(
            mi_nats, h_x_nats + h_y_nats - h_xy_nats, atol=1e-6
        )
