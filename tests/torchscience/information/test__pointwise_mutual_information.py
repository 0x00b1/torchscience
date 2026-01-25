"""Tests for pointwise_mutual_information operator."""

import math

import pytest
import torch

from torchscience.information import pointwise_mutual_information


class TestPointwiseMutualInformationForward:
    """Test forward computation of PMI."""

    def test_independent_distribution(self) -> None:
        """Test that PMI is zero for independent distributions."""
        p_x = torch.tensor([0.3, 0.7])
        p_y = torch.tensor([0.4, 0.6])
        joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        pmi = pointwise_mutual_information(joint)

        # For independent distribution, PMI should be zero everywhere
        assert pmi.shape == joint.shape
        assert torch.allclose(pmi, torch.zeros_like(pmi), atol=1e-5)

    def test_positive_correlation(self) -> None:
        """Test PMI for positively correlated events."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]])

        pmi = pointwise_mutual_information(joint)

        # Diagonal elements should be positive (co-occur more than expected)
        assert pmi[0, 0] > 0
        assert pmi[1, 1] > 0
        # Off-diagonal elements should be negative (co-occur less than expected)
        assert pmi[0, 1] < 0
        assert pmi[1, 0] < 0

    def test_manual_computation(self) -> None:
        """Test PMI against manual computation."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_x = joint.sum(dim=1)  # [0.5, 0.5]
        p_y = joint.sum(dim=0)  # [0.5, 0.5]

        pmi = pointwise_mutual_information(joint)

        # Compute expected PMI manually
        for i in range(2):
            for j in range(2):
                expected = math.log(
                    joint[i, j].item() / (p_x[i].item() * p_y[j].item())
                )
                assert abs(pmi[i, j].item() - expected) < 1e-5

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input shape."""
        joint = torch.rand(3, 4)
        joint = joint / joint.sum()

        pmi = pointwise_mutual_information(joint)

        assert pmi.shape == joint.shape

    def test_log_probability_input(self) -> None:
        """Test PMI with log-probability input."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        log_joint = joint.log()

        pmi_prob = pointwise_mutual_information(
            joint, input_type="probability"
        )
        pmi_log = pointwise_mutual_information(
            log_joint, input_type="log_probability"
        )

        assert torch.allclose(pmi_prob, pmi_log, atol=1e-5)

    def test_custom_base(self) -> None:
        """Test PMI with different logarithm bases."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)

        pmi_nats = pointwise_mutual_information(joint, base=None)
        pmi_bits = pointwise_mutual_information(joint, base=2.0)

        # Convert nats to bits: bits = nats / log(2)
        assert torch.allclose(pmi_bits, pmi_nats / math.log(2), atol=1e-5)

    def test_custom_dims(self) -> None:
        """Test PMI with custom dimension specification."""
        joint = torch.rand(3, 4)
        joint = joint / joint.sum()

        pmi_default = pointwise_mutual_information(joint, dims=(-2, -1))
        pmi_explicit = pointwise_mutual_information(joint, dims=(0, 1))

        assert torch.allclose(pmi_default, pmi_explicit)


class TestPointwiseMutualInformationBatched:
    """Test batched PMI computation."""

    def test_batched_2d(self) -> None:
        """Test PMI with batch dimensions."""
        batch = 3
        size_x, size_y = 4, 5
        joint = torch.rand(batch, size_x, size_y)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)

        pmi = pointwise_mutual_information(joint, dims=(-2, -1))

        assert pmi.shape == (batch, size_x, size_y)

    def test_batched_independent_distribution(self) -> None:
        """Test batched PMI for independent distributions."""
        batch = 3
        size_x, size_y = 4, 5

        # Create independent distributions for each batch
        joint = torch.zeros(batch, size_x, size_y)
        for b in range(batch):
            p_x = torch.rand(size_x)
            p_x = p_x / p_x.sum()
            p_y = torch.rand(size_y)
            p_y = p_y / p_y.sum()
            joint[b] = p_x.unsqueeze(1) * p_y.unsqueeze(0)

        pmi = pointwise_mutual_information(joint, dims=(-2, -1))

        # PMI should be approximately zero for all batches
        assert torch.allclose(pmi, torch.zeros_like(pmi), atol=1e-5)


class TestPointwiseMutualInformationGradients:
    """Test gradient computation for PMI."""

    def test_gradcheck(self) -> None:
        """Test first-order gradients with gradcheck."""
        joint = torch.rand(3, 4, dtype=torch.float64, requires_grad=True)
        joint = joint / joint.detach().sum()

        def pmi_func(j):
            # Normalize to get valid probability distribution
            j_normalized = j / j.sum()
            return pointwise_mutual_information(j_normalized)

        assert torch.autograd.gradcheck(
            pmi_func, (joint,), raise_exception=True
        )

    def test_gradgradcheck(self) -> None:
        """Test second-order gradients with gradgradcheck."""
        joint = torch.rand(3, 4, dtype=torch.float64, requires_grad=True)
        joint = joint / joint.detach().sum()

        def pmi_func(j):
            j_normalized = j / j.sum()
            return pointwise_mutual_information(j_normalized)

        assert torch.autograd.gradgradcheck(
            pmi_func, (joint,), raise_exception=True
        )

    def test_gradient_computation_batched(self) -> None:
        """Test gradient computation for batched input."""
        joint = torch.rand(2, 3, 4, dtype=torch.float64, requires_grad=True)
        joint = joint / joint.detach().sum(dim=(-2, -1), keepdim=True)

        def pmi_func(j):
            j_normalized = j / j.sum(dim=(-2, -1), keepdim=True)
            return pointwise_mutual_information(j_normalized).sum()

        assert torch.autograd.gradcheck(
            pmi_func, (joint,), raise_exception=True
        )


class TestPointwiseMutualInformationMeta:
    """Test meta tensor support for PMI."""

    def test_meta_tensor_shape(self) -> None:
        """Test that meta tensors produce correct output shape."""
        joint = torch.rand(3, 4, device="meta")

        pmi = pointwise_mutual_information(joint)

        assert pmi.shape == joint.shape
        assert pmi.device.type == "meta"

    def test_meta_tensor_batched(self) -> None:
        """Test meta tensors with batch dimensions."""
        joint = torch.rand(2, 3, 4, device="meta")

        pmi = pointwise_mutual_information(joint, dims=(-2, -1))

        assert pmi.shape == (2, 3, 4)
        assert pmi.device.type == "meta"


class TestPointwiseMutualInformationDtypes:
    """Test different data types for PMI."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype: torch.dtype) -> None:
        """Test PMI with different floating point types."""
        joint = torch.rand(3, 4, dtype=dtype)
        joint = joint / joint.sum()

        pmi = pointwise_mutual_information(joint)

        assert pmi.dtype == dtype

    def test_preserves_dtype(self) -> None:
        """Test that output dtype matches input dtype."""
        joint32 = torch.rand(3, 4, dtype=torch.float32)
        joint32 = joint32 / joint32.sum()
        joint64 = joint32.to(torch.float64)

        pmi32 = pointwise_mutual_information(joint32)
        pmi64 = pointwise_mutual_information(joint64)

        assert pmi32.dtype == torch.float32
        assert pmi64.dtype == torch.float64


class TestPointwiseMutualInformationEdgeCases:
    """Test edge cases for PMI."""

    def test_uniform_distribution(self) -> None:
        """Test PMI for uniform distribution (independent by definition)."""
        joint = torch.ones(3, 4) / 12.0

        pmi = pointwise_mutual_information(joint)

        # Uniform distribution means independence, so PMI should be zero
        assert torch.allclose(pmi, torch.zeros_like(pmi), atol=1e-5)

    def test_deterministic_relationship(self) -> None:
        """Test PMI for deterministic X=Y relationship."""
        # X=Y with 50% probability each
        joint = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

        pmi = pointwise_mutual_information(joint)

        # Diagonal: PMI = log(0.5/(0.5*0.5)) = log(2) ≈ 0.693
        expected_diag = math.log(2)
        assert abs(pmi[0, 0].item() - expected_diag) < 1e-5
        assert abs(pmi[1, 1].item() - expected_diag) < 1e-5
        # Off-diagonal should be very negative (clamped)
        assert pmi[0, 1] < -10
        assert pmi[1, 0] < -10

    def test_near_zero_probabilities(self) -> None:
        """Test PMI handles near-zero probabilities."""
        joint = torch.tensor([[0.45, 0.05], [0.05, 0.45]])

        pmi = pointwise_mutual_information(joint)

        # Should not have NaN or Inf
        assert not torch.isnan(pmi).any()
        assert not torch.isinf(pmi).any()


class TestPointwiseMutualInformationSymmetry:
    """Test symmetry properties of PMI."""

    def test_symmetric_joint_gives_symmetric_pmi(self) -> None:
        """Test that symmetric joint distribution gives symmetric PMI."""
        joint = torch.tensor(
            [[0.3, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.0]]
        )
        joint = joint + joint.T  # Make symmetric
        joint = joint / joint.sum()

        pmi = pointwise_mutual_information(joint)

        assert torch.allclose(pmi, pmi.T, atol=1e-6)


class TestPointwiseMutualInformationRelationships:
    """Test relationships between PMI and other information measures."""

    def test_expected_pmi_equals_mi(self) -> None:
        """Test that E[PMI] = MI (mutual information is expected PMI)."""
        from torchscience.information import mutual_information

        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)

        pmi = pointwise_mutual_information(joint)
        mi = mutual_information(joint)

        # MI = sum p(x,y) * PMI(x,y)
        expected_mi = (joint * pmi).sum()

        assert torch.allclose(mi, expected_mi, atol=1e-6)

    def test_pmi_relationship_with_entropies(self) -> None:
        """Test PMI relationship: PMI(x,y) = log p(x,y) - log p(x) - log p(y)."""
        joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float64)
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)

        pmi = pointwise_mutual_information(joint)

        # Compute PMI using entropy decomposition
        for i in range(2):
            for j in range(2):
                expected = (
                    math.log(joint[i, j].item())
                    - math.log(p_x[i].item())
                    - math.log(p_y[j].item())
                )
                assert abs(pmi[i, j].item() - expected) < 1e-5
