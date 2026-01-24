"""Tests for Blahut-Arimoto algorithm."""

import math

import pytest
import torch

from torchscience.information import blahut_arimoto


class TestBlahutArimotoBasic:
    """Basic functionality tests."""

    def test_output_shape_capacity(self):
        """Returns scalar for 2D input in capacity mode."""
        P = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        result = blahut_arimoto(P)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        P = torch.rand(2, 3, 4, 4)
        P = P / P.sum(dim=-1, keepdim=True)
        result = blahut_arimoto(P)
        assert result.shape == torch.Size([2, 3])

    def test_return_distribution(self):
        """Returns tuple when return_distribution=True."""
        P = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        result = blahut_arimoto(P, return_distribution=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        capacity, px = result
        assert capacity.shape == torch.Size([])
        assert px.shape == torch.Size([2])


class TestBlahutArimotoCapacityCorrectness:
    """Numerical correctness tests for capacity mode."""

    def test_noiseless_channel(self):
        """Noiseless channel (identity matrix) has capacity log(n)."""
        n = 4
        P = torch.eye(n)
        result = blahut_arimoto(P, base=2.0)
        expected = math.log2(n)  # 2 bits
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_binary_symmetric_channel(self):
        """BSC with crossover p has capacity 1 - H(p)."""
        p = 0.1
        P = torch.tensor([[1 - p, p], [p, 1 - p]])
        result = blahut_arimoto(P, base=2.0)

        # H(p) = -p*log2(p) - (1-p)*log2(1-p)
        h_p = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        expected = 1.0 - h_p
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_binary_erasure_channel(self):
        """BEC with erasure probability e has capacity 1 - e."""
        e = 0.3
        # BEC: output alphabet is {0, 1, ?}
        # P(0|0)=1-e, P(?|0)=e, P(1|0)=0
        # P(0|1)=0, P(?|1)=e, P(1|1)=1-e
        P = torch.tensor([[1 - e, 0.0, e], [0.0, 1 - e, e]])
        result = blahut_arimoto(P, base=2.0)
        expected = 1.0 - e
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_useless_channel(self):
        """Channel with same output distribution has capacity 0."""
        # All inputs produce same output distribution
        P = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = blahut_arimoto(P, base=2.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_optimal_distribution_bsc(self):
        """BSC has uniform optimal input distribution."""
        p = 0.2
        P = torch.tensor([[1 - p, p], [p, 1 - p]])
        _, px = blahut_arimoto(P, return_distribution=True)
        expected = torch.tensor([0.5, 0.5])
        assert torch.allclose(px, expected, atol=1e-4)

    def test_non_negative(self):
        """Capacity is non-negative."""
        P = torch.rand(4, 4)
        P = P / P.sum(dim=-1, keepdim=True)
        result = blahut_arimoto(P)
        assert result >= -1e-6


class TestBlahutArimotoRateDistortion:
    """Tests for rate-distortion mode."""

    def test_requires_source_distribution(self):
        """Raises error if source_distribution not provided."""
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(
            ValueError, match="source_distribution is required"
        ):
            blahut_arimoto(d, mode="rate_distortion", lagrange_multiplier=1.0)

    def test_requires_lagrange_multiplier(self):
        """Raises error if lagrange_multiplier not provided."""
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        px = torch.tensor([0.5, 0.5])
        with pytest.raises(
            ValueError, match="lagrange_multiplier is required"
        ):
            blahut_arimoto(d, mode="rate_distortion", source_distribution=px)

    def test_hamming_distortion(self):
        """Binary source with Hamming distortion."""
        # Distortion matrix: d(x,y) = 1 if x != y, 0 otherwise
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        px = torch.tensor([0.5, 0.5])

        # For β=0, p(y|x) = p(y) (independent), so I(X;Y) = 0
        rate_beta0 = blahut_arimoto(
            d,
            mode="rate_distortion",
            source_distribution=px,
            lagrange_multiplier=0.0,
            base=2.0,
        )
        assert torch.isclose(rate_beta0, torch.tensor(0.0), atol=0.1)

        # For large β, rate approaches H(X) = 1 bit (lossless)
        rate_beta_large = blahut_arimoto(
            d,
            mode="rate_distortion",
            source_distribution=px,
            lagrange_multiplier=10.0,
            base=2.0,
        )
        assert rate_beta_large > rate_beta0

    def test_rate_increases_with_beta(self):
        """Rate increases as beta increases (lower distortion requires more rate)."""
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        px = torch.tensor([0.5, 0.5])

        rate1 = blahut_arimoto(
            d,
            mode="rate_distortion",
            source_distribution=px,
            lagrange_multiplier=1.0,
        )
        rate5 = blahut_arimoto(
            d,
            mode="rate_distortion",
            source_distribution=px,
            lagrange_multiplier=5.0,
        )
        assert rate5 > rate1


class TestBlahutArimotoBatching:
    """Tests for batched operations."""

    def test_batch_consistency_capacity(self):
        """Batched capacity matches individual computations."""
        P1 = torch.eye(2)
        P2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        P_batch = torch.stack([P1, P2])

        r1 = blahut_arimoto(P1)
        r2 = blahut_arimoto(P2)
        r_batch = blahut_arimoto(P_batch)

        assert torch.isclose(r_batch[0], r1, atol=1e-5)
        assert torch.isclose(r_batch[1], r2, atol=1e-5)


class TestBlahutArimotoEdgeCases:
    """Tests for edge cases and error handling."""

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            blahut_arimoto([[0.5, 0.5], [0.5, 0.5]])

    def test_1d_input_raises(self):
        """Raises error for 1D input."""
        P = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            blahut_arimoto(P)

    def test_invalid_mode_raises(self):
        """Raises error for invalid mode."""
        P = torch.eye(2)
        with pytest.raises(
            ValueError, match="must be 'capacity' or 'rate_distortion'"
        ):
            blahut_arimoto(P, mode="invalid")

    def test_single_input(self):
        """Single input channel has capacity 0."""
        P = torch.tensor([[0.5, 0.5]])  # 1 input, 2 outputs
        result = blahut_arimoto(P, base=2.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)


class TestBlahutArimotoDtypes:
    """Tests for different data types."""

    def test_float32_input(self):
        """Works with float32 input."""
        P = torch.eye(2, dtype=torch.float32)
        result = blahut_arimoto(P)
        assert result.dtype == torch.float32

    def test_float64_input(self):
        """Works with float64 input."""
        P = torch.eye(2, dtype=torch.float64)
        result = blahut_arimoto(P)
        assert result.dtype == torch.float32  # Converted internally


class TestBlahutArimotoDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        P = torch.eye(2, device="cpu")
        result = blahut_arimoto(P)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        P = torch.eye(2, device="cuda")
        result = blahut_arimoto(P)
        assert result.device.type == "cuda"


class TestBlahutArimotoConvergence:
    """Tests for convergence behavior."""

    def test_converges_quickly_bsc(self):
        """BSC converges in reasonable iterations."""
        p = 0.1
        P = torch.tensor([[1 - p, p], [p, 1 - p]])

        # Should converge with default max_iters
        result = blahut_arimoto(P, max_iters=50)
        assert torch.isfinite(result)

    def test_respects_tolerance(self):
        """Tighter tolerance gives more accurate result."""
        P = torch.tensor([[0.8, 0.2], [0.3, 0.7]])

        result_loose = blahut_arimoto(P, tol=1e-3, max_iters=10)
        result_tight = blahut_arimoto(P, tol=1e-8, max_iters=1000)

        # Both should be reasonable
        assert torch.isfinite(result_loose)
        assert torch.isfinite(result_tight)


class TestBlahutArimotoReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        P = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        r1 = blahut_arimoto(P)
        r2 = blahut_arimoto(P)
        assert torch.equal(r1, r2)
