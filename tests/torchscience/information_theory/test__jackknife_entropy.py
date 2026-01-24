"""Tests for jackknife entropy estimation."""

import math

import pytest
import torch

from torchscience.information import (
    jackknife_entropy,
    kozachenko_leonenko_entropy,
)


class TestJackknifeEntropyBasic:
    """Basic functionality tests."""

    def test_returns_tuple(self):
        """Returns a tuple of (entropy, standard_error)."""
        samples = torch.randn(100, 2)
        result = jackknife_entropy(samples)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shape_2d(self):
        """Returns scalar tensors for (n_samples, n_dims) input."""
        samples = torch.randn(100, 2)
        h, se = jackknife_entropy(samples)
        assert h.shape == torch.Size([])
        assert se.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensors for batched input."""
        samples = torch.randn(3, 100, 2)
        h, se = jackknife_entropy(samples)
        assert h.shape == torch.Size([3])
        assert se.shape == torch.Size([3])

    def test_outputs_finite(self):
        """Both outputs are finite."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h, se = jackknife_entropy(samples)
        assert torch.isfinite(h)
        assert torch.isfinite(se)


class TestJackknifeEntropyCorrectness:
    """Numerical correctness tests."""

    def test_entropy_close_to_direct_estimate(self):
        """Jackknife entropy is close to direct KL estimate."""
        torch.manual_seed(42)
        samples = torch.randn(200, 2)
        h_jk, _ = jackknife_entropy(samples, k=3)
        h_kl = kozachenko_leonenko_entropy(samples, k=3)
        # Jackknife should be close to direct estimate
        assert torch.isclose(h_jk, h_kl, rtol=0.2), (
            f"Jackknife {h_jk} differs too much from direct {h_kl}"
        )

    def test_standard_error_positive(self):
        """Standard error is non-negative."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        _, se = jackknife_entropy(samples)
        assert se >= 0

    def test_standard_error_reasonable(self):
        """Standard error is reasonably sized."""
        torch.manual_seed(42)
        samples = torch.randn(200, 2)
        h, se = jackknife_entropy(samples)
        # SE should be much smaller than the estimate itself
        assert se < torch.abs(h), f"SE {se} is larger than |H| {torch.abs(h)}"


class TestJackknifeEntropyEstimator:
    """Tests for different estimator choices."""

    def test_kozachenko_leonenko_default(self):
        """Default uses kozachenko_leonenko."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h_default, _ = jackknife_entropy(samples, k=1)
        h_kl, _ = jackknife_entropy(
            samples, estimator="kozachenko_leonenko", k=1
        )
        assert torch.equal(h_default, h_kl)

    def test_kraskov_estimator(self):
        """Kraskov estimator works."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h, se = jackknife_entropy(samples, estimator="kraskov", k=3)
        assert torch.isfinite(h)
        assert torch.isfinite(se)

    def test_invalid_estimator_raises(self):
        """Invalid estimator raises error."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="estimator must be"):
            jackknife_entropy(samples, estimator="invalid")


class TestJackknifeEntropyKValues:
    """Tests for different k values."""

    def test_k_1(self):
        """Works with k=1."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h, se = jackknife_entropy(samples, k=1)
        assert torch.isfinite(h)
        assert torch.isfinite(se)

    def test_k_3(self):
        """Works with k=3."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h, se = jackknife_entropy(samples, k=3)
        assert torch.isfinite(h)
        assert torch.isfinite(se)

    def test_k_too_large_raises(self):
        """Raises error when k >= n_samples - 1."""
        samples = torch.randn(20, 2)
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            jackknife_entropy(samples, k=19)


class TestJackknifeEntropyBase:
    """Tests for logarithm base conversion."""

    def test_base_2_bits(self):
        """Base 2 gives entropy in bits."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2)
        h_nats, se_nats = jackknife_entropy(samples)
        h_bits, se_bits = jackknife_entropy(samples, base=2)
        expected_h = h_nats / math.log(2)
        expected_se = se_nats / math.log(2)
        assert torch.isclose(h_bits, expected_h, rtol=1e-5)
        assert torch.isclose(se_bits, expected_se, rtol=1e-5)


class TestJackknifeEntropyBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        torch.manual_seed(42)
        s1 = torch.randn(100, 2)
        s2 = torch.randn(100, 2)
        s_batch = torch.stack([s1, s2])

        h1, se1 = jackknife_entropy(s1)
        h2, se2 = jackknife_entropy(s2)
        h_batch, se_batch = jackknife_entropy(s_batch)

        assert torch.isclose(h_batch[0], h1, rtol=1e-5)
        assert torch.isclose(h_batch[1], h2, rtol=1e-5)
        assert torch.isclose(se_batch[0], se1, rtol=1e-5)
        assert torch.isclose(se_batch[1], se2, rtol=1e-5)


class TestJackknifeEntropyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_1d_input_raises(self):
        """Raises error for 1D input."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            jackknife_entropy(samples)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            jackknife_entropy([[1, 2], [3, 4]])


class TestJackknifeEntropyDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        samples = torch.randn(100, 2, dtype=torch.float32)
        h, se = jackknife_entropy(samples)
        assert h.dtype == torch.float32
        assert se.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        samples = torch.randn(100, 2, dtype=torch.float64)
        h, se = jackknife_entropy(samples)
        assert h.dtype == torch.float64
        assert se.dtype == torch.float64


class TestJackknifeEntropyDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        samples = torch.randn(100, 2, device="cpu")
        h, se = jackknife_entropy(samples)
        assert h.device.type == "cpu"
        assert se.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        samples = torch.randn(100, 2, device="cuda")
        h, se = jackknife_entropy(samples)
        assert h.device.type == "cuda"
        assert se.device.type == "cuda"


class TestJackknifeEntropyReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        samples = torch.randn(100, 2)
        h1, se1 = jackknife_entropy(samples)
        h2, se2 = jackknife_entropy(samples)
        assert torch.equal(h1, h2)
        assert torch.equal(se1, se2)
