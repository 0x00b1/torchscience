"""Tests for Kraskov-Stögbauer-Grassberger mutual information estimator."""

import math

import pytest
import torch

from torchscience.information import kraskov_mutual_information


class TestKraskovMIBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for (n_samples, n_dims) inputs."""
        x = torch.randn(1000, 2)
        y = torch.randn(1000, 3)
        result = kraskov_mutual_information(x, y)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched input."""
        x = torch.randn(5, 1000, 2)
        y = torch.randn(5, 1000, 3)
        result = kraskov_mutual_information(x, y)
        assert result.shape == torch.Size([5])

    def test_output_is_finite(self):
        """Result is finite for normal inputs."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        result = kraskov_mutual_information(x, y)
        assert torch.isfinite(result)


class TestKraskovMICorrectness:
    """Numerical correctness tests."""

    def test_independent_variables_zero_mi(self):
        """Independent variables have MI close to 0."""
        torch.manual_seed(42)
        n = 2000
        x = torch.randn(n, 1)
        y = torch.randn(n, 1)
        mi = kraskov_mutual_information(x, y)
        # MI should be close to 0 for independent variables
        assert torch.abs(mi) < 0.1, (
            f"Expected MI ~ 0 for independent, got {mi}"
        )

    def test_correlated_variables_positive_mi(self):
        """Correlated variables have positive MI."""
        torch.manual_seed(42)
        n = 2000
        x = torch.randn(n, 1)
        noise = torch.randn(n, 1)
        y = 0.9 * x + 0.1 * noise  # Highly correlated
        mi = kraskov_mutual_information(x, y)
        assert mi > 0.5, (
            f"Expected significant MI for correlated vars, got {mi}"
        )

    def test_perfect_correlation(self):
        """Perfectly correlated variables have high MI."""
        torch.manual_seed(42)
        n = 1000
        x = torch.randn(n, 1)
        y = x.clone()  # Perfect correlation (y = x)
        mi = kraskov_mutual_information(x, y)
        # MI should be high (theoretically infinite for deterministic relationship)
        assert mi > 3.0, f"Expected high MI for y=x, got {mi}"

    def test_mi_non_negative(self):
        """MI is always non-negative."""
        torch.manual_seed(42)
        for _ in range(5):
            n = 500
            x = torch.randn(n, 2)
            y = torch.randn(n, 3)
            mi = kraskov_mutual_information(x, y)
            # KSG can give slightly negative estimates due to bias
            assert mi > -0.2, f"MI should be roughly non-negative, got {mi}"

    def test_bivariate_gaussian_mi(self):
        """MI of bivariate Gaussian matches theory."""
        torch.manual_seed(42)
        n = 5000
        rho = 0.8  # Correlation coefficient
        # Generate bivariate Gaussian
        x = torch.randn(n, 1)
        y = rho * x + math.sqrt(1 - rho**2) * torch.randn(n, 1)
        mi = kraskov_mutual_information(x, y)
        # Theoretical MI for bivariate Gaussian: I = -0.5 * log(1 - rho^2)
        expected = -0.5 * math.log(1 - rho**2)
        assert torch.isclose(mi, torch.tensor(expected), rtol=0.3), (
            f"Expected MI ~ {expected}, got {mi}"
        )


class TestKraskovMIAlgorithms:
    """Tests for different algorithm choices."""

    def test_algorithm_1_default(self):
        """Algorithm 1 is the default."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        mi_default = kraskov_mutual_information(x, y)
        mi_alg1 = kraskov_mutual_information(x, y, algorithm=1)
        assert torch.equal(mi_default, mi_alg1)

    def test_algorithm_2(self):
        """Algorithm 2 works and gives finite results."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        mi_alg2 = kraskov_mutual_information(x, y, algorithm=2)
        # Algorithm 2 should produce finite results
        assert torch.isfinite(mi_alg2)

    def test_invalid_algorithm_raises(self):
        """Invalid algorithm raises error."""
        x = torch.randn(100, 2)
        y = torch.randn(100, 2)
        with pytest.raises(ValueError, match="algorithm must be 1 or 2"):
            kraskov_mutual_information(x, y, algorithm=3)


class TestKraskovMIKValues:
    """Tests for different k values."""

    def test_k_1(self):
        """k=1 works."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        mi = kraskov_mutual_information(x, y, k=1)
        assert torch.isfinite(mi)

    def test_k_5(self):
        """k=5 works."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        mi = kraskov_mutual_information(x, y, k=5)
        assert torch.isfinite(mi)

    def test_different_k_similar(self):
        """Different k values give similar results for large samples."""
        torch.manual_seed(42)
        n = 2000
        x = torch.randn(n, 1)
        y = 0.8 * x + 0.2 * torch.randn(n, 1)
        mi_k1 = kraskov_mutual_information(x, y, k=1)
        mi_k3 = kraskov_mutual_information(x, y, k=3)
        mi_k5 = kraskov_mutual_information(x, y, k=5)
        # All should be within 30% of each other
        assert torch.isclose(mi_k1, mi_k3, rtol=0.3)
        assert torch.isclose(mi_k3, mi_k5, rtol=0.3)


class TestKraskovMIBase:
    """Tests for logarithm base conversion."""

    def test_base_2_bits(self):
        """Base 2 gives MI in bits."""
        torch.manual_seed(42)
        x = torch.randn(500, 2)
        y = 0.8 * x + 0.2 * torch.randn(500, 2)
        mi_nats = kraskov_mutual_information(x, y)
        mi_bits = kraskov_mutual_information(x, y, base=2)
        expected_bits = mi_nats / math.log(2)
        assert torch.isclose(mi_bits, expected_bits, rtol=1e-5)


class TestKraskovMIBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        torch.manual_seed(42)
        x1 = torch.randn(500, 2)
        y1 = torch.randn(500, 3)
        x2 = torch.randn(500, 2)
        y2 = torch.randn(500, 3)

        x_batch = torch.stack([x1, x2])
        y_batch = torch.stack([y1, y2])

        mi1 = kraskov_mutual_information(x1, y1)
        mi2 = kraskov_mutual_information(x2, y2)
        mi_batch = kraskov_mutual_information(x_batch, y_batch)

        assert torch.isclose(mi_batch[0], mi1, rtol=1e-5)
        assert torch.isclose(mi_batch[1], mi2, rtol=1e-5)


class TestKraskovMIEdgeCases:
    """Tests for edge cases and error handling."""

    def test_k_too_large_raises(self):
        """Raises error when k >= n_samples."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            kraskov_mutual_information(x, y, k=10)

    def test_mismatched_n_samples_raises(self):
        """Raises error when n_samples differ."""
        x = torch.randn(100, 2)
        y = torch.randn(50, 2)
        with pytest.raises(ValueError, match="matching batch dimensions"):
            kraskov_mutual_information(x, y)

    def test_mismatched_batch_dims_raises(self):
        """Raises error when batch dimensions differ."""
        x = torch.randn(5, 100, 2)
        y = torch.randn(3, 100, 2)
        with pytest.raises(ValueError, match="matching batch dimensions"):
            kraskov_mutual_information(x, y)

    def test_1d_input_raises(self):
        """Raises error for 1D input."""
        x = torch.randn(100)
        y = torch.randn(100)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            kraskov_mutual_information(x, y)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kraskov_mutual_information([[1, 2]], [[3, 4]])

    def test_different_dims_allowed(self):
        """X and Y can have different dimensionality."""
        x = torch.randn(500, 3)
        y = torch.randn(500, 5)
        mi = kraskov_mutual_information(x, y)
        assert torch.isfinite(mi)


class TestKraskovMIDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        x = torch.randn(500, 2, dtype=torch.float32)
        y = torch.randn(500, 2, dtype=torch.float32)
        result = kraskov_mutual_information(x, y)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        x = torch.randn(500, 2, dtype=torch.float64)
        y = torch.randn(500, 2, dtype=torch.float64)
        result = kraskov_mutual_information(x, y)
        assert result.dtype == torch.float64


class TestKraskovMIDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(500, 2, device="cpu")
        y = torch.randn(500, 2, device="cpu")
        result = kraskov_mutual_information(x, y)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(500, 2, device="cuda")
        y = torch.randn(500, 2, device="cuda")
        result = kraskov_mutual_information(x, y)
        assert result.device.type == "cuda"


class TestKraskovMIGradients:
    """Tests for gradient computation."""

    @pytest.mark.skip(
        reason="KSG MI involves boolean counting operations that are non-differentiable"
    )
    def test_gradients_exist(self):
        """Gradients flow through the computation.

        Note: KSG MI involves counting points within epsilon distance,
        which uses boolean comparisons and is not differentiable in the
        traditional sense. This test is skipped.
        """
        torch.manual_seed(42)
        x = torch.randn(100, 2, requires_grad=True)
        y = torch.randn(100, 2, requires_grad=True)
        result = kraskov_mutual_information(x, y)
        result.backward()
        assert x.grad is not None
        assert y.grad is not None


class TestKraskovMIReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        x = torch.randn(500, 2)
        y = torch.randn(500, 3)
        mi1 = kraskov_mutual_information(x, y)
        mi2 = kraskov_mutual_information(x, y)
        assert torch.equal(mi1, mi2)
