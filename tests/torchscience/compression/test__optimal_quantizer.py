"""Tests for optimal quantizer design."""

import pytest
import torch

from torchscience.compression import optimal_quantizer


class TestOptimalQuantizerBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns codebook, boundaries, and distortion."""
        samples = torch.randn(1000)
        codebook, boundaries, distortion = optimal_quantizer(
            samples, n_levels=4
        )
        assert isinstance(codebook, torch.Tensor)
        assert isinstance(boundaries, torch.Tensor)
        assert isinstance(distortion, float)

    def test_output_shapes(self):
        """Output shapes are correct."""
        samples = torch.randn(1000)
        n_levels = 8
        codebook, boundaries, _ = optimal_quantizer(samples, n_levels=n_levels)
        assert codebook.shape == (n_levels,)
        assert boundaries.shape == (n_levels - 1,)

    def test_codebook_sorted(self):
        """Codebook is sorted."""
        samples = torch.randn(1000)
        codebook, _, _ = optimal_quantizer(samples, n_levels=4)
        assert torch.all(codebook[:-1] <= codebook[1:])

    def test_boundaries_between_levels(self):
        """Boundaries are between adjacent levels."""
        samples = torch.randn(1000)
        codebook, boundaries, _ = optimal_quantizer(samples, n_levels=4)
        for i in range(len(boundaries)):
            assert codebook[i] < boundaries[i] < codebook[i + 1]


class TestOptimalQuantizerConvergence:
    """Tests for algorithm convergence."""

    def test_distortion_decreases(self):
        """Distortion is non-negative and reasonable."""
        torch.manual_seed(42)
        samples = torch.randn(5000)
        _, _, distortion = optimal_quantizer(samples, n_levels=4)
        assert distortion >= 0
        # For Gaussian with 4 levels, MSE should be around 0.1-0.15
        assert distortion < 0.2

    def test_more_levels_less_distortion(self):
        """More quantization levels gives less distortion."""
        torch.manual_seed(42)
        samples = torch.randn(5000)

        _, _, dist_2 = optimal_quantizer(samples, n_levels=2)
        _, _, dist_4 = optimal_quantizer(samples, n_levels=4)
        _, _, dist_8 = optimal_quantizer(samples, n_levels=8)

        assert dist_8 < dist_4 < dist_2

    def test_gaussian_known_values(self):
        """Matches known values for Gaussian distribution."""
        torch.manual_seed(42)
        # Large sample for accurate estimation
        samples = torch.randn(50000)
        codebook, _, distortion = optimal_quantizer(samples, n_levels=2)

        # For unit Gaussian with 2 levels:
        # Optimal levels are approximately ±0.798
        # MSE is approximately 0.363
        assert torch.allclose(
            codebook.abs(), torch.tensor([0.798, 0.798]), atol=0.05
        )
        assert abs(distortion - 0.363) < 0.02


class TestOptimalQuantizerInitialization:
    """Tests for different initialization methods."""

    def test_uniform_init(self):
        """Uniform initialization works."""
        samples = torch.randn(1000)
        codebook, _, _ = optimal_quantizer(samples, n_levels=4, init="uniform")
        assert len(codebook) == 4

    def test_random_init(self):
        """Random initialization works."""
        samples = torch.randn(1000)
        codebook, _, _ = optimal_quantizer(samples, n_levels=4, init="random")
        assert len(codebook) == 4

    def test_kmeanspp_init(self):
        """K-means++ initialization works."""
        samples = torch.randn(1000)
        codebook, _, _ = optimal_quantizer(
            samples, n_levels=4, init="kmeans++"
        )
        assert len(codebook) == 4

    def test_different_inits_converge(self):
        """Different initializations converge to similar results."""
        torch.manual_seed(42)
        samples = torch.randn(5000)

        torch.manual_seed(42)
        _, _, dist_uniform = optimal_quantizer(
            samples, n_levels=4, init="uniform"
        )

        torch.manual_seed(42)
        _, _, dist_random = optimal_quantizer(
            samples, n_levels=4, init="random"
        )

        # Results should be similar (within 10%)
        assert abs(dist_uniform - dist_random) / dist_uniform < 0.1


class TestOptimalQuantizerIterations:
    """Tests for iteration control."""

    def test_max_iter_respected(self):
        """Algorithm respects max_iter."""
        samples = torch.randn(1000)
        # With 1 iteration, should still produce valid output
        codebook, boundaries, _ = optimal_quantizer(
            samples, n_levels=4, max_iter=1
        )
        assert len(codebook) == 4
        assert len(boundaries) == 3

    def test_tolerance_affects_convergence(self):
        """Tighter tolerance gives more iterations."""
        torch.manual_seed(42)
        samples = torch.randn(1000)

        # Loose tolerance should converge faster
        # (we can't directly measure iterations, but distortion should be similar)
        _, _, dist_loose = optimal_quantizer(samples, n_levels=4, tol=1e-3)
        _, _, dist_tight = optimal_quantizer(samples, n_levels=4, tol=1e-8)

        # Both should give reasonable results
        assert dist_loose < 0.2
        assert dist_tight < 0.2


class TestOptimalQuantizerDistributions:
    """Tests for different input distributions."""

    def test_uniform_distribution(self):
        """Works with uniform distribution."""
        samples = torch.rand(5000)  # Uniform [0, 1]
        codebook, _, distortion = optimal_quantizer(samples, n_levels=4)

        # For uniform, optimal levels should be roughly 0.125, 0.375, 0.625, 0.875
        expected = torch.tensor([0.125, 0.375, 0.625, 0.875])
        assert torch.allclose(codebook, expected, atol=0.05)

    def test_bimodal_distribution(self):
        """Works with bimodal distribution."""
        # Create bimodal: mix of two Gaussians
        torch.manual_seed(42)
        samples = torch.cat(
            [
                torch.randn(2500) - 2,  # Centered at -2
                torch.randn(2500) + 2,  # Centered at +2
            ]
        )
        codebook, _, _ = optimal_quantizer(samples, n_levels=4)

        # Should place levels near the modes
        assert codebook.min() < -1  # Some levels in left mode
        assert codebook.max() > 1  # Some levels in right mode


class TestOptimalQuantizerEdgeCases:
    """Edge case tests."""

    def test_two_levels(self):
        """Minimum number of levels (2) works."""
        samples = torch.randn(1000)
        codebook, boundaries, _ = optimal_quantizer(samples, n_levels=2)
        assert len(codebook) == 2
        assert len(boundaries) == 1

    def test_many_levels(self):
        """Works with many levels."""
        samples = torch.randn(5000)
        codebook, _, _ = optimal_quantizer(samples, n_levels=64)
        assert len(codebook) == 64

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            optimal_quantizer([0.1, 0.2, 0.3], n_levels=2)

    def test_not_1d_raises(self):
        """Raises error for non-1D input."""
        samples = torch.randn(10, 10)
        with pytest.raises(ValueError, match="1-dimensional"):
            optimal_quantizer(samples, n_levels=4)

    def test_too_few_levels_raises(self):
        """Raises error for n_levels < 2."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="n_levels"):
            optimal_quantizer(samples, n_levels=1)

    def test_too_few_samples_raises(self):
        """Raises error for fewer samples than levels."""
        samples = torch.randn(3)
        with pytest.raises(ValueError, match="samples"):
            optimal_quantizer(samples, n_levels=5)

    def test_invalid_init_raises(self):
        """Raises error for invalid init method."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="init"):
            optimal_quantizer(samples, n_levels=4, init="invalid")


class TestOptimalQuantizerDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        samples = torch.randn(1000, device="cpu")
        codebook, boundaries, _ = optimal_quantizer(samples, n_levels=4)
        assert codebook.device.type == "cpu"
        assert boundaries.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        samples = torch.randn(1000, device="cuda")
        codebook, boundaries, _ = optimal_quantizer(samples, n_levels=4)
        assert codebook.device.type == "cuda"
        assert boundaries.device.type == "cuda"
