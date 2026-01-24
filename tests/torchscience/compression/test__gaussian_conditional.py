"""Tests for Gaussian conditional entropy model."""

import pytest
import torch

from torchscience.compression._gaussian_conditional import (
    GaussianConditional,
    gaussian_conditional,
)


class TestGaussianConditionalClass:
    """Tests for GaussianConditional class."""

    def test_initialization(self):
        """GaussianConditional initializes correctly."""
        gc = GaussianConditional()
        assert gc.scale_bound == 0.11

    def test_forward_output_types(self):
        """Forward returns tensor tuple."""
        gc = GaussianConditional()
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        y_hat, likelihoods = gc(y, scales)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(likelihoods, torch.Tensor)

    def test_forward_output_shapes(self):
        """Forward preserves shape."""
        gc = GaussianConditional()
        y = torch.randn(4, 64, 16, 16)
        scales = torch.abs(torch.randn(4, 64, 16, 16)) + 0.5
        y_hat, likelihoods = gc(y, scales)
        assert y_hat.shape == y.shape
        assert likelihoods.shape == y.shape

    def test_forward_with_means(self):
        """Forward works with mean parameters."""
        gc = GaussianConditional()
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        means = torch.randn(2, 32, 8, 8)
        y_hat, likelihoods = gc(y, scales, means)
        assert y_hat.shape == y.shape

    def test_training_mode_adds_noise(self):
        """Training mode adds noise."""
        gc = GaussianConditional()
        gc.train()
        y = torch.randn(2, 16, 4, 4)
        scales = torch.ones(2, 16, 4, 4)

        torch.manual_seed(42)
        y_hat1, _ = gc(y, scales)
        torch.manual_seed(43)
        y_hat2, _ = gc(y, scales)

        assert not torch.allclose(y_hat1, y_hat2)

    def test_eval_mode_quantizes(self):
        """Eval mode performs hard quantization."""
        gc = GaussianConditional()
        gc.eval()
        y = torch.randn(2, 16, 4, 4) * 5
        scales = torch.ones(2, 16, 4, 4)

        y_hat, _ = gc(y, scales)

        # Without means, output should be integers
        assert torch.allclose(y_hat, y_hat.round())

    def test_eval_with_means_shifts_output(self):
        """Eval mode with means shifts the quantized output."""
        gc = GaussianConditional()
        gc.eval()
        y = torch.randn(2, 16, 4, 4) * 5
        scales = torch.ones(2, 16, 4, 4)
        means = torch.randn(2, 16, 4, 4)

        y_hat, _ = gc(y, scales, means)

        # Output minus means should be integers
        assert torch.allclose(y_hat - means, (y_hat - means).round())

    def test_likelihoods_positive(self):
        """Likelihoods are positive."""
        gc = GaussianConditional()
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        _, likelihoods = gc(y, scales)
        assert (likelihoods > 0).all()

    def test_likelihoods_at_most_one(self):
        """Likelihoods are at most 1."""
        gc = GaussianConditional()
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        _, likelihoods = gc(y, scales)
        assert (likelihoods <= 1).all()

    def test_scale_affects_likelihood(self):
        """Scale parameter affects likelihood distribution."""
        gc = GaussianConditional()
        gc.eval()

        # For values near quantization center, smaller scale = higher likelihood
        y_near = torch.zeros(1, 1, 1, 1)  # At quantization center
        scales_small = torch.ones(1, 1, 1, 1) * 0.5
        scales_large = torch.ones(1, 1, 1, 1) * 2.0

        _, likelihood_small = gc(y_near, scales_small)
        _, likelihood_large = gc(y_near, scales_large)

        # Smaller scale concentrates probability near center
        assert likelihood_small > likelihood_large

    def test_compress_decompress_roundtrip(self):
        """Compress and decompress are inverses."""
        gc = GaussianConditional()
        y = torch.randn(2, 16, 4, 4)
        scales = torch.ones(2, 16, 4, 4)

        y_hat, symbols = gc.compress(y, scales)
        y_recovered = gc.decompress(symbols)

        assert torch.allclose(y_hat, y_recovered)

    def test_compress_decompress_with_means(self):
        """Compress/decompress work with means."""
        gc = GaussianConditional()
        y = torch.randn(2, 16, 4, 4)
        scales = torch.ones(2, 16, 4, 4)
        means = torch.randn(2, 16, 4, 4)

        y_hat, symbols = gc.compress(y, scales, means)
        y_recovered = gc.decompress(symbols, means)

        assert torch.allclose(y_hat, y_recovered)

    def test_gradients_flow_through(self):
        """Gradients flow through in training mode."""
        gc = GaussianConditional()
        gc.train()
        y = torch.randn(2, 16, 4, 4, requires_grad=True)
        scales = torch.abs(torch.randn(2, 16, 4, 4)) + 0.5

        y_hat, likelihoods = gc(y, scales)
        loss = y_hat.sum() - likelihoods.log().sum()
        loss.backward()

        assert y.grad is not None


class TestGaussianConditionalFunction:
    """Tests for functional gaussian_conditional."""

    def test_output_types(self):
        """Returns tensor tuple."""
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        y_hat, likelihoods = gaussian_conditional(y, scales)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(likelihoods, torch.Tensor)

    def test_output_shapes(self):
        """Preserves input shape."""
        y = torch.randn(4, 64, 16, 16)
        scales = torch.abs(torch.randn(4, 64, 16, 16)) + 0.5
        y_hat, likelihoods = gaussian_conditional(y, scales)
        assert y_hat.shape == y.shape

    def test_with_means(self):
        """Works with mean parameters."""
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        means = torch.randn(2, 32, 8, 8)
        y_hat, likelihoods = gaussian_conditional(y, scales, means)
        assert y_hat.shape == y.shape

    def test_training_adds_noise(self):
        """Training mode adds noise."""
        y = torch.randn(2, 16, 4, 4)
        scales = torch.ones(2, 16, 4, 4)

        torch.manual_seed(42)
        y_hat1, _ = gaussian_conditional(y, scales, training=True)
        torch.manual_seed(43)
        y_hat2, _ = gaussian_conditional(y, scales, training=True)

        assert not torch.allclose(y_hat1, y_hat2)

    def test_eval_quantizes(self):
        """Eval mode quantizes."""
        y = torch.randn(2, 16, 4, 4) * 5
        scales = torch.ones(2, 16, 4, 4)

        y_hat, _ = gaussian_conditional(y, scales, training=False)
        assert torch.allclose(y_hat, y_hat.round())

    def test_likelihoods_positive(self):
        """Likelihoods are positive."""
        y = torch.randn(2, 32, 8, 8)
        scales = torch.abs(torch.randn(2, 32, 8, 8)) + 0.5
        _, likelihoods = gaussian_conditional(y, scales)
        assert (likelihoods > 0).all()

    def test_not_tensor_y_raises(self):
        """Raises error for non-tensor y."""
        scales = torch.randn(10, 32)
        with pytest.raises(TypeError, match="y must be a Tensor"):
            gaussian_conditional([1.0, 2.0], scales)

    def test_not_tensor_scales_raises(self):
        """Raises error for non-tensor scales."""
        y = torch.randn(10, 32)
        with pytest.raises(TypeError, match="scales must be a Tensor"):
            gaussian_conditional(y, [1.0] * 320)

    def test_shape_mismatch_raises(self):
        """Raises error for shape mismatch."""
        y = torch.randn(10, 32)
        scales = torch.randn(10, 64)
        with pytest.raises(ValueError, match="must match"):
            gaussian_conditional(y, scales)

    def test_means_shape_mismatch_raises(self):
        """Raises error for means shape mismatch."""
        y = torch.randn(10, 32)
        scales = torch.randn(10, 32)
        means = torch.randn(10, 64)
        with pytest.raises(ValueError, match="must match"):
            gaussian_conditional(y, scales, means)


class TestGaussianConditionalDevice:
    """Device compatibility tests."""

    def test_class_cpu(self):
        """Class works on CPU."""
        gc = GaussianConditional()
        y = torch.randn(2, 16, 4, 4, device="cpu")
        scales = torch.ones(2, 16, 4, 4, device="cpu")
        y_hat, likelihoods = gc(y, scales)
        assert y_hat.device.type == "cpu"

    def test_function_cpu(self):
        """Function works on CPU."""
        y = torch.randn(2, 16, 4, 4, device="cpu")
        scales = torch.ones(2, 16, 4, 4, device="cpu")
        y_hat, likelihoods = gaussian_conditional(y, scales)
        assert y_hat.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_class_cuda(self):
        """Class works on CUDA."""
        gc = GaussianConditional()
        y = torch.randn(2, 16, 4, 4, device="cuda")
        scales = torch.ones(2, 16, 4, 4, device="cuda")
        y_hat, likelihoods = gc(y, scales)
        assert y_hat.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_function_cuda(self):
        """Function works on CUDA."""
        y = torch.randn(2, 16, 4, 4, device="cuda")
        scales = torch.ones(2, 16, 4, 4, device="cuda")
        y_hat, likelihoods = gaussian_conditional(y, scales)
        assert y_hat.device.type == "cuda"
