"""Tests for entropy bottleneck."""

import pytest
import torch

from torchscience.compression._entropy_bottleneck import (
    EntropyBottleneck,
    entropy_bottleneck,
)


class TestEntropyBottleneckClass:
    """Tests for EntropyBottleneck class."""

    def test_initialization(self):
        """EntropyBottleneck initializes correctly."""
        eb = EntropyBottleneck(channels=192)
        assert eb.channels == 192

    def test_forward_output_types(self):
        """Forward returns tensor tuple."""
        eb = EntropyBottleneck(channels=64)
        x = torch.randn(2, 64, 8, 8)
        y, likelihoods = eb(x)
        assert isinstance(y, torch.Tensor)
        assert isinstance(likelihoods, torch.Tensor)

    def test_forward_output_shapes(self):
        """Forward preserves shape."""
        eb = EntropyBottleneck(channels=32)
        x = torch.randn(4, 32, 16, 16)
        y, likelihoods = eb(x)
        assert y.shape == x.shape
        assert likelihoods.shape == x.shape

    def test_training_mode_adds_noise(self):
        """Training mode adds noise instead of quantizing."""
        eb = EntropyBottleneck(channels=16)
        eb.train()
        x = torch.randn(2, 16, 4, 4)

        torch.manual_seed(42)
        y1, _ = eb(x)
        torch.manual_seed(43)
        y2, _ = eb(x)

        # Different seeds should give different outputs
        assert not torch.allclose(y1, y2)

    def test_eval_mode_quantizes(self):
        """Eval mode performs hard quantization."""
        eb = EntropyBottleneck(channels=16)
        eb.eval()
        x = torch.randn(2, 16, 4, 4) * 5  # Scale up for clear quantization

        y, _ = eb(x)

        # Output should be integers
        assert torch.allclose(y, y.round())

    def test_likelihoods_positive(self):
        """Likelihoods are always positive."""
        eb = EntropyBottleneck(channels=32)
        x = torch.randn(2, 32, 8, 8)
        _, likelihoods = eb(x)
        assert (likelihoods > 0).all()

    def test_likelihoods_at_most_one(self):
        """Likelihoods are at most 1."""
        eb = EntropyBottleneck(channels=32)
        x = torch.randn(2, 32, 8, 8)
        _, likelihoods = eb(x)
        assert (likelihoods <= 1).all()

    def test_compress_decompress_roundtrip(self):
        """Compress and decompress are inverses."""
        eb = EntropyBottleneck(channels=16)
        x = torch.randn(2, 16, 4, 4)

        y_compressed, symbols = eb.compress(x)
        y_decompressed = eb.decompress(symbols)

        assert torch.allclose(y_compressed, y_decompressed)

    def test_compress_produces_integers(self):
        """Compress produces integer symbols."""
        eb = EntropyBottleneck(channels=16)
        x = torch.randn(2, 16, 4, 4)

        _, symbols = eb.compress(x)
        assert symbols.dtype == torch.int32

    def test_gradients_flow_through(self):
        """Gradients flow through in training mode."""
        eb = EntropyBottleneck(channels=16)
        eb.train()
        x = torch.randn(2, 16, 4, 4, requires_grad=True)

        y, likelihoods = eb(x)
        loss = y.sum() - likelihoods.log().sum()
        loss.backward()

        assert x.grad is not None

    def test_different_filter_configs(self):
        """Different filter configurations work."""
        for filters in [(3,), (3, 3), (3, 3, 3, 3)]:
            eb = EntropyBottleneck(channels=16, filters=filters)
            x = torch.randn(2, 16, 4, 4)
            y, likelihoods = eb(x)
            assert y.shape == x.shape


class TestEntropyBottleneckFunction:
    """Tests for functional entropy_bottleneck."""

    def test_output_types(self):
        """Returns tensor tuple."""
        x = torch.randn(2, 32, 8, 8)
        y, likelihoods = entropy_bottleneck(x)
        assert isinstance(y, torch.Tensor)
        assert isinstance(likelihoods, torch.Tensor)

    def test_output_shapes(self):
        """Preserves input shape."""
        x = torch.randn(4, 64, 16, 16)
        y, likelihoods = entropy_bottleneck(x)
        assert y.shape == x.shape
        assert likelihoods.shape == x.shape

    def test_training_adds_noise(self):
        """Training mode adds noise."""
        x = torch.randn(2, 16, 4, 4)

        torch.manual_seed(42)
        y1, _ = entropy_bottleneck(x, training=True)
        torch.manual_seed(43)
        y2, _ = entropy_bottleneck(x, training=True)

        assert not torch.allclose(y1, y2)

    def test_eval_quantizes(self):
        """Eval mode quantizes to integers."""
        x = torch.randn(2, 16, 4, 4) * 5
        y, _ = entropy_bottleneck(x, training=False)
        assert torch.allclose(y, y.round())

    def test_likelihoods_positive(self):
        """Likelihoods are positive."""
        x = torch.randn(2, 32, 8, 8)
        _, likelihoods = entropy_bottleneck(x)
        assert (likelihoods > 0).all()

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            entropy_bottleneck([1.0, 2.0])

    def test_1d_input(self):
        """Works with 1D input."""
        x = torch.randn(100)
        y, likelihoods = entropy_bottleneck(x)
        assert y.shape == x.shape

    def test_2d_input(self):
        """Works with 2D input."""
        x = torch.randn(10, 32)
        y, likelihoods = entropy_bottleneck(x)
        assert y.shape == x.shape


class TestEntropyBottleneckDevice:
    """Device compatibility tests."""

    def test_class_cpu(self):
        """Class works on CPU."""
        eb = EntropyBottleneck(channels=16).cpu()
        x = torch.randn(2, 16, 4, 4, device="cpu")
        y, likelihoods = eb(x)
        assert y.device.type == "cpu"

    def test_function_cpu(self):
        """Function works on CPU."""
        x = torch.randn(2, 16, 4, 4, device="cpu")
        y, likelihoods = entropy_bottleneck(x)
        assert y.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_class_cuda(self):
        """Class works on CUDA."""
        eb = EntropyBottleneck(channels=16).cuda()
        x = torch.randn(2, 16, 4, 4, device="cuda")
        y, likelihoods = eb(x)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_function_cuda(self):
        """Function works on CUDA."""
        x = torch.randn(2, 16, 4, 4, device="cuda")
        y, likelihoods = entropy_bottleneck(x)
        assert y.device.type == "cuda"
