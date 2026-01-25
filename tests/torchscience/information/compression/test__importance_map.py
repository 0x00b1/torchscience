"""Tests for importance map and gain unit."""

import pytest
import torch

from torchscience.information.compression import gain_unit, importance_map


class TestImportanceMap:
    """Tests for importance_map function."""

    def test_output_shape(self):
        """Output shape is (batch, 1, H, W)."""
        x = torch.randn(4, 3, 64, 64)
        imp = importance_map(x)
        assert imp.shape == (4, 1, 64, 64)

    def test_single_channel_input(self):
        """Works with single channel input."""
        x = torch.randn(2, 1, 32, 32)
        imp = importance_map(x)
        assert imp.shape == (2, 1, 32, 32)

    def test_normalized_range(self):
        """Normalized output is in [0, 1]."""
        x = torch.randn(4, 3, 64, 64)
        imp = importance_map(x, normalize=True)
        assert imp.min() >= 0
        assert imp.max() <= 1

    def test_unnormalized_not_bounded(self):
        """Unnormalized output can be unbounded."""
        x = torch.randn(4, 3, 64, 64) * 10
        imp = importance_map(x, normalize=False)
        # Just check it runs without error
        assert imp.shape == (4, 1, 64, 64)

    def test_gradient_method(self):
        """Gradient method detects edges."""
        # Create image with vertical edge
        x = torch.zeros(1, 1, 32, 32)
        x[:, :, :, 16:] = 1.0

        imp = importance_map(x, method="gradient", normalize=False)

        # Importance should be highest at the edge (column 15-16)
        edge_importance = imp[0, 0, :, 14:18].mean()
        flat_importance = imp[0, 0, :, :10].mean()
        assert edge_importance > flat_importance * 5

    def test_variance_method(self):
        """Variance method detects texture."""
        # Create image with textured region
        x = torch.zeros(1, 1, 32, 32)
        x[:, :, :16, :] = torch.randn(1, 1, 16, 32) * 0.5  # Texture
        x[:, :, 16:, :] = 0.5  # Flat

        imp = importance_map(x, method="variance", normalize=False)

        # Texture region should have higher importance
        texture_importance = imp[0, 0, :14, :].mean()
        flat_importance = imp[0, 0, 18:, :].mean()
        assert texture_importance > flat_importance

    def test_entropy_method(self):
        """Entropy method works."""
        x = torch.randn(2, 3, 32, 32)
        imp = importance_map(x, method="entropy")
        assert imp.shape == (2, 1, 32, 32)

    def test_uniform_method(self):
        """Uniform method returns constant map."""
        x = torch.randn(2, 3, 32, 32)
        imp = importance_map(x, method="uniform", normalize=False)
        assert torch.allclose(imp, torch.ones_like(imp))

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            importance_map([[[1.0]]])

    def test_wrong_dims_raises(self):
        """Raises error for non-4D input."""
        x = torch.randn(3, 64, 64)  # 3D
        with pytest.raises(ValueError, match="4D"):
            importance_map(x)

    def test_invalid_method_raises(self):
        """Raises error for invalid method."""
        x = torch.randn(1, 1, 32, 32)
        with pytest.raises(ValueError, match="method must be"):
            importance_map(x, method="invalid")


class TestGainUnit:
    """Tests for gain_unit function."""

    def test_output_shape(self):
        """Output shape matches input."""
        x = torch.randn(4, 64, 16, 16)
        gain = torch.ones(4, 1, 16, 16)
        y = gain_unit(x, gain)
        assert y.shape == x.shape

    def test_multiplicative_mode(self):
        """Multiplicative mode multiplies."""
        x = torch.ones(2, 4, 8, 8) * 2
        gain = torch.ones(2, 1, 8, 8) * 3
        y = gain_unit(x, gain, mode="multiplicative")
        assert torch.allclose(y, torch.ones_like(y) * 6)

    def test_additive_mode(self):
        """Additive mode adds."""
        x = torch.ones(2, 4, 8, 8) * 2
        gain = torch.ones(2, 1, 8, 8) * 3
        y = gain_unit(x, gain, mode="additive")
        assert torch.allclose(y, torch.ones_like(y) * 5)

    def test_broadcast_gain(self):
        """Gain with 1 channel broadcasts to all channels."""
        x = torch.randn(2, 8, 16, 16)
        gain = torch.ones(2, 1, 16, 16) * 2
        y = gain_unit(x, gain)
        assert torch.allclose(y, x * 2)

    def test_channel_wise_gain(self):
        """Gain with matching channels works."""
        x = torch.randn(2, 8, 16, 16)
        gain = torch.randn(2, 8, 16, 16)
        y = gain_unit(x, gain)
        assert torch.allclose(y, x * gain)

    def test_gradients_flow(self):
        """Gradients flow through."""
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        gain = torch.randn(2, 1, 8, 8, requires_grad=True)
        y = gain_unit(x, gain)
        y.sum().backward()
        assert x.grad is not None
        assert gain.grad is not None

    def test_not_tensor_x_raises(self):
        """Raises error for non-tensor x."""
        gain = torch.randn(2, 1, 8, 8)
        with pytest.raises(TypeError, match="x must be a Tensor"):
            gain_unit([[[1.0]]], gain)

    def test_not_tensor_gain_raises(self):
        """Raises error for non-tensor gain."""
        x = torch.randn(2, 4, 8, 8)
        with pytest.raises(TypeError, match="gain must be a Tensor"):
            gain_unit(x, [[[1.0]]])

    def test_invalid_mode_raises(self):
        """Raises error for invalid mode."""
        x = torch.randn(2, 4, 8, 8)
        gain = torch.ones(2, 1, 8, 8)
        with pytest.raises(ValueError, match="mode must be"):
            gain_unit(x, gain, mode="invalid")


class TestImportanceMapDevice:
    """Device compatibility tests."""

    def test_importance_map_cpu(self):
        """importance_map works on CPU."""
        x = torch.randn(2, 3, 32, 32, device="cpu")
        imp = importance_map(x)
        assert imp.device.type == "cpu"

    def test_gain_unit_cpu(self):
        """gain_unit works on CPU."""
        x = torch.randn(2, 4, 16, 16, device="cpu")
        gain = torch.ones(2, 1, 16, 16, device="cpu")
        y = gain_unit(x, gain)
        assert y.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_importance_map_cuda(self):
        """importance_map works on CUDA."""
        x = torch.randn(2, 3, 32, 32, device="cuda")
        imp = importance_map(x)
        assert imp.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gain_unit_cuda(self):
        """gain_unit works on CUDA."""
        x = torch.randn(2, 4, 16, 16, device="cuda")
        gain = torch.ones(2, 1, 16, 16, device="cuda")
        y = gain_unit(x, gain)
        assert y.device.type == "cuda"
