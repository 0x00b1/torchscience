"""Tests for morphological closing."""

import torch
from torch.autograd import gradcheck

from torchscience.morphology import closing


class TestClosingShape:
    """Tests for shape handling."""

    def test_2d_image(self):
        """2D image closing."""
        image = torch.rand(64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se)
        assert result.shape == (64, 64)

    def test_2d_batch(self):
        """Batch of 2D images."""
        batch = torch.rand(8, 64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(batch, se)
        assert result.shape == (8, 64, 64)


class TestClosingProperties:
    """Tests for mathematical properties of closing."""

    def test_extensive(self):
        """Closing is extensive: closing(f) >= f."""
        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se, padding_mode="replicate")
        assert (result >= image - 1e-6).all()

    def test_idempotent(self):
        """Closing is idempotent: closing(closing(f)) = closing(f)."""
        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)
        once = closing(image, se, padding_mode="replicate")
        twice = closing(once, se, padding_mode="replicate")
        assert torch.allclose(once, twice, atol=1e-5)

    def test_fills_small_dark_holes(self):
        """Closing fills small dark holes."""
        image = torch.ones(20, 20)
        image[10, 10] = 0.0  # Single dark pixel (hole)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se, padding_mode="replicate")
        # The single hole should be filled
        assert result[10, 10].item() == 1.0

    def test_preserves_large_structures(self):
        """Closing approximately preserves larger dark structures."""
        image = torch.ones(20, 20)
        image[5:15, 5:15] = 0.0  # 10x10 dark square
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se, padding_mode="replicate")
        # Most of the dark square should remain
        assert result[10, 10].item() == 0.0


class TestClosingDuality:
    """Tests for duality with opening."""

    def test_duality_with_opening(self):
        """closing(f) = -opening(-f)."""
        from torchscience.morphology import opening

        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)

        clos = closing(image, se, padding_mode="replicate")
        neg_open_neg = -opening(-image, se, padding_mode="replicate")

        assert torch.allclose(clos, neg_open_neg, atol=1e-5)


class TestClosingGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check for closing."""
        image = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        assert gradcheck(
            lambda x: closing(x, se, padding_mode="replicate"),
            (image,),
            eps=1e-6,
            atol=1e-4,
            nondet_tol=1e-4,
        )

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        image = torch.rand(16, 16, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se)
        loss = result.sum()
        loss.backward()
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()


class TestClosingDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        image = torch.rand(16, 16, dtype=torch.float32)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        image = torch.rand(16, 16, dtype=torch.float64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = closing(image, se)
        assert result.dtype == torch.float64
