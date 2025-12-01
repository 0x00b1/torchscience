"""Tests for FiniteDifferenceStencil."""

import torch

from torchscience.differentiation import FiniteDifferenceStencil


class TestStencilImport:
    """Tests for stencil imports."""

    def test_stencil_importable(self):
        """FiniteDifferenceStencil is importable."""
        from torchscience.differentiation import FiniteDifferenceStencil

        assert FiniteDifferenceStencil is not None

    def test_exceptions_importable(self):
        """Exceptions are importable."""
        from torchscience.differentiation import StencilError

        assert StencilError is not None


class TestStencilConstruction:
    """Tests for stencil construction."""

    def test_1d_stencil_creation(self):
        """Create 1D stencil manually."""
        stencil = FiniteDifferenceStencil(
            offsets=torch.tensor([[-1], [0], [1]]),
            coeffs=torch.tensor([1.0, -2.0, 1.0]),
            derivative=(2,),
            accuracy=2,
        )
        assert stencil.ndim == 1
        assert stencil.n_points == 3
        assert stencil.order == 2

    def test_2d_stencil_creation(self):
        """Create 2D stencil manually."""
        stencil = FiniteDifferenceStencil(
            offsets=torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]),
            coeffs=torch.tensor([-4.0, 1.0, 1.0, 1.0, 1.0]),
            derivative=(2, 2),
            accuracy=2,
        )
        assert stencil.ndim == 2
        assert stencil.n_points == 5
        assert stencil.order == 4  # Sum of derivative orders

    def test_stencil_to_dense_1d(self):
        """Convert 1D stencil to dense kernel."""
        stencil = FiniteDifferenceStencil(
            offsets=torch.tensor([[-1], [0], [1]]),
            coeffs=torch.tensor([1.0, -2.0, 1.0]),
            derivative=(2,),
            accuracy=2,
        )
        kernel = stencil.to_dense()
        assert kernel.shape == (1, 1, 3)
        torch.testing.assert_close(
            kernel.squeeze(), torch.tensor([1.0, -2.0, 1.0])
        )

    def test_stencil_to_dense_2d(self):
        """Convert 2D stencil to dense kernel."""
        stencil = FiniteDifferenceStencil(
            offsets=torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]),
            coeffs=torch.tensor([-4.0, 1.0, 1.0, 1.0, 1.0]),
            derivative=(2, 2),
            accuracy=2,
        )
        kernel = stencil.to_dense()
        assert kernel.shape == (1, 1, 3, 3)
        # Center should be -4
        assert kernel[0, 0, 1, 1].item() == -4.0
        # Cardinal neighbors should be 1
        assert kernel[0, 0, 0, 1].item() == 1.0  # top
        assert kernel[0, 0, 2, 1].item() == 1.0  # bottom
        assert kernel[0, 0, 1, 0].item() == 1.0  # left
        assert kernel[0, 0, 1, 2].item() == 1.0  # right
        # Corners should be 0
        assert kernel[0, 0, 0, 0].item() == 0.0

    def test_stencil_preserves_dtype(self):
        """Stencil preserves dtype."""
        stencil = FiniteDifferenceStencil(
            offsets=torch.tensor([[-1], [0], [1]]),
            coeffs=torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64),
            derivative=(2,),
            accuracy=2,
        )
        kernel = stencil.to_dense()
        assert kernel.dtype == torch.float64
