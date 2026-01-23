"""Tests for enstrophy operator."""

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import enstrophy, vorticity


class TestEnstrophy:
    """Tests for enstrophy function."""

    def test_enstrophy_field_2d(self):
        """Enstrophy field is half squared vorticity."""
        n = 32
        x = torch.linspace(-1, 1, n)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        dx = 2.0 / (n - 1)

        # Rigid body rotation
        velocity = torch.stack([-Y, X], dim=0)

        ens = enstrophy(velocity, dx=dx, integrated=False)
        omega = vorticity(velocity, dx=dx)
        expected = 0.5 * omega**2

        torch.testing.assert_close(ens, expected)

    def test_enstrophy_integrated_2d(self):
        """Integrated enstrophy is scalar."""
        n = 32
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        ens = enstrophy(velocity, dx=dx, integrated=True)

        assert ens.ndim == 0  # scalar

    def test_enstrophy_field_3d(self):
        """3D enstrophy from vector vorticity."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        ens = enstrophy(velocity, dx=dx, integrated=False)
        omega = vorticity(velocity, dx=dx)
        expected = 0.5 * (omega**2).sum(dim=0)

        torch.testing.assert_close(ens, expected)

    def test_enstrophy_shape_2d(self):
        """2D enstrophy field has same shape as velocity spatial dims."""
        n = 32
        velocity = torch.randn(2, n, n)

        ens = enstrophy(velocity, dx=0.1, integrated=False)

        assert ens.shape == (n, n)

    def test_enstrophy_shape_3d(self):
        """3D enstrophy field has same shape as velocity spatial dims."""
        n = 16
        velocity = torch.randn(3, n, n, n)

        ens = enstrophy(velocity, dx=0.1, integrated=False)

        assert ens.shape == (n, n, n)

    def test_enstrophy_positive(self):
        """Enstrophy is always non-negative."""
        n = 32
        velocity = torch.randn(2, n, n)

        ens = enstrophy(velocity, dx=0.1, integrated=False)

        assert (ens >= 0).all()

    def test_enstrophy_rigid_rotation_2d(self):
        """Enstrophy of rigid body rotation is constant."""
        n = 32
        x = torch.linspace(-1, 1, n)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        dx = 2.0 / (n - 1)

        # Rigid body rotation: v = (-y, x), vorticity = 2
        velocity = torch.stack([-Y, X], dim=0)

        ens = enstrophy(velocity, dx=dx, integrated=False)

        # Enstrophy = 0.5 * omega^2 = 0.5 * 4 = 2
        interior = ens[5:-5, 5:-5]
        expected = 2.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, rtol=0.1, atol=0.1)

    def test_enstrophy_anisotropic_spacing(self):
        """Enstrophy works with different spacing per dimension."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-2, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)
        dy = 4.0 / (n - 1)

        velocity = torch.stack([-Y, X], dim=0)

        ens = enstrophy(velocity, dx=(dx, dy), integrated=False)
        omega = vorticity(velocity, dx=(dx, dy))
        expected = 0.5 * omega**2

        torch.testing.assert_close(ens, expected)


class TestEnstrophyAutograd:
    """Autograd tests for enstrophy."""

    def test_gradcheck_2d(self):
        """Enstrophy passes gradcheck for 2D."""
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: enstrophy(v, dx=0.1, integrated=False),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Enstrophy passes gradcheck for 3D."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: enstrophy(v, dx=0.1, integrated=False),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_integrated(self):
        """Integrated enstrophy passes gradcheck."""
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: enstrophy(v, dx=0.1, integrated=True),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Enstrophy passes gradgradcheck for 2D."""
        from torch.autograd import gradgradcheck

        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda v: enstrophy(v, dx=0.1, integrated=False).sum(),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestEnstrophyAutocast:
    """Autocast tests for enstrophy."""

    def test_enstrophy_autocast_2d(self):
        """Enstrophy upcasts to fp32 under autocast for 2D."""
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = enstrophy(velocity, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_enstrophy_autocast_3d(self):
        """Enstrophy upcasts to fp32 under autocast for 3D."""
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = enstrophy(velocity, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestEnstrophyVmap:
    """Vmap tests for enstrophy."""

    def test_enstrophy_vmap_2d(self):
        """Enstrophy works with torch.vmap for 2D."""
        batch_velocity = torch.randn(4, 2, 16, 16)

        # vmap over batch dimension
        batched_enstrophy = torch.vmap(
            lambda v: enstrophy(v, dx=0.1), in_dims=0
        )

        result = batched_enstrophy(batch_velocity)

        # Each velocity field is (2, 16, 16), enstrophy is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [enstrophy(batch_velocity[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)

    def test_enstrophy_vmap_3d(self):
        """Enstrophy works with torch.vmap for 3D."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_enstrophy = torch.vmap(
            lambda v: enstrophy(v, dx=0.1), in_dims=0
        )

        result = batched_enstrophy(batch_velocity)

        # Each velocity field is (3, 8, 8, 8), enstrophy is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)
