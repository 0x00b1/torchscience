"""Tests for helicity operator."""

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import helicity, vorticity


class TestHelicity:
    """Tests for helicity function."""

    def test_helicity_field_3d(self):
        """Helicity is v . omega."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        h = helicity(velocity, dx=dx, integrated=False)
        omega = vorticity(velocity, dx=dx)
        expected = (velocity * omega).sum(dim=0)

        torch.testing.assert_close(h, expected)

    def test_helicity_integrated(self):
        """Integrated helicity is scalar."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        h = helicity(velocity, dx=dx, integrated=True)

        assert h.ndim == 0

    def test_helicity_zero_for_2d(self):
        """2D flows have zero helicity (vorticity perpendicular to velocity plane)."""
        n = 32
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        h = helicity(velocity, dx=dx, integrated=True)

        torch.testing.assert_close(h, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_helicity_zero_for_2d_field(self):
        """2D helicity field is zero everywhere."""
        n = 32
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        h = helicity(velocity, dx=dx, integrated=False)

        torch.testing.assert_close(h, torch.zeros(n, n), atol=1e-6, rtol=1e-6)

    def test_helicity_shape_3d(self):
        """3D helicity field has same shape as velocity spatial dims."""
        n = 16
        velocity = torch.randn(3, n, n, n)

        h = helicity(velocity, dx=0.1, integrated=False)

        assert h.shape == (n, n, n)

    def test_helicity_shape_2d(self):
        """2D helicity field has same shape as velocity spatial dims."""
        n = 32
        velocity = torch.randn(2, n, n)

        h = helicity(velocity, dx=0.1, integrated=False)

        assert h.shape == (n, n)

    def test_helicity_abc_flow(self):
        """ABC flow has non-zero helicity."""
        # Arnold-Beltrami-Childress flow has v parallel to omega
        import math

        n = 32
        pi = math.pi
        x = torch.linspace(0, 2 * pi, n)
        y = torch.linspace(0, 2 * pi, n)
        z = torch.linspace(0, 2 * pi, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 2 * pi / (n - 1)

        A, B, C = 1.0, 1.0, 1.0
        vx = A * torch.sin(Z) + C * torch.cos(Y)
        vy = B * torch.sin(X) + A * torch.cos(Z)
        vz = C * torch.sin(Y) + B * torch.cos(X)
        velocity = torch.stack([vx, vy, vz], dim=0)

        h = helicity(velocity, dx=dx, integrated=True)

        # ABC flow should have significant helicity
        assert h.abs().item() > 0.01

    def test_helicity_beltrami_property(self):
        """For ABC flow, vorticity is proportional to velocity."""
        # ABC flow is a Beltrami flow where omega = k * v (with k = 1)
        import math

        n = 32
        pi = math.pi
        x = torch.linspace(0, 2 * pi, n)
        y = torch.linspace(0, 2 * pi, n)
        z = torch.linspace(0, 2 * pi, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 2 * pi / (n - 1)

        A, B, C = 1.0, 1.0, 1.0
        vx = A * torch.sin(Z) + C * torch.cos(Y)
        vy = B * torch.sin(X) + A * torch.cos(Z)
        vz = C * torch.sin(Y) + B * torch.cos(X)
        velocity = torch.stack([vx, vy, vz], dim=0)

        h = helicity(velocity, dx=dx, integrated=False)

        # Helicity should be positive for ABC flow (right-handed helical structure)
        # Check that helicity has consistent sign in interior
        interior = h[5:-5, 5:-5, 5:-5]
        # For ABC flow, v . omega = v . v = |v|^2 > 0 in many regions
        assert interior.mean() > 0

    def test_helicity_anisotropic_spacing(self):
        """Helicity works with different spacing per dimension."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = (0.1, 0.2, 0.15)

        h = helicity(velocity, dx=dx, integrated=False)
        omega = vorticity(velocity, dx=dx)
        expected = (velocity * omega).sum(dim=0)

        torch.testing.assert_close(h, expected)


class TestHelicityAutograd:
    """Autograd tests for helicity."""

    def test_gradcheck_3d(self):
        """Helicity passes gradcheck for 3D."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: helicity(v, dx=0.1, integrated=False),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_integrated(self):
        """Integrated helicity passes gradcheck."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: helicity(v, dx=0.1, integrated=True),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_3d(self):
        """Helicity passes gradgradcheck for 3D."""
        from torch.autograd import gradgradcheck

        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda v: helicity(v, dx=0.1, integrated=False).sum(),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestHelicityAutocast:
    """Autocast tests for helicity."""

    def test_helicity_autocast_3d(self):
        """Helicity upcasts to fp32 under autocast for 3D."""
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = helicity(velocity, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)

    def test_helicity_autocast_2d(self):
        """Helicity handles 2D with autocast."""
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = helicity(velocity, dx=0.1)

        # 2D returns zeros, should still be fp32
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)


class TestHelicityVmap:
    """Vmap tests for helicity."""

    def test_helicity_vmap_3d(self):
        """Helicity works with torch.vmap for 3D."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_helicity = torch.vmap(lambda v: helicity(v, dx=0.1), in_dims=0)

        result = batched_helicity(batch_velocity)

        # Each velocity field is (3, 8, 8, 8), helicity is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)

        # Compare with manual loop
        manual = torch.stack(
            [helicity(batch_velocity[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)

    def test_helicity_vmap_integrated(self):
        """Integrated helicity works with torch.vmap."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_helicity = torch.vmap(
            lambda v: helicity(v, dx=0.1, integrated=True), in_dims=0
        )

        result = batched_helicity(batch_velocity)

        # Each helicity is a scalar, batch of scalars gives (4,)
        assert result.shape == (4,)

        # Compare with manual loop
        manual = torch.stack(
            [
                helicity(batch_velocity[i], dx=0.1, integrated=True)
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)
