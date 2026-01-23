"""Tests for vorticity operator."""

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import vorticity


class TestVorticity:
    """Tests for vorticity function."""

    def test_rigid_body_rotation_2d(self):
        """Vorticity of rigid body rotation is constant."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)

        # Rigid body rotation: v = (-y, x) -> curl = 2
        vx = -Y
        vy = X
        velocity = torch.stack([vx, vy], dim=0)

        omega = vorticity(velocity, dx=dx)

        # Vorticity should be 2 everywhere (interior)
        interior = omega[5:-5, 5:-5]
        expected = 2.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, rtol=0.1, atol=0.1)

    def test_irrotational_flow_2d(self):
        """Vorticity of irrotational flow is zero."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Uniform flow: v = (1, 0) - irrotational
        velocity = torch.stack(
            [torch.ones_like(X), torch.zeros_like(Y)], dim=0
        )

        omega = vorticity(velocity, dx=dx)

        torch.testing.assert_close(
            omega, torch.zeros_like(omega), atol=1e-5, rtol=1e-5
        )

    def test_vorticity_3d_shape(self):
        """3D vorticity is a vector field."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        omega = vorticity(velocity, dx=dx)

        assert omega.shape == (3, n, n, n)

    def test_vorticity_3d_equals_curl(self):
        """3D vorticity equals curl of velocity."""
        from torchscience.differentiation import curl

        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        omega = vorticity(velocity, dx=dx)
        curl_v = curl(velocity, dx=dx)

        torch.testing.assert_close(omega, curl_v)

    def test_vorticity_2d_shape(self):
        """2D vorticity is a scalar field."""
        n = 32
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        omega = vorticity(velocity, dx=dx)

        assert omega.shape == (n, n)

    def test_vorticity_potential_flow_2d(self):
        """Vorticity of potential flow (gradient field) is zero."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Potential flow: v = grad(phi) where phi = x^2 + y^2
        # v = (2x, 2y), curl(v) = 0
        vx = 2 * X
        vy = 2 * Y
        velocity = torch.stack([vx, vy], dim=0)

        omega = vorticity(velocity, dx=dx)

        # Interior should be close to 0
        torch.testing.assert_close(
            omega[2:-2, 2:-2],
            torch.zeros(n - 4, n - 4),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_vorticity_shear_flow_2d(self):
        """Vorticity of linear shear flow."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Linear shear: v = (y, 0), omega = -1
        vx = Y
        vy = torch.zeros_like(Y)
        velocity = torch.stack([vx, vy], dim=0)

        omega = vorticity(velocity, dx=dx)

        # Interior should be -1
        torch.testing.assert_close(
            omega[3:-3, 3:-3],
            torch.full((n - 6, n - 6), -1.0),
            rtol=0.1,
            atol=0.1,
        )

    def test_vorticity_anisotropic_spacing(self):
        """Vorticity works with different spacing per dimension."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-2, 2, n)  # Different range
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)
        dy = 4.0 / (n - 1)

        # Rigid body rotation: v = (-y, x) -> curl = 2
        vx = -Y
        vy = X
        velocity = torch.stack([vx, vy], dim=0)

        omega = vorticity(velocity, dx=(dx, dy))

        # Vorticity should be 2 everywhere (interior)
        interior = omega[5:-5, 5:-5]
        expected = 2.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, rtol=0.15, atol=0.15)


class TestVorticityAutograd:
    """Autograd tests for vorticity."""

    def test_gradcheck_2d(self):
        """Vorticity passes gradcheck for 2D."""
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: vorticity(v, dx=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Vorticity passes gradcheck for 3D."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: vorticity(v, dx=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Vorticity passes gradgradcheck for 2D."""
        from torch.autograd import gradgradcheck

        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda v: vorticity(v, dx=0.1).sum(),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestVorticityAutocast:
    """Autocast tests for vorticity."""

    def test_vorticity_autocast_2d(self):
        """Vorticity upcasts to fp32 under autocast for 2D."""
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = vorticity(velocity, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_vorticity_autocast_3d(self):
        """Vorticity upcasts to fp32 under autocast for 3D."""
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = vorticity(velocity, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (3, 8, 8, 8)


class TestVorticityVmap:
    """Vmap tests for vorticity."""

    def test_vorticity_vmap_2d(self):
        """Vorticity works with torch.vmap for 2D."""
        batch_velocity = torch.randn(4, 2, 16, 16)

        # vmap over batch dimension
        batched_vorticity = torch.vmap(
            lambda v: vorticity(v, dx=0.1), in_dims=0
        )

        result = batched_vorticity(batch_velocity)

        # Each velocity field is (2, 16, 16), vorticity is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [vorticity(batch_velocity[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)

    def test_vorticity_vmap_3d(self):
        """Vorticity works with torch.vmap for 3D."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_vorticity = torch.vmap(
            lambda v: vorticity(v, dx=0.1), in_dims=0
        )

        result = batched_vorticity(batch_velocity)

        # Each velocity field is (3, 8, 8, 8), vorticity is (3, 8, 8, 8)
        assert result.shape == (4, 3, 8, 8, 8)
