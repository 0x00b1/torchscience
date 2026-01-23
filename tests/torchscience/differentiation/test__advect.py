"""Tests for advection operator."""

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import advect


class TestAdvect:
    """Tests for advect function."""

    def test_advect_shape_2d(self):
        """Advection preserves field shape."""
        n = 32
        field = torch.randn(n, n)
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        result = advect(field, velocity, dx=dx)

        assert result.shape == field.shape

    def test_advect_shape_3d(self):
        """3D advection preserves field shape."""
        n = 16
        field = torch.randn(n, n, n)
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        result = advect(field, velocity, dx=dx)

        assert result.shape == field.shape

    def test_zero_velocity(self):
        """Zero velocity means zero advection."""
        n = 32
        field = torch.randn(n, n)
        velocity = torch.zeros(2, n, n)
        dx = 1.0 / n

        result = advect(field, velocity, dx=dx)

        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-6, rtol=1e-6
        )

    def test_constant_field(self):
        """Constant field has zero advection."""
        n = 32
        field = torch.ones(n, n) * 5.0
        velocity = torch.randn(2, n, n)  # Any velocity
        dx = 1.0 / n

        result = advect(field, velocity, dx=dx)

        # Interior should be near zero (boundary effects exist)
        interior = result[3:-3, 3:-3]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=1e-4, rtol=1e-4
        )

    def test_linear_field_uniform_velocity(self):
        """Linear field with uniform velocity has constant advection."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # f = x + y, v = (1, 1)
        field = X + Y
        velocity = torch.stack([torch.ones_like(X), torch.ones_like(Y)], dim=0)

        result = advect(field, velocity, dx=dx)

        # (v . grad)f = 1*1 + 1*1 = 2
        interior = result[3:-3, 3:-3]
        expected = 2.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)

    def test_advect_3d_linear_field(self):
        """3D linear field advection."""
        n = 16
        x = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        dx = 1.0 / (n - 1)

        # f = x + 2y + 3z, v = (1, 2, 3)
        field = X + 2 * Y + 3 * Z
        velocity = torch.stack(
            [
                torch.ones_like(X),
                2 * torch.ones_like(Y),
                3 * torch.ones_like(Z),
            ],
            dim=0,
        )

        result = advect(field, velocity, dx=dx)

        # (v . grad)f = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        interior = result[3:-3, 3:-3, 3:-3]
        expected = 14.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.5, rtol=0.1)

    def test_advect_anisotropic_spacing(self):
        """Advection works with different spacing per dimension."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)
        dy = 2.0 / (n - 1)

        # f = x + y, v = (1, 1)
        field = X + Y
        velocity = torch.stack([torch.ones_like(X), torch.ones_like(Y)], dim=0)

        result = advect(field, velocity, dx=(dx, dy))

        # (v . grad)f = 1*1 + 1*1 = 2
        interior = result[3:-3, 3:-3]
        expected = 2.0 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)


class TestAdvectAutograd:
    """Autograd tests for advect."""

    def test_gradcheck_field(self):
        """Advection passes gradcheck for field input."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(2, 8, 8, dtype=torch.float64)

        assert gradcheck(
            lambda f: advect(f, velocity, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_velocity(self):
        """Advection passes gradcheck for velocity input."""
        field = torch.randn(8, 8, dtype=torch.float64)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: advect(field, v, dx=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_both(self):
        """Advection passes gradcheck for both inputs."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda f, v: advect(f, v, dx=0.1),
            (field, velocity),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Advection passes gradcheck for 3D."""
        field = torch.randn(6, 6, 6, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda f, v: advect(f, v, dx=0.1),
            (field, velocity),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Advection passes gradgradcheck for 2D."""
        from torch.autograd import gradgradcheck

        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda f, v: advect(f, v, dx=0.1).sum(),
            (field, velocity),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestAdvectAutocast:
    """Autocast tests for advect."""

    def test_advect_autocast_2d(self):
        """Advection upcasts to fp32 under autocast for 2D."""
        field = torch.randn(16, 16, dtype=torch.float16, device="cpu")
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = advect(field, velocity, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_advect_autocast_3d(self):
        """Advection upcasts to fp32 under autocast for 3D."""
        field = torch.randn(8, 8, 8, dtype=torch.float16, device="cpu")
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = advect(field, velocity, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestAdvectVmap:
    """Vmap tests for advect."""

    def test_advect_vmap_2d(self):
        """Advection works with torch.vmap for 2D."""
        batch_field = torch.randn(4, 16, 16)
        batch_velocity = torch.randn(4, 2, 16, 16)

        # vmap over batch dimension
        batched_advect = torch.vmap(
            lambda f, v: advect(f, v, dx=0.1), in_dims=(0, 0)
        )

        result = batched_advect(batch_field, batch_velocity)

        # Each field is (16, 16), advection output is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [
                advect(batch_field[i], batch_velocity[i], dx=0.1)
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)

    def test_advect_vmap_3d(self):
        """Advection works with torch.vmap for 3D."""
        batch_field = torch.randn(4, 8, 8, 8)
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_advect = torch.vmap(
            lambda f, v: advect(f, v, dx=0.1), in_dims=(0, 0)
        )

        result = batched_advect(batch_field, batch_velocity)

        # Each field is (8, 8, 8), advection output is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)
