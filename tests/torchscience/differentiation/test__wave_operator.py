"""Tests for wave operator."""

import math

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import laplacian, wave_operator


class TestWaveOperator:
    """Tests for wave_operator function."""

    def test_wave_operator_constant_speed(self):
        """Wave operator with constant speed equals c^2 * laplacian."""
        n = 32
        field = torch.randn(n, n)
        c = 2.0  # wave speed
        dx = 1.0 / n

        result = wave_operator(field, wave_speed=c, dx=dx)
        expected = c**2 * laplacian(field, dx=dx)

        torch.testing.assert_close(result, expected)

    def test_wave_operator_shape_2d(self):
        """Wave operator preserves field shape."""
        n = 32
        field = torch.randn(n, n)
        dx = 1.0 / n

        result = wave_operator(field, wave_speed=1.0, dx=dx)

        assert result.shape == field.shape

    def test_wave_operator_shape_3d(self):
        """3D wave operator."""
        n = 16
        field = torch.randn(n, n, n)
        c = 0.5
        dx = 1.0 / n

        result = wave_operator(field, wave_speed=c, dx=dx)
        expected = c**2 * laplacian(field, dx=dx)

        torch.testing.assert_close(result, expected)

    def test_wave_operator_variable_speed(self):
        """Wave operator with spatially varying speed."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
        c = 1.0 + 0.5 * X  # Spatially varying speed
        dx = 1.0 / (n - 1)

        result = wave_operator(field, wave_speed=c, dx=dx)

        # Variable speed: c^2 * laplacian(f)
        expected = c**2 * laplacian(field, dx=dx)
        torch.testing.assert_close(result, expected)

    def test_wave_operator_sinusoidal(self):
        """Wave operator on sinusoidal field."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # f = sin(2*pi*x)*sin(2*pi*y), laplacian(f) = -8*pi^2*f
        field = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
        c = 1.0

        result = wave_operator(field, wave_speed=c, dx=dx)
        expected = c**2 * (-8 * math.pi**2) * field

        # Check interior (boundary effects)
        interior_result = result[3:-3, 3:-3]
        interior_expected = expected[3:-3, 3:-3]
        torch.testing.assert_close(
            interior_result, interior_expected, atol=0.5, rtol=0.1
        )

    def test_wave_operator_linear_field(self):
        """Linear field has zero Laplacian, hence zero wave operator."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = X + Y  # Linear
        dx = 1.0 / (n - 1)
        c = 1.0  # Use unit speed to avoid amplifying truncation errors

        result = wave_operator(field, wave_speed=c, dx=dx)

        # Interior should be near zero (finite-difference truncation error)
        interior = result[3:-3, 3:-3]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=1e-3, rtol=1e-3
        )

    def test_wave_operator_quadratic_field(self):
        """Quadratic field has constant Laplacian."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = X**2 + Y**2  # Laplacian = 4
        dx = 1.0 / (n - 1)
        c = 2.0

        result = wave_operator(field, wave_speed=c, dx=dx)

        # Interior should be c^2 * 4 = 16
        interior = result[3:-3, 3:-3]
        expected = 4 * c**2 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)

    def test_wave_operator_3d_linear_field(self):
        """3D linear field has zero wave operator."""
        n = 16
        x = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        field = X + Y + Z  # Linear
        dx = 1.0 / (n - 1)

        result = wave_operator(field, wave_speed=1.0, dx=dx)

        # Interior should be near zero
        interior = result[3:-3, 3:-3, 3:-3]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=1e-3, rtol=1e-3
        )

    def test_wave_operator_anisotropic_spacing(self):
        """Wave operator works with different spacing per dimension."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)
        dy = 2.0 / (n - 1)

        # f = x^2 + y^2, Laplacian = 2 + 2 = 4
        field = X**2 + Y**2
        c = 0.5

        result = wave_operator(field, wave_speed=c, dx=(dx, dy))

        # Interior should be c^2 * 4 = 1
        interior = result[3:-3, 3:-3]
        expected = 4 * c**2 * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)


class TestWaveOperatorAutograd:
    """Autograd tests for wave_operator."""

    def test_gradcheck_constant_speed(self):
        """Wave operator passes gradcheck with constant speed."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: wave_operator(f, wave_speed=1.5, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_variable_speed(self):
        """Wave operator passes gradcheck with variable speed."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        c = torch.rand(8, 8, dtype=torch.float64, requires_grad=True) + 0.5

        assert gradcheck(
            lambda f, s: wave_operator(f, wave_speed=s, dx=0.1),
            (field, c),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Wave operator passes gradcheck for 3D."""
        field = torch.randn(6, 6, 6, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: wave_operator(f, wave_speed=1.0, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Wave operator passes gradgradcheck for 2D."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradgradcheck(
            lambda f: wave_operator(f, wave_speed=1.5, dx=0.1).sum(),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestWaveOperatorAutocast:
    """Autocast tests for wave_operator."""

    def test_wave_operator_autocast_2d(self):
        """Wave operator upcasts to fp32 under autocast for 2D."""
        field = torch.randn(16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = wave_operator(field, wave_speed=343.0, dx=0.01)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_wave_operator_autocast_3d(self):
        """Wave operator upcasts to fp32 under autocast for 3D."""
        field = torch.randn(8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = wave_operator(field, wave_speed=1500.0, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestWaveOperatorVmap:
    """Vmap tests for wave_operator."""

    def test_wave_operator_vmap_2d(self):
        """Wave operator works with torch.vmap for 2D."""
        batch_field = torch.randn(4, 16, 16)

        # vmap over batch dimension
        batched_wave_op = torch.vmap(
            lambda f: wave_operator(f, wave_speed=343.0, dx=0.01), in_dims=0
        )

        result = batched_wave_op(batch_field)

        # Each field is (16, 16), wave operator output is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [
                wave_operator(batch_field[i], wave_speed=343.0, dx=0.01)
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)

    def test_wave_operator_vmap_3d(self):
        """Wave operator works with torch.vmap for 3D."""
        batch_field = torch.randn(4, 8, 8, 8)

        # vmap over batch dimension
        batched_wave_op = torch.vmap(
            lambda f: wave_operator(f, wave_speed=1.0, dx=0.1), in_dims=0
        )

        result = batched_wave_op(batch_field)

        # Each field is (8, 8, 8), wave operator output is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)
