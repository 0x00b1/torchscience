"""Tests for material derivative operator."""

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import material_derivative


class TestMaterialDerivative:
    """Tests for material_derivative function."""

    def test_material_derivative_shape(self):
        """Material derivative preserves field shape."""
        n = 32
        field = torch.randn(n, n)
        velocity = torch.randn(2, n, n)
        time_deriv = torch.randn(n, n)
        dx = 1.0 / n

        result = material_derivative(
            field, velocity, time_derivative=time_deriv, dx=dx
        )

        assert result.shape == field.shape

    def test_zero_velocity_equals_time_derivative(self):
        """With zero velocity, Df/Dt = df/dt."""
        n = 32
        field = torch.randn(n, n)
        velocity = torch.zeros(2, n, n)
        time_deriv = torch.randn(n, n)
        dx = 1.0 / n

        result = material_derivative(
            field, velocity, time_derivative=time_deriv, dx=dx
        )

        torch.testing.assert_close(result, time_deriv)

    def test_zero_time_derivative_equals_advection(self):
        """With zero df/dt, Df/Dt = (v . grad)f."""
        from torchscience.differentiation import advect

        n = 32
        field = torch.randn(n, n)
        velocity = torch.randn(2, n, n)
        dx = 1.0 / n

        result = material_derivative(
            field, velocity, time_derivative=None, dx=dx
        )
        expected = advect(field, velocity, dx=dx)

        torch.testing.assert_close(result, expected)

    def test_steady_uniform_flow(self):
        """In steady uniform flow, Df/Dt = (v . grad)f."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # f = x, v = (1, 0), df/dt = 0
        field = X
        velocity = torch.stack(
            [torch.ones_like(X), torch.zeros_like(Y)], dim=0
        )

        result = material_derivative(
            field, velocity, time_derivative=None, dx=dx
        )

        # Df/Dt = 1 * 1 + 0 * 0 = 1
        interior = result[3:-3, 3:-3]
        expected = torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)

    def test_material_derivative_3d(self):
        """3D material derivative."""
        n = 16
        field = torch.randn(n, n, n)
        velocity = torch.randn(3, n, n, n)
        time_deriv = torch.randn(n, n, n)
        dx = 1.0 / n

        result = material_derivative(
            field, velocity, time_derivative=time_deriv, dx=dx
        )

        assert result.shape == field.shape

    def test_material_derivative_combines_terms(self):
        """Df/Dt = df/dt + (v . grad)f."""
        from torchscience.differentiation import advect

        n = 32
        field = torch.randn(n, n)
        velocity = torch.randn(2, n, n)
        time_deriv = torch.randn(n, n)
        dx = 1.0 / n

        result = material_derivative(
            field, velocity, time_derivative=time_deriv, dx=dx
        )
        advection = advect(field, velocity, dx=dx)
        expected = time_deriv + advection

        torch.testing.assert_close(result, expected)

    def test_material_derivative_anisotropic_spacing(self):
        """Material derivative works with different spacing per dimension."""
        n = 32
        field = torch.randn(n, n)
        velocity = torch.randn(2, n, n)
        time_deriv = torch.randn(n, n)
        dx = (0.1, 0.2)

        result = material_derivative(
            field, velocity, time_derivative=time_deriv, dx=dx
        )

        assert result.shape == field.shape


class TestMaterialDerivativeAutograd:
    """Autograd tests for material_derivative."""

    def test_gradcheck_field(self):
        """Material derivative passes gradcheck for field."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(2, 8, 8, dtype=torch.float64)
        time_deriv = torch.randn(8, 8, dtype=torch.float64)

        assert gradcheck(
            lambda f: material_derivative(
                f, velocity, time_derivative=time_deriv, dx=0.1
            ),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_velocity(self):
        """Material derivative passes gradcheck for velocity."""
        field = torch.randn(8, 8, dtype=torch.float64)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: material_derivative(
                field, v, time_derivative=None, dx=0.1
            ),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_time_derivative(self):
        """Material derivative passes gradcheck for time derivative."""
        field = torch.randn(8, 8, dtype=torch.float64)
        velocity = torch.randn(2, 8, 8, dtype=torch.float64)
        time_deriv = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda t: material_derivative(
                field, velocity, time_derivative=t, dx=0.1
            ),
            (time_deriv,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_all_inputs(self):
        """Material derivative passes gradcheck for all inputs."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )
        time_deriv = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f, v, t: material_derivative(
                f, v, time_derivative=t, dx=0.1
            ),
            (field, velocity, time_deriv),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Material derivative passes gradcheck for 3D."""
        field = torch.randn(6, 6, 6, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda f, v: material_derivative(
                f, v, time_derivative=None, dx=0.1
            ),
            (field, velocity),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Material derivative passes gradgradcheck for 2D."""
        from torch.autograd import gradgradcheck

        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda f, v: material_derivative(
                f, v, time_derivative=None, dx=0.1
            ).sum(),
            (field, velocity),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestMaterialDerivativeAutocast:
    """Autocast tests for material_derivative."""

    def test_autocast_2d(self):
        """Material derivative upcasts to fp32 under autocast for 2D."""
        field = torch.randn(16, 16, dtype=torch.float16, device="cpu")
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = material_derivative(
                field, velocity, time_derivative=None, dx=0.1
            )

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_autocast_3d(self):
        """Material derivative upcasts to fp32 under autocast for 3D."""
        field = torch.randn(8, 8, 8, dtype=torch.float16, device="cpu")
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = material_derivative(
                field, velocity, time_derivative=None, dx=0.1
            )

        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestMaterialDerivativeVmap:
    """Vmap tests for material_derivative."""

    def test_vmap_2d(self):
        """Material derivative works with torch.vmap for 2D."""
        batch_field = torch.randn(4, 16, 16)
        batch_velocity = torch.randn(4, 2, 16, 16)

        # vmap over batch dimension
        batched_material_deriv = torch.vmap(
            lambda f, v: material_derivative(
                f, v, time_derivative=None, dx=0.1
            ),
            in_dims=(0, 0),
        )

        result = batched_material_deriv(batch_field, batch_velocity)

        # Each field is (16, 16), output is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [
                material_derivative(
                    batch_field[i],
                    batch_velocity[i],
                    time_derivative=None,
                    dx=0.1,
                )
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)

    def test_vmap_3d(self):
        """Material derivative works with torch.vmap for 3D."""
        batch_field = torch.randn(4, 8, 8, 8)
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_material_deriv = torch.vmap(
            lambda f, v: material_derivative(
                f, v, time_derivative=None, dx=0.1
            ),
            in_dims=(0, 0),
        )

        result = batched_material_deriv(batch_field, batch_velocity)

        # Each field is (8, 8, 8), output is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)
