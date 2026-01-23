"""Tests for stress tensor operator."""

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import stress_tensor


class TestStressTensor:
    """Tests for stress_tensor function."""

    def test_stress_tensor_shape(self):
        """Stress tensor has shape (ndim, ndim, *spatial)."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        pressure = torch.randn(n, n, n)
        mu = 0.1
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=dx)

        assert sigma.shape == (3, 3, n, n, n)

    def test_stress_tensor_2d_shape(self):
        """2D stress tensor has shape (2, 2, *spatial)."""
        n = 32
        velocity = torch.randn(2, n, n)
        pressure = torch.randn(n, n)
        mu = 0.1
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=dx)

        assert sigma.shape == (2, 2, n, n)

    def test_stress_tensor_symmetric(self):
        """Stress tensor is symmetric."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        pressure = torch.randn(n, n, n)
        mu = 0.1
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=dx)

        # sigma_ij = sigma_ji
        torch.testing.assert_close(sigma, sigma.transpose(0, 1))

    def test_stress_tensor_zero_viscosity(self):
        """Zero viscosity gives pressure-only stress: sigma_ij = -p*delta_ij."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        pressure = torch.randn(n, n, n)
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=0.0, spacing=dx)

        # Diagonal elements should be -p
        for i in range(3):
            torch.testing.assert_close(sigma[i, i], -pressure)

        # Off-diagonal elements should be zero
        torch.testing.assert_close(sigma[0, 1], torch.zeros_like(sigma[0, 1]))
        torch.testing.assert_close(sigma[0, 2], torch.zeros_like(sigma[0, 2]))
        torch.testing.assert_close(sigma[1, 2], torch.zeros_like(sigma[1, 2]))

    def test_stress_tensor_uniform_pressure(self):
        """Uniform pressure contributes -p to diagonal."""
        n = 16
        velocity = torch.zeros(3, n, n, n)  # No flow
        pressure = 5.0 * torch.ones(n, n, n)
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=0.1, spacing=dx)

        # With zero velocity, diagonal should be -p
        for i in range(3):
            torch.testing.assert_close(
                sigma[i, i], -pressure, atol=1e-5, rtol=1e-5
            )

    def test_stress_tensor_zero_velocity_zero_viscous(self):
        """Zero velocity produces zero viscous stress."""
        n = 16
        velocity = torch.zeros(3, n, n, n)
        pressure = torch.randn(n, n, n)
        mu = 0.5
        dx = 1.0 / n

        sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=dx)

        # With zero velocity, should be -p*delta_ij (no viscous contribution)
        for i in range(3):
            torch.testing.assert_close(
                sigma[i, i], -pressure, atol=1e-5, rtol=1e-5
            )
        # Off-diagonals should be zero
        torch.testing.assert_close(
            sigma[0, 1], torch.zeros_like(sigma[0, 1]), atol=1e-5, rtol=1e-5
        )

    def test_simple_shear_flow(self):
        """Simple shear flow produces off-diagonal shear stress."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Simple shear: v_x = gamma*y, v_y = 0
        gamma = 0.2
        vx = gamma * Y
        vy = torch.zeros_like(Y)
        velocity = torch.stack([vx, vy], dim=0)
        pressure = torch.zeros(n, n)
        mu = 1.0

        sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=dx)

        # sigma_xy = mu*(dv_x/dy + dv_y/dx) = mu*gamma
        # Check interior
        interior = sigma[:, :, 5:-5, 5:-5]
        torch.testing.assert_close(
            interior[0, 1].mean(),
            torch.tensor(mu * gamma),
            atol=0.02,
            rtol=0.02,
        )

    def test_spatially_varying_viscosity(self):
        """Spatially varying viscosity works correctly."""
        n = 16
        velocity = torch.randn(2, n, n)
        pressure = torch.randn(n, n)
        # Viscosity varies spatially
        viscosity = 0.1 * torch.ones(n, n)
        dx = 1.0 / n

        sigma = stress_tensor(
            velocity, pressure, viscosity=viscosity, spacing=dx
        )

        assert sigma.shape == (2, 2, n, n)
        # Should still be symmetric
        torch.testing.assert_close(sigma, sigma.transpose(0, 1))


class TestStressTensorAutograd:
    """Autograd tests for stress_tensor."""

    def test_gradcheck_velocity(self):
        """Stress tensor passes gradcheck for velocity."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )
        pressure = torch.randn(6, 6, 6, dtype=torch.float64)

        assert gradcheck(
            lambda v: stress_tensor(v, pressure, viscosity=0.1, spacing=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_pressure(self):
        """Stress tensor passes gradcheck for pressure."""
        velocity = torch.randn(3, 6, 6, 6, dtype=torch.float64)
        pressure = torch.randn(
            6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda p: stress_tensor(velocity, p, viscosity=0.1, spacing=0.1),
            (pressure,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_2d(self):
        """Stress tensor passes gradcheck for 2D."""
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )
        pressure = torch.randn(8, 8, dtype=torch.float64)

        assert gradcheck(
            lambda v: stress_tensor(v, pressure, viscosity=0.1, spacing=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_velocity(self):
        """Stress tensor passes gradgradcheck for velocity."""
        velocity = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )
        pressure = torch.randn(8, 8, dtype=torch.float64)

        assert gradgradcheck(
            lambda v: stress_tensor(
                v, pressure, viscosity=0.1, spacing=0.1
            ).sum(),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_pressure(self):
        """Stress tensor passes gradgradcheck for pressure."""
        velocity = torch.randn(2, 8, 8, dtype=torch.float64)
        pressure = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradgradcheck(
            lambda p: stress_tensor(
                velocity, p, viscosity=0.1, spacing=0.1
            ).sum(),
            (pressure,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestStressTensorAutocast:
    """Autocast tests for stress_tensor."""

    def test_stress_tensor_autocast_2d(self):
        """Stress tensor upcasts to fp32 under autocast for 2D."""
        velocity = torch.randn(2, 16, 16, dtype=torch.float16, device="cpu")
        pressure = torch.randn(16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = stress_tensor(
                velocity, pressure, viscosity=0.1, spacing=0.1
            )

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (2, 2, 16, 16)

    def test_stress_tensor_autocast_3d(self):
        """Stress tensor upcasts to fp32 under autocast for 3D."""
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")
        pressure = torch.randn(8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = stress_tensor(
                velocity, pressure, viscosity=0.1, spacing=0.1
            )

        assert result.dtype == torch.float32
        assert result.shape == (3, 3, 8, 8, 8)


class TestStressTensorVmap:
    """Vmap tests for stress_tensor."""

    def test_stress_tensor_vmap_2d(self):
        """Stress tensor works with torch.vmap for 2D."""
        batch_velocity = torch.randn(4, 2, 16, 16)
        batch_pressure = torch.randn(4, 16, 16)

        # vmap over batch dimension
        batched_stress = torch.vmap(
            lambda v, p: stress_tensor(v, p, viscosity=0.1, spacing=0.1),
            in_dims=(0, 0),
        )

        result = batched_stress(batch_velocity, batch_pressure)

        # Each velocity field is (2, 16, 16), stress tensor is (2, 2, 16, 16)
        assert result.shape == (4, 2, 2, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [
                stress_tensor(
                    batch_velocity[i],
                    batch_pressure[i],
                    viscosity=0.1,
                    spacing=0.1,
                )
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)

    def test_stress_tensor_vmap_3d(self):
        """Stress tensor works with torch.vmap for 3D."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)
        batch_pressure = torch.randn(4, 8, 8, 8)

        # vmap over batch dimension
        batched_stress = torch.vmap(
            lambda v, p: stress_tensor(v, p, viscosity=0.1, spacing=0.1),
            in_dims=(0, 0),
        )

        result = batched_stress(batch_velocity, batch_pressure)

        # Each velocity field is (3, 8, 8, 8), stress tensor is (3, 3, 8, 8, 8)
        assert result.shape == (4, 3, 3, 8, 8, 8)
