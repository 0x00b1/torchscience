"""Tests for Q-criterion operator."""

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import q_criterion


class TestQCriterion:
    """Tests for Q-criterion function."""

    def test_q_criterion_shape(self):
        """Q-criterion has same spatial shape as velocity."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = 1.0 / n

        Q = q_criterion(velocity, dx=dx)

        assert Q.shape == (n, n, n)

    def test_rigid_body_rotation(self):
        """Rigid body rotation has positive Q (pure rotation, no strain)."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        z = torch.linspace(-1, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 2.0 / (n - 1)

        # Rotation around z-axis: v = (-y, x, 0)
        vx = -Y
        vy = X
        vz = torch.zeros_like(X)
        velocity = torch.stack([vx, vy, vz], dim=0)

        Q = q_criterion(velocity, dx=dx)

        # Q should be positive for rotation-dominated flow
        # Interior points (away from boundary effects)
        interior = Q[5:-5, 5:-5, 5:-5]
        assert (interior > 0).float().mean() > 0.9

    def test_pure_strain(self):
        """Pure extensional strain has negative Q."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        z = torch.linspace(-1, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 2.0 / (n - 1)

        # Pure extensional strain: v = (x, -y/2, -z/2) - incompressible
        # This is a stagnation point flow
        velocity = torch.stack([X, -Y / 2, -Z / 2], dim=0)

        Q = q_criterion(velocity, dx=dx)

        # Q should be negative for strain-dominated flow
        interior = Q[5:-5, 5:-5, 5:-5]
        assert (interior < 0).float().mean() > 0.9

    def test_uniform_flow(self):
        """Uniform flow has Q = 0."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 1.0 / (n - 1)

        # Uniform flow: v = (1, 0, 0)
        velocity = torch.stack(
            [torch.ones_like(X), torch.zeros_like(Y), torch.zeros_like(Z)],
            dim=0,
        )

        Q = q_criterion(velocity, dx=dx)

        # Q should be zero (no rotation, no strain gradients)
        torch.testing.assert_close(
            Q, torch.zeros_like(Q), atol=1e-4, rtol=1e-4
        )

    def test_requires_3d_velocity(self):
        """Q-criterion raises error for non-3D velocity fields."""
        velocity_2d = torch.randn(2, 16, 16)

        try:
            q_criterion(velocity_2d, dx=0.1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "3D velocity field" in str(e)

    def test_anisotropic_spacing(self):
        """Q-criterion works with different spacing per dimension."""
        n = 16
        velocity = torch.randn(3, n, n, n)
        dx = (0.1, 0.2, 0.15)

        Q = q_criterion(velocity, dx=dx)

        assert Q.shape == (n, n, n)


class TestQCriterionAutograd:
    """Autograd tests for Q-criterion."""

    def test_gradcheck(self):
        """Q-criterion passes gradcheck."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda v: q_criterion(v, dx=0.1),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """Q-criterion passes gradgradcheck."""
        velocity = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda v: q_criterion(v, dx=0.1).sum(),
            (velocity,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestQCriterionAutocast:
    """Autocast tests for Q-criterion."""

    def test_q_criterion_autocast(self):
        """Q-criterion upcasts to fp32 under autocast."""
        velocity = torch.randn(3, 8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = q_criterion(velocity, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestQCriterionVmap:
    """Vmap tests for Q-criterion."""

    def test_q_criterion_vmap(self):
        """Q-criterion works with torch.vmap."""
        batch_velocity = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_q_criterion = torch.vmap(
            lambda v: q_criterion(v, dx=0.1), in_dims=0
        )

        result = batched_q_criterion(batch_velocity)

        # Each velocity field is (3, 8, 8, 8), Q-criterion is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)

        # Compare with manual loop
        manual = torch.stack(
            [q_criterion(batch_velocity[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)
