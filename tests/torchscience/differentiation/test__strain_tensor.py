"""Tests for strain tensor operator."""

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import strain_tensor


class TestStrainTensor:
    """Tests for strain_tensor function."""

    def test_strain_tensor_shape_3d(self):
        """Strain tensor has shape (ndim, ndim, *spatial)."""
        n = 16
        displacement = torch.randn(3, n, n, n)
        dx = 1.0 / n

        eps = strain_tensor(displacement, dx=dx)

        assert eps.shape == (3, 3, n, n, n)

    def test_strain_tensor_shape_2d(self):
        """2D strain tensor has shape (2, 2, *spatial)."""
        n = 32
        displacement = torch.randn(2, n, n)
        dx = 1.0 / n

        eps = strain_tensor(displacement, dx=dx)

        assert eps.shape == (2, 2, n, n)

    def test_strain_tensor_symmetric(self):
        """Strain tensor is symmetric."""
        n = 16
        displacement = torch.randn(3, n, n, n)
        dx = 1.0 / n

        eps = strain_tensor(displacement, dx=dx)

        # eps_ij = eps_ji
        torch.testing.assert_close(eps, eps.transpose(0, 1))

    def test_uniform_compression(self):
        """Uniform compression has diagonal strain tensor."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 1.0 / (n - 1)

        # Uniform compression: u = -0.1*(x, y, z)
        scale = -0.1
        displacement = torch.stack([scale * X, scale * Y, scale * Z], dim=0)

        eps = strain_tensor(displacement, dx=dx)

        # Should have eps_ii = -0.1, eps_ij = 0 for i != j
        # Check interior (avoid boundary effects)
        interior = eps[:, :, 5:-5, 5:-5, 5:-5]
        for i in range(3):
            for j in range(3):
                if i == j:
                    torch.testing.assert_close(
                        interior[i, j].mean(),
                        torch.tensor(scale),
                        atol=0.02,
                        rtol=0.02,
                    )
                else:
                    torch.testing.assert_close(
                        interior[i, j].mean(),
                        torch.tensor(0.0),
                        atol=0.02,
                        rtol=0.02,
                    )

    def test_simple_shear(self):
        """Simple shear produces off-diagonal strain."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Simple shear: u_x = gamma * y, u_y = 0
        gamma = 0.2
        ux = gamma * Y
        uy = torch.zeros_like(Y)
        displacement = torch.stack([ux, uy], dim=0)

        eps = strain_tensor(displacement, dx=dx)

        # eps_xy = eps_yx = gamma/2
        interior = eps[:, :, 5:-5, 5:-5]
        torch.testing.assert_close(
            interior[0, 1].mean(),
            torch.tensor(gamma / 2),
            atol=0.02,
            rtol=0.02,
        )

    def test_rigid_body_rotation_zero_strain(self):
        """Rigid body rotation has zero strain."""
        n = 32
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)

        # Small rotation: u = (-theta*y, theta*x)
        theta = 0.1
        ux = -theta * Y
        uy = theta * X
        displacement = torch.stack([ux, uy], dim=0)

        eps = strain_tensor(displacement, dx=dx)

        # Should have zero strain (interior)
        interior = eps[:, :, 5:-5, 5:-5]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=0.02, rtol=0.02
        )

    def test_uniaxial_extension(self):
        """Uniaxial extension produces diagonal strain."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Uniaxial extension in x: u_x = 0.1*x, u_y = 0
        strain_val = 0.1
        ux = strain_val * X
        uy = torch.zeros_like(Y)
        displacement = torch.stack([ux, uy], dim=0)

        eps = strain_tensor(displacement, dx=dx)

        # eps_xx should be 0.1, eps_yy and eps_xy should be 0
        interior = eps[:, :, 3:-3, 3:-3]
        torch.testing.assert_close(
            interior[0, 0].mean(),
            torch.tensor(strain_val),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(
            interior[1, 1].mean(),
            torch.tensor(0.0),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(
            interior[0, 1].mean(),
            torch.tensor(0.0),
            atol=0.02,
            rtol=0.02,
        )

    def test_biaxial_strain(self):
        """Biaxial strain has distinct diagonal components."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Biaxial: u_x = 0.1*x, u_y = -0.05*y
        strain_xx = 0.1
        strain_yy = -0.05
        ux = strain_xx * X
        uy = strain_yy * Y
        displacement = torch.stack([ux, uy], dim=0)

        eps = strain_tensor(displacement, dx=dx)

        interior = eps[:, :, 3:-3, 3:-3]
        torch.testing.assert_close(
            interior[0, 0].mean(),
            torch.tensor(strain_xx),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(
            interior[1, 1].mean(),
            torch.tensor(strain_yy),
            atol=0.02,
            rtol=0.02,
        )


class TestStrainTensorAutograd:
    """Autograd tests for strain tensor."""

    def test_gradcheck_2d(self):
        """Strain tensor passes gradcheck for 2D."""
        displacement = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda u: strain_tensor(u, dx=0.1),
            (displacement,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Strain tensor passes gradcheck for 3D."""
        displacement = torch.randn(
            3, 6, 6, 6, dtype=torch.float64, requires_grad=True
        )

        assert gradcheck(
            lambda u: strain_tensor(u, dx=0.1),
            (displacement,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Strain tensor passes gradgradcheck for 2D."""
        displacement = torch.randn(
            2, 8, 8, dtype=torch.float64, requires_grad=True
        )

        assert gradgradcheck(
            lambda u: strain_tensor(u, dx=0.1).sum(),
            (displacement,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestStrainTensorAutocast:
    """Autocast tests for strain tensor."""

    def test_strain_tensor_autocast_2d(self):
        """Strain tensor upcasts to fp32 under autocast for 2D."""
        displacement = torch.randn(
            2, 16, 16, dtype=torch.float16, device="cpu"
        )

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = strain_tensor(displacement, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (2, 2, 16, 16)

    def test_strain_tensor_autocast_3d(self):
        """Strain tensor upcasts to fp32 under autocast for 3D."""
        displacement = torch.randn(
            3, 8, 8, 8, dtype=torch.float16, device="cpu"
        )

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = strain_tensor(displacement, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (3, 3, 8, 8, 8)


class TestStrainTensorVmap:
    """Vmap tests for strain tensor."""

    def test_strain_tensor_vmap_2d(self):
        """Strain tensor works with torch.vmap for 2D."""
        batch_displacement = torch.randn(4, 2, 16, 16)

        # vmap over batch dimension
        batched_strain = torch.vmap(
            lambda u: strain_tensor(u, dx=0.1), in_dims=0
        )

        result = batched_strain(batch_displacement)

        # Each displacement field is (2, 16, 16), strain tensor is (2, 2, 16, 16)
        assert result.shape == (4, 2, 2, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [strain_tensor(batch_displacement[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)

    def test_strain_tensor_vmap_3d(self):
        """Strain tensor works with torch.vmap for 3D."""
        batch_displacement = torch.randn(4, 3, 8, 8, 8)

        # vmap over batch dimension
        batched_strain = torch.vmap(
            lambda u: strain_tensor(u, dx=0.1), in_dims=0
        )

        result = batched_strain(batch_displacement)

        # Each displacement field is (3, 8, 8, 8), strain tensor is (3, 3, 8, 8, 8)
        assert result.shape == (4, 3, 3, 8, 8, 8)
