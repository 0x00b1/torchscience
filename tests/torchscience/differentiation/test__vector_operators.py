"""Tests for vector field differential operators."""

import pytest
import torch

from torchscience.differentiation import curl, divergence, jacobian


class TestDivergence:
    """Tests for divergence function."""

    def test_divergence_of_linear(self):
        """Divergence of (x, y) is 2."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        V = torch.stack([X, Y], dim=0)  # (2, n, n)

        div = divergence(V, dx=dx)

        # Shape should be (n, n)
        assert div.shape == (n, n)

        # Interior should be 2
        torch.testing.assert_close(
            div[2:-2, 2:-2],
            torch.full((n - 4, n - 4), 2.0),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_divergence_of_solenoidal(self):
        """Divergence of curl-like field is 0."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        # Solenoidal: V = (-y, x)
        V = torch.stack([-Y, X], dim=0)

        div = divergence(V, dx=dx)

        torch.testing.assert_close(
            div[2:-2, 2:-2],
            torch.zeros(n - 4, n - 4),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_divergence_3d(self):
        """3D divergence works."""
        V = torch.randn(3, 10, 10, 10)
        div = divergence(V, dx=0.1)
        assert div.shape == (10, 10, 10)


class TestCurl:
    """Tests for curl function."""

    def test_curl_of_gradient(self):
        """Curl of gradient field is zero."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        # Gradient of f = x*y*z is (yz, xz, xy)
        V = torch.stack([Y * Z, X * Z, X * Y], dim=0)

        c = curl(V, dx=dx)

        # Shape should be (3, n, n, n)
        assert c.shape == (3, n, n, n)

        # Interior should be close to 0
        torch.testing.assert_close(
            c[:, 3:-3, 3:-3, 3:-3],
            torch.zeros(3, n - 6, n - 6, n - 6),
            rtol=1e-1,
            atol=1e-2,
        )

    def test_curl_2d_raises(self):
        """Curl requires 3D."""
        V = torch.randn(2, 10, 10)
        with pytest.raises(ValueError, match="3D"):
            curl(V, dx=0.1)


class TestJacobian:
    """Tests for Jacobian function."""

    def test_jacobian_of_linear(self):
        """Jacobian of linear field is constant."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # V = (2x + 3y, 4x + 5y), Jacobian = [[2, 3], [4, 5]]
        V = torch.stack([2 * X + 3 * Y, 4 * X + 5 * Y], dim=0)

        J = jacobian(V, dx=dx)

        # Shape should be (2, 2, n, n)
        assert J.shape == (2, 2, n, n)

        # Check values in interior
        torch.testing.assert_close(
            J[0, 0, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 2.0),
            rtol=1e-1,
            atol=1e-2,
        )
        torch.testing.assert_close(
            J[0, 1, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 3.0),
            rtol=1e-1,
            atol=1e-2,
        )
        torch.testing.assert_close(
            J[1, 0, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 4.0),
            rtol=1e-1,
            atol=1e-2,
        )
        torch.testing.assert_close(
            J[1, 1, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 5.0),
            rtol=1e-1,
            atol=1e-2,
        )
