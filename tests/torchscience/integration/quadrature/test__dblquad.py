import torch


class TestDblquad:
    def test_unit_square(self):
        """Integrate x*y over unit square = 1/4"""
        from torchscience.integration.quadrature import dblquad

        result = dblquad(lambda x, y: x * y, (0, 1), (0, 1))

        assert torch.allclose(
            result, torch.tensor(0.25, dtype=result.dtype), rtol=1e-6
        )

    def test_separable(self):
        """Separable integral: integral of sin(x)cos(y) dxdy"""
        from torchscience.integration.quadrature import dblquad

        # integral_0^pi sin(x) dx = 2
        # integral_0^(pi/2) cos(y) dy = 1
        # Product = 2
        result = dblquad(
            lambda x, y: torch.sin(x) * torch.cos(y),
            (0, torch.pi),
            (0, torch.pi / 2),
        )

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-6
        )

    def test_polynomial_exact(self):
        """Tensor product of Gauss-Legendre should be exact for polynomials"""
        from torchscience.integration.quadrature import dblquad

        # integral of x^2 * y^2 over [0,1] x [0,1] = (1/3) * (1/3) = 1/9
        result = dblquad(lambda x, y: x**2 * y**2, (0, 1), (0, 1), nx=5, ny=5)

        assert torch.allclose(
            result, torch.tensor(1 / 9, dtype=result.dtype), rtol=1e-10
        )

    def test_different_n(self):
        """Test with different nx and ny"""
        from torchscience.integration.quadrature import dblquad

        result = dblquad(lambda x, y: x * y, (0, 1), (0, 1), nx=10, ny=20)

        assert torch.allclose(
            result, torch.tensor(0.25, dtype=result.dtype), rtol=1e-6
        )


class TestDblquadBatched:
    def test_batched_x_bounds(self):
        """Batched integration bounds for x"""
        from torchscience.integration.quadrature import dblquad

        x_high = torch.tensor([1.0, 2.0, 3.0])

        # integral of 1 over [0, x_high] x [0, 1] = x_high
        result = dblquad(lambda x, y: torch.ones_like(x), (0, x_high), (0, 1))

        assert result.shape == (3,)
        assert torch.allclose(result, x_high, rtol=1e-6)

    def test_batched_both_bounds(self):
        """Batched bounds for both x and y"""
        from torchscience.integration.quadrature import dblquad

        r = torch.linspace(0.5, 2, 5)

        # integral of 1 over [-r, r] x [-r, r] = (2r)^2 = 4r^2
        result = dblquad(lambda x, y: torch.ones_like(x), (-r, r), (-r, r))

        expected = 4 * r**2
        assert torch.allclose(result, expected, rtol=1e-4)


class TestDblquadGradients:
    def test_gradient_closure(self):
        """Gradient through closure parameters"""
        from torchscience.integration.quadrature import dblquad

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        # integral of theta over [0,1] x [0,1] = theta
        result = dblquad(
            lambda x, y: theta * torch.ones_like(x), (0, 1), (0, 1)
        )
        result.backward()

        assert theta.grad is not None
        assert torch.allclose(
            theta.grad, torch.tensor(1.0, dtype=torch.float64), rtol=1e-6
        )

    def test_gradcheck_closure(self):
        """Numerical gradient check for closure"""
        from torchscience.integration.quadrature import dblquad

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def fn(theta_):
            return dblquad(
                lambda x, y: theta_ * x * y, (0, 1), (0, 1), nx=8, ny=8
            )

        assert torch.autograd.gradcheck(fn, (theta,), raise_exception=True)
