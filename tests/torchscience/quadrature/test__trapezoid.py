import pytest
import scipy.integrate
import torch


class TestTrapezoid:
    def test_uniform_spacing(self):
        """Integrate sin(x) from 0 to pi = 2.0"""
        x = torch.linspace(0, torch.pi, 1000)
        y = torch.sin(x)

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x)

        expected = scipy.integrate.trapezoid(y.numpy(), x.numpy())
        assert torch.allclose(result, torch.tensor(expected), rtol=1e-6)

    def test_dx_parameter(self):
        """Test with dx instead of x"""
        y = torch.sin(torch.linspace(0, torch.pi, 1000))
        dx = torch.pi / 999

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, dx=dx)

        assert torch.allclose(result, torch.tensor(2.0), rtol=1e-3)

    def test_matches_torch_trapezoid(self):
        """Result should match torch.trapezoid"""
        x = torch.linspace(0, 1, 100)
        y = x**2

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x)
        expected = torch.trapezoid(y, x)

        assert torch.allclose(result, expected)


class TestTrapezoidGradients:
    def test_gradient_wrt_y(self):
        """Gradient should flow through y"""
        x = torch.linspace(0, 1, 10)
        y = torch.randn(10, requires_grad=True, dtype=torch.float64)

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x)
        result.backward()

        assert y.grad is not None
        assert y.grad.shape == y.shape

    def test_gradient_wrt_x(self):
        """Gradient should flow through x (non-uniform spacing)"""
        x = torch.linspace(0, 1, 10, requires_grad=True, dtype=torch.float64)
        y = x**2

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x)
        result.backward()

        assert x.grad is not None

    def test_gradcheck_y(self):
        """Numerical gradient check for y"""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.randn(10, requires_grad=True, dtype=torch.float64)

        from torchscience.quadrature import trapezoid

        assert torch.autograd.gradcheck(
            lambda y_: trapezoid(y_, x),
            (y,),
            raise_exception=True,
        )

    def test_gradgradcheck_y(self):
        """Second-order gradient check for y"""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.randn(10, requires_grad=True, dtype=torch.float64)

        from torchscience.quadrature import trapezoid

        assert torch.autograd.gradgradcheck(
            lambda y_: trapezoid(y_, x),
            (y,),
            raise_exception=True,
        )


class TestTrapezoidBatching:
    def test_batch_dimension(self):
        """Integrate over batched data"""
        x = torch.linspace(0, 1, 100)
        y = torch.stack([x**2, x**3, x**4], dim=0)  # Shape: (3, 100)

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x, dim=-1)

        assert result.shape == (3,)
        assert torch.allclose(result[0], torch.tensor(1 / 3), rtol=1e-2)
        assert torch.allclose(result[1], torch.tensor(1 / 4), rtol=1e-2)
        assert torch.allclose(result[2], torch.tensor(1 / 5), rtol=1e-2)

    def test_dim_parameter(self):
        """Test different dim values"""
        y = torch.randn(5, 100, 3)
        x = torch.linspace(0, 1, 100)

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x, dim=1)

        assert result.shape == (5, 3)


class TestTrapezoidEdgeCases:
    def test_single_point_returns_zero(self):
        """Single point has zero area"""
        y = torch.tensor([1.0])

        from torchscience.quadrature import trapezoid

        result = trapezoid(y)

        assert result.item() == 0.0

    def test_two_points(self):
        """Two points = one trapezoid"""
        y = torch.tensor([1.0, 2.0])

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, dx=1.0)

        assert result.item() == 1.5  # (1 + 2) / 2 * 1


class TestTrapezoidDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input"""
        x = torch.linspace(0, 1, 10, dtype=dtype)
        y = torch.sin(x)

        from torchscience.quadrature import trapezoid

        result = trapezoid(y, x)

        assert result.dtype == dtype


class TestCumulativeTrapezoid:
    def test_matches_scipy(self):
        """Compare with scipy.integrate.cumulative_trapezoid"""
        x = torch.linspace(0, torch.pi, 100)
        y = torch.sin(x)

        from torchscience.quadrature import cumulative_trapezoid

        result = cumulative_trapezoid(y, x)

        expected = scipy.integrate.cumulative_trapezoid(y.numpy(), x.numpy())
        assert torch.allclose(
            result, torch.tensor(expected, dtype=result.dtype), rtol=1e-5
        )

    def test_output_shape_without_initial(self):
        """Without initial, output has one fewer element"""
        y = torch.randn(100)

        from torchscience.quadrature import cumulative_trapezoid

        result = cumulative_trapezoid(y)

        assert result.shape == (99,)

    def test_output_shape_with_initial(self):
        """With initial, output has same shape as input"""
        y = torch.randn(100)

        from torchscience.quadrature import cumulative_trapezoid

        result = cumulative_trapezoid(y, initial=0.0)

        assert result.shape == (100,)
        assert result[0].item() == 0.0

    def test_final_value_matches_trapezoid(self):
        """Final cumulative value should match total trapezoid"""
        x = torch.linspace(0, 1, 50)
        y = x**2

        from torchscience.quadrature import (
            cumulative_trapezoid,
            trapezoid,
        )

        total = trapezoid(y, x)
        cumulative = cumulative_trapezoid(y, x)

        assert torch.allclose(cumulative[-1], total)


class TestCumulativeTrapezoidGradients:
    def test_gradcheck(self):
        """Numerical gradient check"""
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = torch.randn(20, requires_grad=True, dtype=torch.float64)

        from torchscience.quadrature import cumulative_trapezoid

        def fn(y_):
            return cumulative_trapezoid(y_, x).sum()

        assert torch.autograd.gradcheck(fn, (y,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient check"""
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = torch.randn(20, requires_grad=True, dtype=torch.float64)

        from torchscience.quadrature import cumulative_trapezoid

        def fn(y_):
            return cumulative_trapezoid(y_, x).sum()

        assert torch.autograd.gradgradcheck(fn, (y,), raise_exception=True)
