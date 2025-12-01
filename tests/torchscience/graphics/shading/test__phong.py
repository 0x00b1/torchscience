"""Tests for Phong specular reflectance."""

import pytest
import torch
from torch.autograd import gradcheck


class TestPhongBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output shape matches batch dimensions for single sample."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, 0.707, -0.707]])
        shininess = torch.tensor([32.0])

        result = phong(normal, view, light, shininess=shininess)

        assert result.shape == (1,)

    def test_output_shape_batch(self):
        """Output shape matches batch dimensions for batched input."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((10,), 32.0)

        result = phong(normal, view, light, shininess=shininess)

        assert result.shape == (10,)

    def test_specular_non_negative(self):
        """Specular values are always non-negative."""
        from torchscience.graphics.shading import phong

        torch.manual_seed(42)
        normal = torch.randn(100, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(100, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(100, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((100,), 32.0)

        result = phong(normal, view, light, shininess=shininess)

        assert (result >= 0).all()


class TestPhongCorrectness:
    """Tests for numerical correctness."""

    def test_perfect_reflection(self):
        """Maximum specular when view equals reflection direction."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor(
            [[0.0, 0.707106781, -0.707106781]], dtype=torch.float64
        )
        # Reflection of light about normal: R = 2(n.l)n - l
        # n.l = 0.707, so R = 2*0.707*(0,1,0) - (0,0.707,-0.707) = (0, 0.707, 0.707)
        view = torch.tensor(
            [[0.0, 0.707106781, 0.707106781]], dtype=torch.float64
        )
        shininess = torch.tensor([32.0], dtype=torch.float64)

        result = phong(normal, view, light, shininess=shininess)

        # R.v = 1, so result = 1^32 = 1.0
        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_back_facing_light_returns_zero(self):
        """Specular returns 0 when light is below surface."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon (n.l < 0)
        shininess = torch.tensor([32.0])

        result = phong(normal, view, light, shininess=shininess)

        assert result.item() == 0.0

    def test_off_specular_lower_value(self):
        """Off-specular directions have lower values than mirror direction."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707, -0.707]], dtype=torch.float64)
        # Mirror reflection
        n_dot_l = (normal * light).sum(dim=-1, keepdim=True)
        view_mirror = 2 * n_dot_l * normal - light
        # Off-specular view
        view_off = torch.tensor([[0.3, 0.7, 0.648]], dtype=torch.float64)
        view_off = view_off / view_off.norm(dim=-1, keepdim=True)
        shininess = torch.tensor([32.0], dtype=torch.float64)

        result_mirror = phong(normal, view_mirror, light, shininess=shininess)
        result_off = phong(normal, view_off, light, shininess=shininess)

        assert result_mirror.item() > result_off.item()


class TestPhongGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.shading import phong

        torch.manual_seed(123)
        normal = torch.randn(3, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(3, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        shininess = torch.tensor(
            [32.0, 16.0, 64.0], dtype=torch.float64, requires_grad=True
        )

        def func(n, v, l, s):
            return phong(n, v, l, shininess=s)

        assert gradcheck(
            func, (normal, view, light, shininess), raise_exception=True
        )

    def test_gradients_finite(self):
        """Gradients are finite for typical inputs."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        view = torch.tensor(
            [[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True
        )
        light = torch.tensor(
            [[0.0, 0.707, -0.707]], dtype=torch.float64, requires_grad=True
        )
        shininess = torch.tensor(
            [32.0], dtype=torch.float64, requires_grad=True
        )

        result = phong(normal, view, light, shininess=shininess)
        result.sum().backward()

        assert normal.grad is not None and torch.isfinite(normal.grad).all()
        assert view.grad is not None and torch.isfinite(view.grad).all()
        assert light.grad is not None and torch.isfinite(light.grad).all()
        assert (
            shininess.grad is not None and torch.isfinite(shininess.grad).all()
        )


class TestPhongValidation:
    """Tests for input validation."""

    def test_normal_wrong_dimension(self):
        """Raises error when normal last dimension != 3."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(10, 2)
        view = torch.randn(10, 3)
        light = torch.randn(10, 3)
        shininess = torch.full((10,), 32.0)

        with pytest.raises(
            ValueError, match="normal must have last dimension 3"
        ):
            phong(normal, view, light, shininess=shininess)


class TestPhongDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(5, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float32)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float32)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((5,), 32.0, dtype=torch.float32)

        result = phong(normal, view, light, shininess=shininess)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((5,), 32.0, dtype=torch.float64)

        result = phong(normal, view, light, shininess=shininess)

        assert result.dtype == torch.float64
