"""Tests for reflect operator."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import reflect


class TestReflectShape:
    """Tests for shape handling."""

    def test_single_vector(self):
        """Single vector (3,) input."""
        direction = torch.tensor([1.0, -1.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = reflect(direction, normal)
        assert result.shape == (3,)

    def test_batch(self):
        """Batch of vectors (B, 3)."""
        direction = torch.randn(10, 3)
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.shape == (10, 3)

    def test_image_shape(self):
        """Image-like shape (H, W, 3)."""
        direction = torch.randn(64, 64, 3)
        normal = torch.randn(64, 64, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.shape == (64, 64, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            reflect(torch.randn(10, 4), torch.randn(10, 4))


class TestReflectKnownValues:
    """Tests for known reflection cases."""

    def test_vertical_reflection(self):
        """Ray hitting horizontal surface from above."""
        direction = torch.tensor([1.0, -1.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = reflect(direction, normal)
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_head_on_reflection(self):
        """Ray hitting surface head-on reverses direction."""
        direction = torch.tensor([0.0, -1.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = reflect(direction, normal)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_grazing_angle(self):
        """Ray parallel to surface is unchanged."""
        direction = torch.tensor([1.0, 0.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = reflect(direction, normal)
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_45_degree_angle(self):
        """Classic 45-degree reflection."""
        direction = torch.tensor([1.0, -1.0, 0.0]) / (2**0.5)
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = reflect(direction, normal)
        expected = torch.tensor([1.0, 1.0, 0.0]) / (2**0.5)
        assert torch.allclose(result, expected, atol=1e-5)


class TestReflectGradients:
    """Tests for gradient computation."""

    def test_gradcheck_direction(self):
        """Gradient check w.r.t. direction."""
        direction = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        assert gradcheck(
            lambda d: reflect(d, normal), (direction,), eps=1e-6, atol=1e-4
        )

    def test_gradcheck_normal(self):
        """Gradient check w.r.t. normal."""
        direction = torch.randn(5, 3, dtype=torch.float64)
        normal = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda n: reflect(direction, n / n.norm(dim=-1, keepdim=True)),
            (normal,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_both(self):
        """Gradient check w.r.t. both inputs."""
        direction = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        normal = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda d, n: reflect(d, n / n.norm(dim=-1, keepdim=True)),
            (direction, normal),
            eps=1e-6,
            atol=1e-4,
        )


class TestReflectDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        direction = torch.randn(10, 3, dtype=torch.float32)
        normal = torch.randn(10, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        direction = torch.randn(10, 3, dtype=torch.float64)
        normal = torch.randn(10, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        direction = torch.randn(10, 3, dtype=torch.bfloat16)
        normal = torch.randn(10, 3, dtype=torch.bfloat16)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        direction = torch.randn(10, 3, dtype=torch.float16)
        normal = torch.randn(10, 3, dtype=torch.float16)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        result = reflect(direction, normal)
        assert result.dtype == torch.float16

    def test_shape_mismatch_error(self):
        """Raises error when batch dimensions don't match."""
        direction = torch.randn(5, 3)
        normal = torch.randn(10, 3)
        with pytest.raises(RuntimeError, match="matching batch dimensions"):
            reflect(direction, normal)

    def test_dtype_mismatch_error(self):
        """Raises error when dtypes don't match."""
        direction = torch.randn(10, 3, dtype=torch.float32)
        normal = torch.randn(10, 3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="same dtype"):
            reflect(direction, normal)
