"""Tests for refract operator."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import refract


class TestRefractShape:
    """Tests for shape handling."""

    def test_single_vector(self):
        """Single vector (3,) input."""
        direction = torch.tensor([0.0, -1.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        eta = torch.tensor(1.0)
        result = refract(direction, normal, eta)
        assert result.shape == (3,)

    def test_batch(self):
        """Batch of vectors (B, 3)."""
        direction = torch.randn(10, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0)
        result = refract(direction, normal, eta)
        assert result.shape == (10, 3)

    def test_image_shape(self):
        """Image-like shape (H, W, 3)."""
        direction = torch.randn(64, 64, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(64, 64, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0)
        result = refract(direction, normal, eta)
        assert result.shape == (64, 64, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            refract(torch.randn(10, 4), torch.randn(10, 4), torch.tensor(1.0))


class TestRefractKnownValues:
    """Tests for known refraction cases."""

    def test_normal_incidence(self):
        """Ray hitting surface head-on passes through (with eta scaling)."""
        direction = torch.tensor([0.0, -1.0, 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        eta = torch.tensor(1.5)
        result = refract(direction, normal, eta)
        # At normal incidence, T = eta * D + (eta * 1 - 1) * N
        # = 1.5 * [0, -1, 0] + (1.5 - 1) * [0, 1, 0]
        # = [0, -1.5, 0] + [0, 0.5, 0] = [0, -1, 0]
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_eta_equals_one(self):
        """With eta=1, ray passes through unchanged."""
        direction = torch.tensor([0.5, -0.866025, 0.0])  # 30 degrees
        normal = torch.tensor([0.0, 1.0, 0.0])
        eta = torch.tensor(1.0)
        result = refract(direction, normal, eta)
        expected = direction  # Should be same direction
        assert torch.allclose(result, expected, atol=1e-5)

    def test_snells_law(self):
        """Verify Snell's law: n1*sin(theta1) = n2*sin(theta2)."""
        # Light coming from air (n1=1) into glass (n2=1.5) at 30 degrees
        theta1 = math.radians(30)
        n1, n2 = 1.0, 1.5
        eta = n1 / n2

        direction = torch.tensor([math.sin(theta1), -math.cos(theta1), 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = refract(direction, normal, torch.tensor(eta))

        # Calculate expected angle using Snell's law
        sin_theta2 = eta * math.sin(theta1)
        theta2 = math.asin(sin_theta2)
        expected = torch.tensor([math.sin(theta2), -math.cos(theta2), 0.0])

        assert torch.allclose(result, expected, atol=1e-5)

    def test_total_internal_reflection(self):
        """Beyond critical angle, returns zero vector."""
        # Glass (n1=1.5) to air (n2=1.0), critical angle is ~41.8 degrees
        # Using 60 degrees, which is above critical angle
        theta = math.radians(60)
        eta = 1.5 / 1.0  # n1/n2 = 1.5

        direction = torch.tensor([math.sin(theta), -math.cos(theta), 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = refract(direction, normal, torch.tensor(eta))

        # Should return zero vector for TIR
        expected = torch.zeros(3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_critical_angle(self):
        """At exactly critical angle, refracted ray grazes surface."""
        # Critical angle for glass (1.5) to air (1.0)
        n1, n2 = 1.5, 1.0
        critical_angle = math.asin(n2 / n1)  # ~41.81 degrees
        eta = n1 / n2

        # Slightly below critical to avoid numerical edge case
        theta = critical_angle - 0.01
        direction = torch.tensor([math.sin(theta), -math.cos(theta), 0.0])
        normal = torch.tensor([0.0, 1.0, 0.0])
        result = refract(direction, normal, torch.tensor(eta))

        # Result should be non-zero (valid refraction)
        assert result.norm() > 0.1

    def test_3d_refraction(self):
        """Test refraction with full 3D direction."""
        direction = torch.tensor([0.3, -0.9, 0.3])
        direction = direction / direction.norm()
        normal = torch.tensor([0.0, 1.0, 0.0])
        eta = torch.tensor(1.0)
        result = refract(direction, normal, eta)

        # With eta=1, should pass through unchanged
        assert torch.allclose(result, direction, atol=1e-5)


class TestRefractTensorEta:
    """Tests for tensor-valued eta (different eta per ray)."""

    def test_batched_eta(self):
        """Different eta for each ray in batch."""
        batch_size = 5
        direction = (
            torch.tensor([[0.0, -1.0, 0.0]]).expand(batch_size, 3).clone()
        )
        normal = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size, 3).clone()
        eta = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])

        result = refract(direction, normal, eta)
        assert result.shape == (batch_size, 3)

        # All results should be [0, -1, 0] at normal incidence
        expected = torch.tensor([[0.0, -1.0, 0.0]]).expand(batch_size, 3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scalar_eta_broadcast(self):
        """Scalar eta broadcasts to all rays."""
        batch_size = 10
        direction = torch.randn(batch_size, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(batch_size, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0)

        result = refract(direction, normal, eta)
        assert result.shape == (batch_size, 3)

    def test_mixed_tir_and_refract(self):
        """Batch with some rays experiencing TIR and others refracting."""
        # Ray 1: normal incidence, will refract
        # Ray 2: high angle with high eta, will TIR
        direction = torch.tensor(
            [
                [0.0, -1.0, 0.0],  # Normal incidence
                [0.866, -0.5, 0.0],  # 60 degrees
            ]
        )
        normal = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        eta = torch.tensor([1.0, 2.0])  # Second has high eta causing TIR

        result = refract(direction, normal, eta)

        # First should refract normally
        assert result[0].norm() > 0.5

        # Second should be zero (TIR)
        assert torch.allclose(result[1], torch.zeros(3), atol=1e-5)


class TestRefractGradients:
    """Tests for gradient computation."""

    def test_gradcheck_direction(self):
        """Gradient check w.r.t. direction."""
        direction = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        direction_normed = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(0.8, dtype=torch.float64)  # Low eta to avoid TIR

        # Use a lambda that normalizes to ensure valid input
        def func(d):
            d_normed = d / d.norm(dim=-1, keepdim=True)
            return refract(d_normed, normal, eta)

        assert gradcheck(func, (direction,), eps=1e-6, atol=1e-4)

    def test_gradcheck_normal(self):
        """Gradient check w.r.t. normal."""
        direction = torch.randn(5, 3, dtype=torch.float64)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        eta = torch.tensor(0.8, dtype=torch.float64)

        def func(n):
            n_normed = n / n.norm(dim=-1, keepdim=True)
            return refract(direction, n_normed, eta)

        assert gradcheck(func, (normal,), eps=1e-6, atol=1e-4)

    def test_gradcheck_eta(self):
        """Gradient check w.r.t. eta."""
        direction = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        eta = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda e: refract(direction, normal, e),
            (eta,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_all(self):
        """Gradient check w.r.t. all inputs."""
        direction = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        normal = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        eta = torch.tensor(
            [0.7, 0.8, 0.9], dtype=torch.float64, requires_grad=True
        )

        def func(d, n, e):
            d_normed = d / d.norm(dim=-1, keepdim=True)
            n_normed = n / n.norm(dim=-1, keepdim=True)
            return refract(d_normed, n_normed, e)

        assert gradcheck(func, (direction, normal, eta), eps=1e-6, atol=1e-4)


class TestRefractDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        direction = torch.randn(10, 3, dtype=torch.float32)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(10, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0, dtype=torch.float32)
        result = refract(direction, normal, eta)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        direction = torch.randn(10, 3, dtype=torch.float64)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(10, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0, dtype=torch.float64)
        result = refract(direction, normal, eta)
        assert result.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        direction = torch.randn(10, 3, dtype=torch.bfloat16)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(10, 3, dtype=torch.bfloat16)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0, dtype=torch.bfloat16)
        result = refract(direction, normal, eta)
        assert result.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        direction = torch.randn(10, 3, dtype=torch.float16)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        normal = torch.randn(10, 3, dtype=torch.float16)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        eta = torch.tensor(1.0, dtype=torch.float16)
        result = refract(direction, normal, eta)
        assert result.dtype == torch.float16


class TestRefractErrors:
    """Tests for error handling."""

    def test_shape_mismatch_error(self):
        """Raises error when batch dimensions don't match."""
        direction = torch.randn(5, 3)
        normal = torch.randn(10, 3)
        eta = torch.tensor(1.0)
        with pytest.raises(RuntimeError, match="matching batch dimensions"):
            refract(direction, normal, eta)

    def test_dtype_mismatch_direction_normal(self):
        """Raises error when direction and normal dtypes don't match."""
        direction = torch.randn(10, 3, dtype=torch.float32)
        normal = torch.randn(10, 3, dtype=torch.float64)
        eta = torch.tensor(1.0, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="same dtype"):
            refract(direction, normal, eta)

    def test_dtype_mismatch_eta(self):
        """Raises error when eta dtype doesn't match."""
        direction = torch.randn(10, 3, dtype=torch.float32)
        normal = torch.randn(10, 3, dtype=torch.float32)
        eta = torch.tensor(1.0, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="same dtype"):
            refract(direction, normal, eta)

    def test_eta_batch_mismatch(self):
        """Raises error when eta batch size doesn't match."""
        direction = torch.randn(10, 3)
        normal = torch.randn(10, 3)
        eta = torch.tensor([1.0, 1.0, 1.0])  # Wrong batch size
        with pytest.raises(RuntimeError, match="broadcast"):
            refract(direction, normal, eta)
