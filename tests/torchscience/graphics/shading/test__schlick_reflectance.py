# tests/torchscience/graphics/shading/test__schlick_reflectance.py
"""Tests for Schlick reflectance approximation."""

import torch
from torch.autograd import gradcheck

from torchscience.graphics.shading import schlick_reflectance


class TestSchlickReflectanceShape:
    """Tests for shape handling."""

    def test_scalar_inputs(self):
        """Works with scalar inputs."""
        cosine = torch.tensor(0.5)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.shape == ()

    def test_batch_inputs(self):
        """Works with batched inputs."""
        cosine = torch.rand(10)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.shape == (10,)

    def test_2d_inputs(self):
        """Works with 2D inputs."""
        cosine = torch.rand(5, 10)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.shape == (5, 10)

    def test_tensor_ior_broadcast(self):
        """Tensor ior broadcasts correctly."""
        cosine = torch.rand(10)
        ior = torch.tensor(1.5)
        result = schlick_reflectance(cosine, ior=ior)
        assert result.shape == (10,)

    def test_batch_tensor_ior(self):
        """Batched tensor ior works correctly."""
        cosine = torch.rand(10)
        ior = torch.rand(10) + 1.0  # ior > 1
        result = schlick_reflectance(cosine, ior=ior)
        assert result.shape == (10,)


class TestSchlickReflectanceKnownValues:
    """Tests for known reference values."""

    def test_normal_incidence(self):
        """At normal incidence (cosine=1), reflectance = r0."""
        # r0 = ((1 - ior) / (1 + ior))^2
        # For ior=1.5: r0 = ((1-1.5)/(1+1.5))^2 = (-0.5/2.5)^2 = 0.04
        cosine = torch.tensor(1.0, dtype=torch.float64)
        result = schlick_reflectance(cosine, ior=1.5)
        expected_r0 = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        torch.testing.assert_close(
            result,
            torch.tensor(expected_r0, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_grazing_angle(self):
        """At grazing angle (cosine=0), reflectance = 1."""
        # reflectance = r0 + (1 - r0) * (1 - 0)^5 = r0 + (1 - r0) = 1
        cosine = torch.tensor(0.0, dtype=torch.float64)
        result = schlick_reflectance(cosine, ior=1.5)
        torch.testing.assert_close(
            result,
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_glass_r0(self):
        """Glass with ior=1.5 has r0=0.04."""
        # r0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04
        cosine = torch.tensor(1.0, dtype=torch.float64)
        result = schlick_reflectance(cosine, ior=1.5)
        expected = 0.04
        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_diamond_r0(self):
        """Diamond with ior=2.42 has r0 ~= 0.17."""
        # r0 = ((1 - 2.42) / (1 + 2.42))^2 = (-1.42/3.42)^2 ~= 0.1724
        cosine = torch.tensor(1.0, dtype=torch.float64)
        ior = 2.42
        expected_r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        result = schlick_reflectance(cosine, ior=ior)
        torch.testing.assert_close(
            result,
            torch.tensor(expected_r0, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_intermediate_value(self):
        """Test intermediate cosine value."""
        # For cosine=0.5, ior=1.5:
        # r0 = 0.04
        # reflectance = 0.04 + 0.96 * (1 - 0.5)^5 = 0.04 + 0.96 * 0.03125 = 0.04 + 0.03 = 0.07
        cosine = torch.tensor(0.5, dtype=torch.float64)
        ior = 1.5
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        expected = r0 + (1.0 - r0) * (1.0 - 0.5) ** 5
        result = schlick_reflectance(cosine, ior=ior)
        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )


class TestSchlickReflectanceGradient:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        cosine = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def func(c):
            return schlick_reflectance(c, ior=1.5)

        assert gradcheck(func, (cosine,), raise_exception=True)

    def test_gradcheck_tensor_ior(self):
        """Passes gradcheck with tensor ior (no grad for ior)."""
        cosine = torch.tensor(
            [0.3, 0.6, 0.9], dtype=torch.float64, requires_grad=True
        )
        ior = torch.tensor(1.5, dtype=torch.float64)  # No grad for ior

        def func(c):
            return schlick_reflectance(c, ior=ior)

        assert gradcheck(func, (cosine,), raise_exception=True)

    def test_gradient_sign(self):
        """Gradient should be negative (reflectance decreases as cosine increases)."""
        # d(reflectance)/d(cosine) = -5 * (1 - r0) * (1 - cosine)^4
        # For cosine in [0, 1), (1 - r0) > 0 and (1 - cosine)^4 >= 0
        # So the gradient should be <= 0
        cosine = torch.tensor(
            [0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64, requires_grad=True
        )
        result = schlick_reflectance(cosine, ior=1.5)
        result.sum().backward()

        # All gradients should be non-positive (and strictly negative for cosine < 1)
        assert (cosine.grad <= 0).all(), (
            f"Expected non-positive gradients, got {cosine.grad}"
        )
        # For cosine < 1, gradient should be strictly negative
        assert (cosine.grad[:-1] < 0).all(), (
            "Expected strictly negative gradients for cosine < 1"
        )

    def test_gradient_at_grazing(self):
        """Gradient at cosine=0 should be -5 * (1 - r0)."""
        # At cosine=0: d/dc = -5 * (1 - r0) * 1 = -5 * (1 - r0)
        cosine = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        ior = 1.5
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        expected_grad = -5.0 * (1.0 - r0)

        result = schlick_reflectance(cosine, ior=ior)
        result.backward()

        torch.testing.assert_close(
            cosine.grad,
            torch.tensor(expected_grad, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )


class TestSchlickReflectanceDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        cosine = torch.rand(10, dtype=torch.float32)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        cosine = torch.rand(10, dtype=torch.float64)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        cosine = torch.rand(10, dtype=torch.bfloat16)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        cosine = torch.rand(10, dtype=torch.float16)
        result = schlick_reflectance(cosine, ior=1.5)
        assert result.dtype == torch.float16


class TestSchlickReflectanceProperties:
    """Tests for mathematical properties."""

    def test_output_range(self):
        """Output is in [r0, 1] for valid cosine inputs."""
        cosine = torch.linspace(0, 1, 100, dtype=torch.float64)
        ior = 1.5
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        result = schlick_reflectance(cosine, ior=ior)

        assert (result >= r0 - 1e-6).all(), (
            f"Output below r0: min={result.min()}"
        )
        assert (result <= 1.0 + 1e-6).all(), (
            f"Output above 1: max={result.max()}"
        )

    def test_monotonically_decreasing(self):
        """Reflectance is monotonically decreasing in cosine."""
        cosine = torch.linspace(0, 1, 100, dtype=torch.float64)
        result = schlick_reflectance(cosine, ior=1.5)

        # Each successive value should be <= the previous
        diffs = result[1:] - result[:-1]
        assert (diffs <= 1e-10).all(), (
            "Reflectance should be monotonically decreasing"
        )

    def test_ior_effect(self):
        """Higher ior gives higher r0 (hence higher reflectance at normal incidence)."""
        cosine = torch.tensor(1.0, dtype=torch.float64)

        result_glass = schlick_reflectance(cosine, ior=1.5)  # r0 = 0.04
        result_diamond = schlick_reflectance(cosine, ior=2.42)  # r0 ~= 0.17

        assert result_diamond > result_glass
