import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestModifiedBesselI1:
    """Tests for the modified Bessel function I₁."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: I₁(0) = 0, I₁(inf) = inf, I₁(nan) = nan."""
        assert torchscience.special_functions.modified_bessel_i_1(
            torch.tensor(0.0)
        ).item() == pytest.approx(0.0)
        assert torchscience.special_functions.modified_bessel_i_1(
            torch.tensor(float("inf"))
        ).isinf()
        assert torchscience.special_functions.modified_bessel_i_1(
            torch.tensor(float("-inf"))
        ).isinf()
        assert torchscience.special_functions.modified_bessel_i_1(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z| <= 8."""
        z = torch.tensor(
            [0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 7.9], dtype=torch.float64
        )
        result = torchscience.special_functions.modified_bessel_i_1(z)
        expected = torch.tensor(
            [scipy.special.i1(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z| > 8."""
        z = torch.tensor([8.1, 10.0, 20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_i_1(z)
        expected = torch.tensor(
            [scipy.special.i1(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_region_boundary(self):
        """Test accuracy near the |z|=8 boundary where approximations switch."""
        z = torch.tensor(
            [7.9, 7.99, 7.999, 8.0, 8.001, 8.01, 8.1], dtype=torch.float64
        )
        result = torchscience.special_functions.modified_bessel_i_1(z)
        expected = torch.tensor(
            [scipy.special.i1(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry_odd_function(self):
        """I₁(-z) = -I₁(z) (odd function)."""
        z = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 4.0, 7.9, 8.1, 15.0, 50.0],
            dtype=torch.float64,
        )
        result_pos = torchscience.special_functions.modified_bessel_i_1(z)
        result_neg = torchscience.special_functions.modified_bessel_i_1(-z)
        torch.testing.assert_close(
            result_pos, -result_neg, rtol=1e-12, atol=1e-12
        )

    def test_negative_real(self):
        """Test I₁ for negative real arguments (odd function)."""
        z_pos = torch.tensor([0.1, 1.0, 5.0, 10.0, 50.0], dtype=torch.float64)
        z_neg = -z_pos
        result_pos = torchscience.special_functions.modified_bessel_i_1(z_pos)
        result_neg = torchscience.special_functions.modified_bessel_i_1(z_neg)
        torch.testing.assert_close(
            result_pos, -result_neg, rtol=1e-12, atol=1e-12
        )

    def test_recurrence_relation(self):
        """Test 2*I₁'(z) = I₀(z) + I₂(z) using scipy for I₂."""
        # This validates the relationship between I₀ and I₁ through derivatives
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        i0 = torchscience.special_functions.modified_bessel_i_0(z)
        i1 = torchscience.special_functions.modified_bessel_i_1(z)
        # I₁'(z) = I₀(z) - I₁(z)/z
        i1_prime = i0 - i1 / z
        # Using scipy for I₂ to verify: 2*I₁' = I₀ + I₂
        i2_scipy = torch.tensor(
            [scipy.special.iv(2, x.item()) for x in z], dtype=torch.float64
        )
        lhs = 2 * i1_prime
        rhs = i0 + i2_scipy
        torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Avoid z=0 where gradient has a removable singularity
        z_pos = torch.rand(5, dtype=torch.float64) + 0.1
        z = torch.cat([z_pos, -z_pos])
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            torchscience.special_functions.modified_bessel_i_1, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        # Avoid z=0 where second derivative has a removable singularity
        z_pos = torch.rand(5, dtype=torch.float64) + 0.1
        z = torch.cat([z_pos, -z_pos])
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.modified_bessel_i_1, z
        )

    def test_gradient_identity(self):
        """Verify d/dz I₁(z) = I₀(z) - I₁(z)/z numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.modified_bessel_i_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detach = z.detach()
        expected = (
            torchscience.special_functions.modified_bessel_i_0(z_detach)
            - torchscience.special_functions.modified_bessel_i_1(z_detach)
            / z_detach
        )
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    def test_gradient_at_zero_limit(self):
        """Test gradient at z=0 uses correct limit (I₁'(0) = 0.5)."""
        z = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.modified_bessel_i_1(z)
        y.backward()
        # I₁'(0) = I₀(0) - lim[I₁(z)/z] = 1 - 0.5 = 0.5
        torch.testing.assert_close(
            z.grad,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        result = torchscience.special_functions.modified_bessel_i_1(z)
        assert result.dtype == torch.complex128

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, 5.0 + 0.2j], dtype=torch.complex128
        )
        result = torchscience.special_functions.modified_bessel_i_1(
            z_near_real
        )
        expected = torch.tensor(
            [scipy.special.iv(1, z.item()) for z in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_farther_from_real(self):
        """Test complex accuracy farther from real axis (relaxed tolerance)."""
        z_far = torch.tensor(
            [1.0 + 1.0j, 2.0 + 2.0j, 3.0 - 3.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.modified_bessel_i_1(z_far)
        expected = torch.tensor(
            [scipy.special.iv(1, z.item()) for z in z_far],
            dtype=torch.complex128,
        )
        # Relaxed tolerance for far-from-real complex values
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real I₁."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.modified_bessel_i_1(
            x_real
        )
        result_complex = torchscience.special_functions.modified_bessel_i_1(
            x_complex
        )
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.modified_bessel_i_1(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.randn(10, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.modified_bessel_i_1(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.modified_bessel_i_1(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.randn(5, 10, dtype=torch.float64)
        result = torch.vmap(
            torchscience.special_functions.modified_bessel_i_1
        )(z)
        expected = torchscience.special_functions.modified_bessel_i_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_i_1
        )
        z = torch.randn(100, dtype=torch.float64)
        result = compiled_fn(z)
        expected = torchscience.special_functions.modified_bessel_i_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_i_1
        )
        # Add 0.1 before requires_grad to keep z as a leaf tensor
        z = (torch.randn(100, dtype=torch.float64) + 0.1).requires_grad_(True)
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.modified_bessel_i_1(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.randn(3, 1, dtype=torch.float64)
        z2 = torch.randn(1, 4, dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_i_1(z1 + z2)
        assert result.shape == (3, 4)

    # =========================================================================
    # dtype tests
    # =========================================================================

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.modified_bessel_i_1(z)
        assert result.dtype == dtype
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.modified_bessel_i_1(z)
        assert result.dtype == dtype
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()
