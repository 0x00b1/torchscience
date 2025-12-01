import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestSphericalBesselJ1:
    """Tests for the spherical Bessel function j_1."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: j_1(0) = 0, j_1(nan) = nan."""
        assert torchscience.special_functions.spherical_bessel_j_1(
            torch.tensor(0.0)
        ).item() == pytest.approx(0.0)
        assert torchscience.special_functions.spherical_bessel_j_1(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z|."""
        z = torch.tensor(
            [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64
        )
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        expected = torch.tensor(
            [scipy.special.spherical_jn(1, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z|."""
        z = torch.tensor([5.0, 10.0, 20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        expected = torch.tensor(
            [scipy.special.spherical_jn(1, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_closed_form(self):
        """Test j_1(z) = sin(z)/z^2 - cos(z)/z for non-zero z."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        expected = torch.sin(z) / (z * z) - torch.cos(z) / z
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_symmetry_odd_function(self):
        """Test j_1(-z) = -j_1(z) (odd function)."""
        z_pos = torch.tensor([0.1, 1.0, 5.0, 10.0], dtype=torch.float64)
        z_neg = -z_pos
        result_pos = torchscience.special_functions.spherical_bessel_j_1(z_pos)
        result_neg = torchscience.special_functions.spherical_bessel_j_1(z_neg)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-12, atol=1e-12
        )

    def test_taylor_series_near_zero(self):
        """Test Taylor series accuracy near z=0: j_1(z) = z/3 - z^3/30 + z^5/840."""
        # Use values within Taylor series range (eps=1e-12 for float64)
        z = torch.tensor([1e-14, 1e-13], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        z2 = z * z
        expected = z / 3 - z * z2 / 30 + z * z2 * z2 / 840
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-30)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        z = torch.randn(10, dtype=torch.float64, requires_grad=True)
        # Avoid values too close to zero
        z = z + torch.sign(z) * 0.5
        assert torch.autograd.gradcheck(
            torchscience.special_functions.spherical_bessel_j_1, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        z = torch.randn(10, dtype=torch.float64, requires_grad=True)
        # Avoid values too close to zero
        z = z + torch.sign(z) * 0.5
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.spherical_bessel_j_1, z
        )

    def test_gradient_formula(self):
        """Verify d/dz j_1(z) = j_0(z) - 2*j_1(z)/z numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.spherical_bessel_j_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detached = z.detach()
        j0 = torchscience.special_functions.spherical_bessel_j_0(z_detached)
        j1 = torchscience.special_functions.spherical_bessel_j_1(z_detached)
        expected = j0 - 2 * j1 / z_detached
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    def test_gradient_at_origin_limit(self):
        """Test gradient near z=0 approaches 1/3."""
        # Use z within Taylor series range (eps=1e-12 for float64)
        z = torch.tensor([1e-14], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.spherical_bessel_j_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        # d/dz j_1(z) at small z is approximately 1/3
        expected = torch.tensor([1.0 / 3.0], dtype=torch.float64)
        torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-10)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        assert result.dtype == torch.complex128

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, 5.0 + 0.2j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_j_1(
            z_near_real
        )
        expected = torch.tensor(
            [scipy.special.spherical_jn(1, z.item()) for z in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_closed_form(self):
        """Test j_1(z) = sin(z)/z^2 - cos(z)/z for complex z."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 3.0 + 1.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        expected = torch.sin(z) / (z * z) - torch.cos(z) / z
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real j_1."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.spherical_bessel_j_1(
            x_real
        )
        result_complex = torchscience.special_functions.spherical_bessel_j_1(
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

    def test_complex_odd_symmetry(self):
        """Test j_1(-z) = -j_1(z) for complex z (odd function)."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 0.5 + 2.0j], dtype=torch.complex128
        )
        result_pos = torchscience.special_functions.spherical_bessel_j_1(z)
        result_neg = torchscience.special_functions.spherical_bessel_j_1(-z)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.randn(10, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.spherical_bessel_j_1(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.spherical_bessel_j_1(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.randn(5, 10, dtype=torch.float64)
        result = torch.vmap(
            torchscience.special_functions.spherical_bessel_j_1
        )(z)
        expected = torchscience.special_functions.spherical_bessel_j_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_j_1
        )
        z = torch.randn(100, dtype=torch.float64)
        result = compiled_fn(z)
        expected = torchscience.special_functions.spherical_bessel_j_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_j_1
        )
        z = torch.randn(100, dtype=torch.float64, requires_grad=True)
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.spherical_bessel_j_1(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.randn(3, 1, dtype=torch.float64)
        z2 = torch.randn(1, 4, dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_j_1(z1 + z2)
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
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_j_1(z)
        assert result.dtype == dtype
