import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestSphericalBesselY1:
    """Tests for the spherical Bessel function y_1."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: y_1(0) = -inf, y_1(nan) = nan."""
        assert torchscience.special_functions.spherical_bessel_y_1(
            torch.tensor(0.0)
        ).item() == float("-inf")
        assert torchscience.special_functions.spherical_bessel_y_1(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z|."""
        # Start from 0.1 to avoid singularity at 0
        z = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        expected = torch.tensor(
            [scipy.special.spherical_yn(1, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z|."""
        z = torch.tensor([5.0, 10.0, 20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        expected = torch.tensor(
            [scipy.special.spherical_yn(1, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_closed_form(self):
        """Test y_1(z) = -cos(z)/z^2 - sin(z)/z for non-zero z."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        expected = -torch.cos(z) / (z * z) - torch.sin(z) / z
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_symmetry_even_function(self):
        """Test y_1(-z) = y_1(z) (even function)."""
        z_pos = torch.tensor([0.5, 1.0, 5.0, 10.0], dtype=torch.float64)
        z_neg = -z_pos
        result_pos = torchscience.special_functions.spherical_bessel_y_1(z_pos)
        result_neg = torchscience.special_functions.spherical_bessel_y_1(z_neg)
        torch.testing.assert_close(
            result_neg, result_pos, rtol=1e-12, atol=1e-12
        )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Use values away from zero to avoid singularity
        z = torch.tensor(
            [0.5, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.spherical_bessel_y_1, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        # Use values away from zero to avoid singularity
        z = torch.tensor(
            [0.5, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.spherical_bessel_y_1, z
        )

    def test_gradient_formula(self):
        """Verify d/dz y_1(z) = y_0(z) - 2*y_1(z)/z numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.spherical_bessel_y_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detached = z.detach()
        y0 = torchscience.special_functions.spherical_bessel_y_0(z_detached)
        y1 = torchscience.special_functions.spherical_bessel_y_1(z_detached)
        expected = y0 - 2 * y1 / z_detached
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=torch.complex128)
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        assert result.dtype == torch.complex128

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, 5.0 + 0.2j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_y_1(
            z_near_real
        )
        expected = torch.tensor(
            [scipy.special.spherical_yn(1, z.item()) for z in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_closed_form(self):
        """Test y_1(z) = -cos(z)/z^2 - sin(z)/z for complex z."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 3.0 + 1.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        expected = -torch.cos(z) / (z * z) - torch.sin(z) / z
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real y_1."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.spherical_bessel_y_1(
            x_real
        )
        result_complex = torchscience.special_functions.spherical_bessel_y_1(
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

    def test_complex_even_symmetry(self):
        """Test y_1(-z) = y_1(z) for complex z (even function)."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 0.5 + 2.0j], dtype=torch.complex128
        )
        result_pos = torchscience.special_functions.spherical_bessel_y_1(z)
        result_neg = torchscience.special_functions.spherical_bessel_y_1(-z)
        torch.testing.assert_close(
            result_neg, result_pos, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.spherical_bessel_y_1(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.spherical_bessel_y_1(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64
        )
        result = torch.vmap(
            torchscience.special_functions.spherical_bessel_y_1
        )(z)
        expected = torchscience.special_functions.spherical_bessel_y_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_y_1
        )
        z = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = compiled_fn(z)
        expected = torchscience.special_functions.spherical_bessel_y_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_y_1
        )
        z = torch.tensor(
            [1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.spherical_bessel_y_1(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        z2 = torch.tensor([[0.5, 1.5, 2.5, 3.5]], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_y_1(z1 + z2)
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
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_y_1(z)
        assert result.dtype == dtype
