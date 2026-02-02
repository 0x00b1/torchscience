import mpmath
import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestStruveH1:
    """Tests for the Struve function H_1."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: H_1(0) = 0, H_1(nan) = nan."""
        # H_1(0) = 0
        assert torchscience.special_functions.struve_h_1(
            torch.tensor(0.0)
        ).item() == pytest.approx(0.0, abs=1e-10)

        # H_1(nan) = nan
        assert torchscience.special_functions.struve_h_1(
            torch.tensor(float("nan"))
        ).isnan()

    def test_moderate_large_values(self):
        """Test for moderately large |z| values where series converges.

        Note: The power series implementation has accuracy limitations for
        very large |z| (> 20). We test up to z=20 where the series converges well.
        """
        z_large = torch.tensor([15.0, 18.0, 20.0], dtype=torch.float64)
        result = torchscience.special_functions.struve_h_1(z_large)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z_large],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z| <= 5."""
        z = torch.tensor(
            [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.9], dtype=torch.float64
        )
        result = torchscience.special_functions.struve_h_1(z)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z| > 5.

        Note: The power series implementation has accuracy limitations for
        very large |z| (> 20). We test up to z=20 where the series converges well.
        """
        z = torch.tensor([5.1, 10.0, 15.0, 20.0], dtype=torch.float64)
        result = torchscience.special_functions.struve_h_1(z)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z], dtype=torch.float64
        )
        # Relaxed tolerance for larger values
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_region_boundary(self):
        """Test accuracy near region boundaries where approximations may switch."""
        z = torch.tensor(
            [4.9, 4.99, 4.999, 5.0, 5.001, 5.01, 5.1], dtype=torch.float64
        )
        result = torchscience.special_functions.struve_h_1(z)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_symmetry_even_function(self):
        """H_1(-z) = H_1(z) (even function)."""
        z = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 4.0, 4.9, 5.1, 15.0, 20.0],
            dtype=torch.float64,
        )
        result_pos = torchscience.special_functions.struve_h_1(z)
        result_neg = torchscience.special_functions.struve_h_1(-z)
        torch.testing.assert_close(
            result_pos, result_neg, rtol=1e-10, atol=1e-10
        )

    def test_negative_real(self):
        """Test H_1 for negative real arguments (even function)."""
        z_pos = torch.tensor([0.1, 1.0, 5.0, 10.0, 20.0], dtype=torch.float64)
        z_neg = -z_pos
        result_pos = torchscience.special_functions.struve_h_1(z_pos)
        result_neg = torchscience.special_functions.struve_h_1(z_neg)
        torch.testing.assert_close(
            result_pos, result_neg, rtol=1e-10, atol=1e-10
        )

    def test_specific_values(self):
        """Test specific values against scipy reference."""
        test_cases = [
            (0.5, scipy.special.struve(1, 0.5)),
            (1.0, scipy.special.struve(1, 1.0)),
            (2.0, scipy.special.struve(1, 2.0)),
            (5.0, scipy.special.struve(1, 5.0)),
            (10.0, scipy.special.struve(1, 10.0)),
        ]
        for z_val, expected_val in test_cases:
            z = torch.tensor(z_val, dtype=torch.float64)
            result = torchscience.special_functions.struve_h_1(z)
            assert result.item() == pytest.approx(
                expected_val, rel=1e-6, abs=1e-8
            )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Avoid z=0 where gradient has special behavior
        z = torch.randn(10, dtype=torch.float64, requires_grad=True)
        z = z + torch.sign(z) * 0.1  # Push away from zero
        assert torch.autograd.gradcheck(
            torchscience.special_functions.struve_h_1, z, eps=1e-6
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        # Avoid z=0 where gradient has special behavior
        z = torch.randn(10, dtype=torch.float64, requires_grad=True)
        z = z + torch.sign(z) * 0.1  # Push away from zero
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.struve_h_1, z, eps=1e-6
        )

    def test_gradient_identity(self):
        """Verify d/dz H_1(z) = H_0(z) - H_1(z)/z numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.struve_h_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        # Expected: d/dz H_1(z) = H_0(z) - H_1(z)/z
        z_detach = z.detach()
        h0 = torchscience.special_functions.struve_h_0(z_detach)
        h1 = torchscience.special_functions.struve_h_1(z_detach)
        expected = h0 - h1 / z_detach
        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-6)

    def test_gradient_at_small_z(self):
        """Test gradient near z=0 (but not at z=0)."""
        z = torch.tensor(
            [0.05, 0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.struve_h_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        # Expected: d/dz H_1(z) = H_0(z) - H_1(z)/z
        z_detach = z.detach()
        h0 = torchscience.special_functions.struve_h_0(z_detach)
        h1 = torchscience.special_functions.struve_h_1(z_detach)
        expected = h0 - h1 / z_detach
        torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-5)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        result = torchscience.special_functions.struve_h_1(z)
        assert result.dtype == torch.complex128

    def test_complex_mpmath_agreement(self):
        """Validate complex accuracy against mpmath.struveh."""
        z_values = [1.0 + 0.5j, 2.0 - 0.3j, 0.5 + 0.5j, 3.0 + 1.0j]
        z = torch.tensor(z_values, dtype=torch.complex128)
        result = torchscience.special_functions.struve_h_1(z)

        expected_values = [complex(mpmath.struveh(1, zv)) for zv in z_values]
        expected = torch.tensor(expected_values, dtype=torch.complex128)

        # Relaxed tolerance for complex values
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against mpmath near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, 5.0 + 0.2j], dtype=torch.complex128
        )
        result = torchscience.special_functions.struve_h_1(z_near_real)
        expected = torch.tensor(
            [complex(mpmath.struveh(1, z.item())) for z in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_complex_farther_from_real(self):
        """Test complex accuracy farther from real axis (relaxed tolerance)."""
        z_far = torch.tensor(
            [1.0 + 1.0j, 2.0 + 2.0j, 3.0 - 3.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.struve_h_1(z_far)
        expected = torch.tensor(
            [complex(mpmath.struveh(1, z.item())) for z in z_far],
            dtype=torch.complex128,
        )
        # More relaxed tolerance for far-from-real complex values
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real H_1."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.struve_h_1(x_real)
        result_complex = torchscience.special_functions.struve_h_1(x_complex)
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
        """Test H_1(-z) = H_1(z) for complex arguments."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 0.5 + 1.0j], dtype=torch.complex128
        )
        result_pos = torchscience.special_functions.struve_h_1(z)
        result_neg = torchscience.special_functions.struve_h_1(-z)
        torch.testing.assert_close(
            result_pos, result_neg, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.struve_h_1(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_meta_tensor_2d(self):
        """Test meta tensor shape inference for 2D tensors."""
        z = torch.randn(5, 10, device="meta")
        result = torchscience.special_functions.struve_h_1(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.randn(10, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.struve_h_1(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.struve_h_1(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.randn(5, 10, dtype=torch.float64)
        result = torch.vmap(torchscience.special_functions.struve_h_1)(z)
        expected = torchscience.special_functions.struve_h_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.struve_h_1)
        z = torch.randn(100, dtype=torch.float64)
        result = compiled_fn(z)
        expected = torchscience.special_functions.struve_h_1(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.struve_h_1)
        z = torch.randn(100, dtype=torch.float64, requires_grad=True)
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.struve_h_1(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.randn(3, 1, dtype=torch.float64)
        z2 = torch.randn(1, 4, dtype=torch.float64)
        result = torchscience.special_functions.struve_h_1(z1 + z2)
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
        result = torchscience.special_functions.struve_h_1(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.struve_h_1(z)
        assert result.dtype == dtype

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_values(self):
        """Test behavior for very small |z|."""
        z = torch.tensor([1e-10, 1e-8, 1e-6, 1e-4], dtype=torch.float64)
        result = torchscience.special_functions.struve_h_1(z)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_moderate_values(self):
        """Test a range of moderate values."""
        z = torch.linspace(0.1, 10.0, 50, dtype=torch.float64)
        result = torchscience.special_functions.struve_h_1(z)
        expected = torch.tensor(
            [scipy.special.struve(1, x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_gradient_formula_small_z(self):
        """Test gradient formula d/dz H_1(z) = H_0(z) - H_1(z)/z for small z."""
        # Test the gradient formula holds for small z
        z = torch.tensor(
            [0.1, 0.2, 0.3, 0.5], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.struve_h_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        # Expected: d/dz H_1(z) = H_0(z) - H_1(z)/z
        z_detach = z.detach()
        h0 = torchscience.special_functions.struve_h_0(z_detach)
        h1 = torchscience.special_functions.struve_h_1(z_detach)
        expected = h0 - h1 / z_detach
        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-6)
