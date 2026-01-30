import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestKelvinKei:
    """Tests for the Kelvin function kei(x)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: kei(0) = -pi/4, kei(nan) = nan."""
        result_zero = torchscience.special_functions.kelvin_kei(
            torch.tensor(0.0)
        ).item()
        expected_zero = -torch.pi / 4
        assert result_zero == pytest.approx(expected_zero, rel=1e-6)

        assert torchscience.special_functions.kelvin_kei(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |x| <= 5."""
        x = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.kelvin_kei(x)
        expected = torch.tensor(
            [scipy.special.kei(val.item()) for val in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-8)

    def test_scipy_agreement_medium(self):
        """Test agreement with scipy for medium |x| (5 < |x| <= 10)."""
        x = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        result = torchscience.special_functions.kelvin_kei(x)
        expected = torch.tensor(
            [scipy.special.kei(val.item()) for val in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for larger |x| > 10."""
        x = torch.tensor([10.0, 12.0, 15.0], dtype=torch.float64)
        result = torchscience.special_functions.kelvin_kei(x)
        expected = torch.tensor(
            [scipy.special.kei(val.item()) for val in x], dtype=torch.float64
        )
        # Relaxed tolerance for large values (asymptotic expansion)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-5)

    def test_symmetry_even_function(self):
        """kei(-x) = kei(x) (even function)."""
        x = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 7.0],
            dtype=torch.float64,
        )
        result_pos = torchscience.special_functions.kelvin_kei(x)
        result_neg = torchscience.special_functions.kelvin_kei(-x)
        torch.testing.assert_close(
            result_pos, result_neg, rtol=1e-12, atol=1e-12
        )

    def test_known_values(self):
        """Test known reference values from scipy."""
        # Reference values from scipy.special.kei
        test_cases = [
            (0.0, -0.7853981633974483),  # -pi/4
            (0.5, -0.6715816950943676),
            (1.0, -0.49499463651872),
            (2.0, -0.20240006776470432),
            (5.0, 0.01118758650986929),
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor(x_val, dtype=torch.float64)
            result = torchscience.special_functions.kelvin_kei(x)
            assert result.item() == pytest.approx(expected_val, rel=1e-6)

    def test_large_x_decay(self):
        """Test that kei(x) -> 0 as x -> infinity (exponential decay)."""
        x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        result = torchscience.special_functions.kelvin_kei(x)
        # Values should be very small for large x
        assert torch.all(torch.abs(result) < 0.01)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Use values where the function is well-behaved (away from singularity)
        x = torch.tensor(
            [0.5, 1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.kelvin_kei, x, rtol=1e-4, atol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.kelvin_kei, x, rtol=1e-3, atol=1e-3
        )

    def test_gradient_at_zero(self):
        """Verify gradient at x=0 is 0 (from the derivative formula)."""
        x = torch.tensor([0.01], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.kelvin_kei(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
        # kei'(0) = 0, so gradient near zero should be small
        assert torch.abs(grad).item() < 0.1

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        x = torch.randn(10, dtype=torch.complex128)
        x = x + 0.5  # Shift away from origin
        result = torchscience.special_functions.kelvin_kei(x)
        assert result.dtype == torch.complex128

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real kei."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.kelvin_kei(x_real)
        result_complex = torchscience.special_functions.kelvin_kei(x_complex)
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-8, atol=1e-8
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-8,
            atol=1e-8,
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.randn(10, device="meta")
        result = torchscience.special_functions.kelvin_kei(x)
        assert result.shape == x.shape
        assert result.device == x.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        x = (
            torch.randn(10, dtype=torch.float32) + 1.0
        )  # Shift away from origin
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.kelvin_kei(x)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.kelvin_kei(x)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        x = (
            torch.randn(5, 10, dtype=torch.float64) + 1.0
        )  # Shift away from origin
        result = torch.vmap(torchscience.special_functions.kelvin_kei)(x)
        expected = torchscience.special_functions.kelvin_kei(x)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.kelvin_kei)
        x = (
            torch.randn(100, dtype=torch.float64) + 1.0
        )  # Shift away from origin
        result = compiled_fn(x)
        expected = torchscience.special_functions.kelvin_kei(x)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.kelvin_kei)
        # Create leaf tensor shifted away from origin
        x = (torch.randn(100, dtype=torch.float64) + 1.0).requires_grad_(True)
        result = compiled_fn(x)
        result.sum().backward()
        assert x.grad is not None
        # Verify gradient matches uncompiled version
        x2 = x.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.kelvin_kei(x2)
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        x1 = torch.randn(3, 1, dtype=torch.float64) + 1.0
        x2 = torch.randn(1, 4, dtype=torch.float64) + 1.0
        result = torchscience.special_functions.kelvin_kei(x1 + x2)
        assert result.shape == (3, 4)

    # =========================================================================
    # dtype tests
    # =========================================================================

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=dtype)
        result = torchscience.special_functions.kelvin_kei(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        x = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.kelvin_kei(x)
        assert result.dtype == dtype
