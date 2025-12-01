import math

import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestSphericalBesselK0:
    """Tests for the modified spherical Bessel function k_0."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: k_0(nan) = nan."""
        assert torchscience.special_functions.spherical_bessel_k_0(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement(self):
        """Test agreement with scipy for positive z."""
        z = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.spherical_kn(0, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z|."""
        z = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.spherical_kn(0, x.item()) for x in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_closed_form_identity(self):
        """Test k_0(z) = (pi/2z) * exp(-z) for positive z."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        expected = (math.pi / (2 * z)) * torch.exp(-z)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_positive_for_positive_z(self):
        """Test k_0(z) > 0 for positive real z."""
        z = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        assert (result > 0).all()

    def test_monotonically_decreasing_positive(self):
        """Test that k_0(z) is monotonically decreasing for positive z."""
        z = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        diffs = result[1:] - result[:-1]
        assert (diffs < 0).all()

    def test_exponential_decay(self):
        """Test exponential decay for large z."""
        z = torch.tensor([5.0, 10.0, 20.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        # k_0(z) ~ (pi/2z) * exp(-z) decays exponentially
        ratios = result[:-1] / result[1:]
        # For large z, ratio should be approximately exp(z_{i+1} - z_i) * z_{i+1}/z_i
        for i in range(len(ratios)):
            z1, z2 = z[i].item(), z[i + 1].item()
            expected_ratio = math.exp(z2 - z1) * z2 / z1
            assert ratios[i].item() == pytest.approx(expected_ratio, rel=0.01)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        z = torch.rand(10, dtype=torch.float64) * 4 + 0.5  # z in [0.5, 4.5]
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            torchscience.special_functions.spherical_bessel_k_0, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        z = torch.rand(10, dtype=torch.float64) * 4 + 0.5  # z in [0.5, 4.5]
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.spherical_bessel_k_0, z
        )

    def test_gradient_formula(self):
        """Verify d/dz k_0(z) = -(pi/2z^2)(1+z)e^(-z) numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.spherical_bessel_k_0(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detached = z.detach()
        pi_over_2 = math.pi / 2
        expected = (
            -pi_over_2
            * torch.exp(-z_detached)
            * (1 + z_detached)
            / (z_detached**2)
        )
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.2j, 3.0 + 0.5j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        assert result.dtype == torch.complex128

    def test_complex_closed_form(self):
        """Test k_0(z) = (pi/2z) * exp(-z) for complex z."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 - 0.3j, 3.0 + 1.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        expected = (math.pi / (2 * z)) * torch.exp(-z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real k_0."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.spherical_bessel_k_0(
            x_real
        )
        result_complex = torchscience.special_functions.spherical_bessel_k_0(
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
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.rand(10, dtype=torch.float32) * 4 + 0.5  # avoid pole at 0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.spherical_bessel_k_0(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.spherical_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.rand(5, 10, dtype=torch.float64) * 4 + 0.5
        result = torch.vmap(
            torchscience.special_functions.spherical_bessel_k_0
        )(z)
        expected = torchscience.special_functions.spherical_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_k_0
        )
        z = torch.rand(100, dtype=torch.float64) * 4 + 0.5
        result = compiled_fn(z)
        expected = torchscience.special_functions.spherical_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_k_0
        )
        z = torch.rand(100, dtype=torch.float64) * 4 + 0.5
        z = z.clone().requires_grad_(True)
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.spherical_bessel_k_0(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.rand(3, 1, dtype=torch.float64) * 4 + 0.5
        z2 = torch.rand(1, 4, dtype=torch.float64) * 4 + 0.5
        result = torchscience.special_functions.spherical_bessel_k_0(z1 + z2)
        assert result.shape == (3, 4)

    # =========================================================================
    # dtype tests
    # =========================================================================

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        z = torch.tensor([0.5, 1.0, 2.0], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_k_0(z)
        assert result.dtype == dtype
