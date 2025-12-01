import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestModifiedBesselK0:
    """Tests for the modified Bessel function of the second kind K_0."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: K_0(0) = +inf, K_0(inf) = 0, K_0(nan) = nan."""
        # K_0(0) = +infinity (logarithmic singularity)
        assert torchscience.special_functions.modified_bessel_k_0(
            torch.tensor(0.0)
        ).item() == float("inf")
        # K_0(+inf) = 0 (exponential decay)
        assert torchscience.special_functions.modified_bessel_k_0(
            torch.tensor(float("inf"))
        ).item() == pytest.approx(0.0, abs=1e-10)
        # K_0(nan) = nan
        assert torchscience.special_functions.modified_bessel_k_0(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z| <= 2."""
        z = torch.tensor([0.1, 0.5, 1.0, 1.5, 1.9], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.k0(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z| > 2."""
        z = torch.tensor(
            [2.1, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=torch.float64
        )
        result = torchscience.special_functions.modified_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.k0(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_region_boundary(self):
        """Test accuracy near the |z|=2 boundary where approximations switch."""
        z = torch.tensor(
            [1.9, 1.99, 1.999, 2.0, 2.001, 2.01, 2.1], dtype=torch.float64
        )
        result = torchscience.special_functions.modified_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.k0(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_real_returns_nan(self):
        """K_0(z) for negative real z should return NaN (not defined)."""
        z = torch.tensor([-0.1, -1.0, -5.0, -10.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert result.isnan().all()

    def test_near_zero_singularity(self):
        """Test behavior near z=0 singularity: K_0(z) ~ -ln(z/2) - gamma for small z."""
        z = torch.tensor([0.01, 0.001, 0.0001], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.k0(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_always_positive(self):
        """K_0(z) > 0 for all z > 0."""
        z = torch.tensor(
            [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert (result > 0).all()

    def test_exponential_decay(self):
        """Test exponential decay at large z: K_0(z) ~ sqrt(pi/2z) * exp(-z)."""
        z = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k_0(z)
        expected = torch.tensor(
            [scipy.special.k0(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Avoid z=0 where K_0 has a singularity
        z = (
            torch.rand(10, dtype=torch.float64) * 4 + 0.5
        )  # values in [0.5, 4.5]
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            torchscience.special_functions.modified_bessel_k_0, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        # Avoid z=0 where K_0 has a singularity
        z = (
            torch.rand(10, dtype=torch.float64) * 4 + 0.5
        )  # values in [0.5, 4.5]
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.modified_bessel_k_0, z
        )

    def test_gradient_identity(self):
        """Verify d/dz K_0(z) = -K_1(z) numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.modified_bessel_k_0(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detach = z.detach()
        expected = -torchscience.special_functions.modified_bessel_k_1(
            z_detach
        )
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        # Ensure positive real part to stay in the domain
        z = z.abs() + 0.1 + 0.1j * z.imag
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert result.dtype == torch.complex128

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 + 0.1j, 5.0 + 0.2j], dtype=torch.complex128
        )
        result = torchscience.special_functions.modified_bessel_k_0(
            z_near_real
        )
        expected = torch.tensor(
            [scipy.special.kv(0, z.item()) for z in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_farther_from_real(self):
        """Test complex accuracy farther from real axis (relaxed tolerance)."""
        z_far = torch.tensor(
            [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.modified_bessel_k_0(z_far)
        expected = torch.tensor(
            [scipy.special.kv(0, z.item()) for z in z_far],
            dtype=torch.complex128,
        )
        # Relaxed tolerance for far-from-real complex values
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real K_0."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.modified_bessel_k_0(
            x_real
        )
        result_complex = torchscience.special_functions.modified_bessel_k_0(
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
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support.

        Note: Special functions cast to float32 for numerical precision,
        so the result is float32 rather than the autocast dtype.
        """
        z = torch.rand(10, dtype=torch.float32) + 0.1  # Avoid singularity at 0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.modified_bessel_k_0(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.modified_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.rand(5, 10, dtype=torch.float64) + 0.1  # Avoid singularity
        result = torch.vmap(
            torchscience.special_functions.modified_bessel_k_0
        )(z)
        expected = torchscience.special_functions.modified_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_k_0
        )
        z = torch.rand(100, dtype=torch.float64) + 0.1  # Avoid singularity
        result = compiled_fn(z)
        expected = torchscience.special_functions.modified_bessel_k_0(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_k_0
        )
        # Create leaf tensor (avoid + 0.1 which creates non-leaf tensor)
        z = (
            (torch.rand(100, dtype=torch.float64) + 0.1)
            .clone()
            .requires_grad_(True)
        )
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.modified_bessel_k_0(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.rand(3, 1, dtype=torch.float64) + 0.1
        z2 = torch.rand(1, 4, dtype=torch.float64) + 0.1
        result = torchscience.special_functions.modified_bessel_k_0(z1 + z2)
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
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 + 0.1j], dtype=dtype)
        result = torchscience.special_functions.modified_bessel_k_0(z)
        assert result.dtype == dtype
