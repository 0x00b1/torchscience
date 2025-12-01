import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestBesselJ1:
    """Tests for the Bessel function J₁."""

    def test_special_values(self):
        """Test special values: J₁(0) = 0."""
        assert torchscience.special_functions.bessel_j_1(
            torch.tensor(0.0)
        ).item() == pytest.approx(0.0, abs=1e-10)

    def test_scipy_agreement_small(self):
        """Test agreement with scipy for small |z| <= 5."""
        z = torch.tensor(
            [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.9], dtype=torch.float64
        )
        result = torchscience.special_functions.bessel_j_1(z)
        expected = torch.tensor(
            [scipy.special.j1(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_large(self):
        """Test agreement with scipy for large |z| > 5."""
        z = torch.tensor([5.1, 10.0, 20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.bessel_j_1(z)
        expected = torch.tensor(
            [scipy.special.j1(x.item()) for x in z], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry_odd_function(self):
        """J₁(-z) = -J₁(z) (odd function)."""
        z = torch.tensor([0.1, 1.0, 2.0, 5.0, 10.0, 50.0], dtype=torch.float64)
        result_pos = torchscience.special_functions.bessel_j_1(z)
        result_neg = torchscience.special_functions.bessel_j_1(-z)
        torch.testing.assert_close(
            result_pos, -result_neg, rtol=1e-12, atol=1e-12
        )

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
            torchscience.special_functions.bessel_j_1, z
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        # Avoid z=0 where second derivative has a removable singularity
        z_pos = torch.rand(5, dtype=torch.float64) + 0.1
        z = torch.cat([z_pos, -z_pos])
        z = z.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.bessel_j_1, z
        )

    def test_gradient_identity(self):
        """Verify d/dz J₁(z) = J₀(z) - J₁(z)/z numerically."""
        z = torch.tensor(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.bessel_j_1(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        z_detach = z.detach()
        expected = (
            torchscience.special_functions.bessel_j_0(z_detach)
            - torchscience.special_functions.bessel_j_1(z_detach) / z_detach
        )
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        result = torchscience.special_functions.bessel_j_1(z)
        assert result.dtype == torch.complex128

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.bessel_j_1(z)
        assert result.shape == z.shape

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.randn(5, 10, dtype=torch.float64)
        result = torch.vmap(torchscience.special_functions.bessel_j_1)(z)
        expected = torchscience.special_functions.bessel_j_1(z)
        torch.testing.assert_close(result, expected)

    def test_autocast(self):
        """Test autocast support."""
        z = torch.randn(10, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.bessel_j_1(z)
        assert result.dtype == torch.float32

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.bessel_j_1)
        z = torch.randn(100, dtype=torch.float64)
        result = compiled_fn(z)
        expected = torchscience.special_functions.bessel_j_1(z)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.bessel_j_1(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.bessel_j_1(z)
        assert result.dtype == dtype
