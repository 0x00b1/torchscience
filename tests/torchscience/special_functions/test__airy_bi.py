import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestAiryBi:
    """Tests for the Airy function Bi(x)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: Bi(0) ~ 0.61493, Bi(+inf) = inf, Bi(nan) = nan."""
        # Bi(0) = 1/(3^(1/6) * Gamma(2/3)) ~ 0.61492662744600073515
        result = torchscience.special_functions.airy_bi(
            torch.tensor(0.0, dtype=torch.float64)
        )
        assert result.item() == pytest.approx(
            0.61492662744600073515, rel=1e-10
        )

        # Bi(+inf) = +inf
        assert torchscience.special_functions.airy_bi(
            torch.tensor(float("inf"))
        ).isinf()

        # Bi(nan) = nan
        assert torchscience.special_functions.airy_bi(
            torch.tensor(float("nan"))
        ).isnan()

    def test_scipy_agreement_positive(self):
        """Test agreement with scipy for positive x."""
        x = torch.tensor(
            [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.airy_bi(x)
        # scipy.special.airy returns (Ai, Ai', Bi, Bi')
        expected = torch.tensor(
            [scipy.special.airy(xi.item())[2] for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_scipy_agreement_negative(self):
        """Test agreement with scipy for negative x (oscillatory region)."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.airy_bi(x)
        expected = torch.tensor(
            [scipy.special.airy(xi.item())[2] for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_scipy_agreement_large_positive(self):
        """Test agreement with scipy for large positive x."""
        x = torch.tensor([6.0, 8.0, 10.0, 15.0], dtype=torch.float64)
        result = torchscience.special_functions.airy_bi(x)
        expected = torch.tensor(
            [scipy.special.airy(xi.item())[2] for xi in x], dtype=torch.float64
        )
        # Large x leads to very large values, use relative tolerance
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-10)

    def test_scipy_agreement_large_negative(self):
        """Test agreement with scipy for large negative x."""
        x = torch.tensor([-6.0, -8.0, -10.0, -15.0], dtype=torch.float64)
        result = torchscience.special_functions.airy_bi(x)
        expected = torch.tensor(
            [scipy.special.airy(xi.item())[2] for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-10)

    def test_near_zeros(self):
        """Test behavior near zeros of Bi(x)."""
        # First few zeros of Bi: approximately -1.174, -3.271, -4.831
        zeros = torch.tensor(
            [-1.173713, -3.271093, -4.830738], dtype=torch.float64
        )
        result = torchscience.special_functions.airy_bi(zeros)
        # Should be very close to zero
        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-5, rtol=1e-4
        )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        x = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, -0.5, -1.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.airy_bi, x, eps=1e-6
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, -0.5, -1.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.airy_bi, x, eps=1e-6
        )

    def test_gradient_vs_scipy(self):
        """Verify d/dx Bi(x) = Bi'(x) using scipy reference."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, -0.5, -1.0, -2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.special_functions.airy_bi(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
        # scipy.special.airy returns (Ai, Ai', Bi, Bi')
        expected = torch.tensor(
            [scipy.special.airy(xi.item())[3] for xi in x.detach()],
            dtype=torch.float64,
        )
        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-8)

    def test_airy_differential_equation(self):
        """Verify the Airy differential equation: Bi''(x) = x * Bi(x)."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, -1.0, -2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.special_functions.airy_bi(x)

        # First derivative
        (grad_y,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        # Second derivative
        (grad2_y,) = torch.autograd.grad(grad_y.sum(), x)

        # Check: y'' should equal x * y
        expected = x.detach() * y.detach()
        torch.testing.assert_close(grad2_y, expected, rtol=1e-5, atol=1e-8)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        z = torch.randn(10, dtype=torch.complex128)
        result = torchscience.special_functions.airy_bi(z)
        assert result.dtype == torch.complex128

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        z_near_real = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, -1.0 + 0.1j], dtype=torch.complex128
        )
        result = torchscience.special_functions.airy_bi(z_near_real)
        expected = torch.tensor(
            [scipy.special.airy(zi.item())[2] for zi in z_near_real],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real Bi."""
        x_real = torch.tensor([0.5, 1.0, 2.0, -0.5, -1.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.airy_bi(x_real)
        result_complex = torchscience.special_functions.airy_bi(x_complex)
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
        x = torch.randn(10, device="meta")
        result = torchscience.special_functions.airy_bi(x)
        assert result.shape == x.shape
        assert result.device == x.device

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        x = torch.randn(10, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.airy_bi(x)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.airy_bi(x)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        x = torch.randn(5, 10, dtype=torch.float64)
        result = torch.vmap(torchscience.special_functions.airy_bi)(x)
        expected = torchscience.special_functions.airy_bi(x)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.airy_bi)
        x = torch.randn(100, dtype=torch.float64)
        result = compiled_fn(x)
        expected = torchscience.special_functions.airy_bi(x)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.airy_bi)
        x = torch.randn(100, dtype=torch.float64, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        assert x.grad is not None
        # Verify gradient matches uncompiled version
        x2 = x.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.airy_bi(x2)
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        x1 = torch.randn(3, 1, dtype=torch.float64)
        x2 = torch.randn(1, 4, dtype=torch.float64)
        result = torchscience.special_functions.airy_bi(x1 + x2)
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
        result = torchscience.special_functions.airy_bi(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.airy_bi(z)
        assert result.dtype == dtype
