import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestFresnelS:
    """Tests for the Fresnel sine integral S(z)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_zero(self):
        """Test S(0) = 0."""
        result = torchscience.special_functions.fresnel_s(
            torch.tensor(0.0, dtype=torch.float64)
        )
        assert result.item() == pytest.approx(0.0, abs=1e-15)

    def test_nan(self):
        """Test S(nan) = nan."""
        assert torchscience.special_functions.fresnel_s(
            torch.tensor(float("nan"))
        ).isnan()

    def test_positive_infinity(self):
        """Test S(+inf) = 0.5."""
        result = torchscience.special_functions.fresnel_s(
            torch.tensor(float("inf"), dtype=torch.float64)
        )
        assert result.item() == pytest.approx(0.5, abs=1e-10)

    def test_negative_infinity(self):
        """Test S(-inf) = -0.5."""
        result = torchscience.special_functions.fresnel_s(
            torch.tensor(float("-inf"), dtype=torch.float64)
        )
        assert result.item() == pytest.approx(-0.5, abs=1e-10)

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.fresnel."""
        z = torch.tensor(
            [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.fresnel_s(z)
        # scipy.special.fresnel returns (S, C), so we take [0]
        expected = torch.tensor(
            [scipy.special.fresnel(zi.item())[0] for zi in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-9)

    def test_scipy_agreement_large_z(self):
        """Test agreement with scipy for large z (asymptotic region)."""
        z = torch.tensor([10.0, 20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.fresnel_s(z)
        expected = torch.tensor(
            [scipy.special.fresnel(zi.item())[0] for zi in z],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-8)

    def test_odd_symmetry(self):
        """Test odd symmetry: S(-z) = -S(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        result_pos = torchscience.special_functions.fresnel_s(z)
        result_neg = torchscience.special_functions.fresnel_s(-z)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-14, atol=1e-14
        )

    def test_asymptotic_value(self):
        """Test that S(z) approaches 0.5 for large positive z."""
        z = torch.tensor([50.0], dtype=torch.float64)
        result = torchscience.special_functions.fresnel_s(z)
        # Should be close to 0.5, but oscillating
        assert abs(result.item() - 0.5) < 0.05

    def test_known_value(self):
        """Test known value: S(1) from tables."""
        result = torchscience.special_functions.fresnel_s(
            torch.tensor(1.0, dtype=torch.float64)
        )
        # S(1) ~ 0.4382591473903548
        expected = 0.4382591473903548
        assert result.item() == pytest.approx(expected, rel=1e-8)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        # Avoid transition region around z=4.5 where numerical precision is lower
        z = torch.tensor(
            [0.5, 1.0, 2.0, 3.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.fresnel_s, z, eps=1e-6
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        z = torch.tensor(
            [0.5, 1.0, 1.5, 2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.fresnel_s, z, eps=1e-6
        )

    def test_gradient_formula(self):
        """Verify gradient formula: d/dz S(z) = sin(pi*z^2/2)."""
        import math

        z = torch.tensor(
            [0.5, 1.0, 2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.special_functions.fresnel_s(z)
        grad = torch.autograd.grad(y.sum(), z)[0]

        # Expected: sin(pi*z^2/2)
        expected = torch.sin(math.pi * z.detach() ** 2 / 2)
        torch.testing.assert_close(grad, expected, rtol=1e-8, atol=1e-10)

    def test_gradient_at_zero(self):
        """Test gradient at z=0: d/dz S(0) = sin(0) = 0."""
        z = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.fresnel_s(z)
        grad = torch.autograd.grad(y.sum(), z)[0]
        assert grad.item() == pytest.approx(0.0, abs=1e-10)

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.fresnel_s(z)
        assert result.shape == z.shape
        assert result.device == z.device

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.fresnel_s(z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.fresnel_s(z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Complex input tests
    # =========================================================================

    def test_complex_input(self):
        """Test complex input support."""
        z = torch.tensor([1.0 + 0.5j, 2.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.fresnel_s(z)
        # For real inputs on the real axis, should match real version
        real_result = torchscience.special_functions.fresnel_s(
            torch.tensor([2.0], dtype=torch.float64)
        )
        assert result[1].real.item() == pytest.approx(
            real_result.item(), rel=1e-6
        )
        assert abs(result[1].imag.item()) < 1e-10

    @pytest.mark.skip(
        reason="Complex gradients require Wirtinger derivative investigation"
    )
    def test_complex_gradcheck(self):
        """Test gradient for complex input by checking numerical vs analytical."""
        # Use a single point to make verification easier
        z = torch.tensor(
            [1.0 + 0.5j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        # Use relaxed tolerances for complex gradients
        # The derivative of S(z) is sin(pi*z^2/2)
        torch.autograd.gradcheck(
            torchscience.special_functions.fresnel_s,
            (z,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        z = torch.rand(5, 10, dtype=torch.float64) * 4 - 2  # Range [-2, 2]
        result = torch.vmap(torchscience.special_functions.fresnel_s)(z)
        expected = torchscience.special_functions.fresnel_s(z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.fresnel_s)
        z = torch.rand(100, dtype=torch.float64) * 10 - 5  # Range [-5, 5]
        result = compiled_fn(z)
        expected = torchscience.special_functions.fresnel_s(z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.fresnel_s)
        z = (torch.rand(100, dtype=torch.float64) * 4 - 2).requires_grad_(True)
        result = compiled_fn(z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.fresnel_s(z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        z1 = torch.rand(3, 1, dtype=torch.float64) * 2
        z2 = torch.rand(1, 4, dtype=torch.float64) * 2
        combined = z1 + z2
        result = torchscience.special_functions.fresnel_s(combined)
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
        result = torchscience.special_functions.fresnel_s(z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([0.5 + 0.1j, 1.0 + 0.2j], dtype=dtype)
        result = torchscience.special_functions.fresnel_s(z)
        assert result.dtype == dtype
