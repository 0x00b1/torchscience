import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestBeta:
    """Tests for the beta function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_against_log_gamma(self):
        """Test beta(a, b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0, 0.5], dtype=torch.float64)
        b = torch.tensor([1.0, 3.0, 2.0, 2.0, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = torch.exp(
            torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_b_1_1(self):
        """Test B(1, 1) = 1."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_b_half_half(self):
        """Test B(0.5, 0.5) = pi."""
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = torch.tensor([math.pi], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_b_1_n(self):
        """Test B(1, n) = 1/n."""
        a = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = 1.0 / b
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """Test B(a, b) = B(b, a)."""
        a = torch.tensor([2.0, 3.0, 0.5, 5.0], dtype=torch.float64)
        b = torch.tensor([5.0, 2.0, 3.0, 0.5], dtype=torch.float64)
        result_ab = torchscience.special_functions.beta(a, b)
        result_ba = torchscience.special_functions.beta(b, a)
        torch.testing.assert_close(
            result_ab, result_ba, rtol=1e-10, atol=1e-10
        )

    def test_relationship_to_gamma(self):
        """Test B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)."""
        a = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
        b = torch.tensor([3.0, 2.0, 2.5], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        gamma = torchscience.special_functions.gamma
        expected = gamma(a) * gamma(b) / gamma(a + b)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        a = torch.tensor([2.0, 3.0], dtype=dtype)
        b = torch.tensor([3.0, 2.0], dtype=dtype)
        result = torchscience.special_functions.beta(a, b)
        assert result.dtype == dtype
        expected = torch.exp(
            torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test forward pass for complex dtypes."""
        a = torch.tensor([2.0 + 0.5j, 3.0 + 0.0j], dtype=dtype)
        b = torch.tensor([3.0 + 0.0j, 2.0 + 0.5j], dtype=dtype)
        result = torchscience.special_functions.beta(a, b)
        assert result.dtype == dtype
        # Just check it runs without error and produces finite values
        assert torch.isfinite(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        a = torch.tensor(
            [2.0, 3.0, 1.5], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor(
            [3.0, 2.0, 2.5], dtype=torch.float64, requires_grad=True
        )

        def func(a, b):
            return torchscience.special_functions.beta(a, b)

        assert torch.autograd.gradcheck(
            func, (a, b), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(a, b):
            return torchscience.special_functions.beta(a, b)

        assert torch.autograd.gradgradcheck(
            func, (a, b), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_values(self):
        """Test gradient values against analytical formula."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.beta(a, b)
        result.backward()

        # dB/da = B(a,b) * (digamma(a) - digamma(a+b))
        beta_val = result.detach()
        psi_a = torch.digamma(a.detach())
        psi_b = torch.digamma(b.detach())
        psi_ab = torch.digamma(a.detach() + b.detach())

        expected_grad_a = beta_val * (psi_a - psi_ab)
        expected_grad_b = beta_val * (psi_b - psi_ab)

        torch.testing.assert_close(
            a.grad, expected_grad_a, rtol=1e-5, atol=1e-10
        )
        torch.testing.assert_close(
            b.grad, expected_grad_b, rtol=1e-5, atol=1e-10
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        a = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        result = torchscience.special_functions.beta(a, b)
        assert result.shape == (2, 3)
        expected = torch.exp(
            torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_small_values(self):
        """Test with small positive values (potential overflow in gamma)."""
        a = torch.tensor([0.1, 0.01], dtype=torch.float64)
        b = torch.tensor([0.1, 0.01], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = torch.exp(
            torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_large_values(self):
        """Test with large values (potential underflow)."""
        a = torch.tensor([10.0, 20.0], dtype=torch.float64)
        b = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = torchscience.special_functions.beta(a, b)
        expected = torch.exp(
            torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-15)

    # =========================================================================
    # Complex input tests
    # =========================================================================

    def test_complex_symmetry(self):
        """Test B(a, b) = B(b, a) for complex inputs."""
        a = torch.tensor([2.0 + 1.0j, 3.0 + 0.5j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.5j, 2.0 + 1.0j], dtype=torch.complex128)
        result_ab = torchscience.special_functions.beta(a, b)
        result_ba = torchscience.special_functions.beta(b, a)
        torch.testing.assert_close(
            result_ab, result_ba, rtol=1e-10, atol=1e-10
        )

    def test_complex_real_axis(self):
        """Test complex inputs on real axis match real results."""
        a_real = torch.tensor([2.0, 3.0], dtype=torch.float64)
        b_real = torch.tensor([3.0, 2.0], dtype=torch.float64)
        a_complex = a_real.to(torch.complex128)
        b_complex = b_real.to(torch.complex128)

        result_real = torchscience.special_functions.beta(a_real, b_real)
        result_complex = torchscience.special_functions.beta(
            a_complex, b_complex
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_complex.imag),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.xfail(
        reason="Complex dispatch not yet implemented in CPU macros"
    )
    def test_gradcheck_complex(self, dtype):
        """Test first-order gradients for complex inputs."""
        # Use values away from singularities
        a = torch.tensor(
            [2.0 + 0.5j, 3.0 + 0.3j], dtype=dtype, requires_grad=True
        )
        b = torch.tensor(
            [3.0 + 0.3j, 2.0 + 0.5j], dtype=dtype, requires_grad=True
        )

        def func(a, b):
            return torchscience.special_functions.beta(a, b)

        # Complex gradcheck uses Wirtinger derivatives
        assert torch.autograd.gradcheck(
            func, (a, b), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        a = torch.empty(3, 4, device="meta")
        b = torch.empty(3, 4, device="meta")
        result = torchscience.special_functions.beta(a, b)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
