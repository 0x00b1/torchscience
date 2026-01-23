"""Tests for FFT-based polynomial multiplication."""

import pytest
import torch

from torchscience.polynomial import (
    polynomial,
    polynomial_evaluate,
    polynomial_multiply,
    polynomial_multiply_auto,
    polynomial_multiply_fft,
)


class TestPolynomialMultiplyFFT:
    """Tests for FFT-based polynomial multiplication."""

    def test_simple_multiply(self):
        """Test multiplication of two simple polynomials."""
        # (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        result = polynomial_multiply_fft(p, q)

        expected = torch.tensor([3.0, 10.0, 8.0])
        torch.testing.assert_close(result, expected)

    def test_matches_direct_multiply(self):
        """FFT multiply should match direct multiply."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64))

        result_fft = polynomial_multiply_fft(p, q)
        result_direct = polynomial_multiply(p, q)

        torch.testing.assert_close(result_fft, result_direct)

    def test_high_degree(self):
        """Test with high-degree polynomials where FFT shines."""
        degree = 100
        p = polynomial(torch.randn(degree + 1, dtype=torch.float64))
        q = polynomial(torch.randn(degree + 1, dtype=torch.float64))

        result_fft = polynomial_multiply_fft(p, q)
        result_direct = polynomial_multiply(p, q)

        # Should match within numerical tolerance
        torch.testing.assert_close(
            result_fft, result_direct, rtol=1e-10, atol=1e-10
        )

    def test_evaluation_consistency(self):
        """Result should evaluate correctly at test points."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        q = polynomial(torch.tensor([4.0, -1.0, 2.0], dtype=torch.float64))

        result = polynomial_multiply_fft(p, q)

        # Evaluate at several points
        x = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5], dtype=torch.float64)
        p_vals = torch.Tensor(polynomial_evaluate(p, x))
        q_vals = torch.Tensor(polynomial_evaluate(q, x))
        result_vals = torch.Tensor(polynomial_evaluate(result, x))

        # p(x) * q(x) should equal result(x)
        # Note: use torch.Tensor() to get element-wise multiplication, not polynomial multiplication
        torch.testing.assert_close(p_vals * q_vals, result_vals)

    def test_multiply_by_one(self):
        """Multiplying by 1 should return the same polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        one = polynomial(torch.tensor([1.0], dtype=torch.float64))

        result = polynomial_multiply_fft(p, one)

        torch.testing.assert_close(result, p)

    def test_multiply_by_zero(self):
        """Multiplying by 0 should return zero polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        zero = polynomial(torch.tensor([0.0], dtype=torch.float64))

        result = polynomial_multiply_fft(p, zero)

        # Result should be all zeros (with shape deg(p) + deg(0) + 1 = 3)
        expected_shape = p.shape[-1]
        assert result.shape[-1] == expected_shape
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_multiply_by_constant(self):
        """Multiplying by constant scales coefficients."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        const = polynomial(torch.tensor([2.5], dtype=torch.float64))

        result = polynomial_multiply_fft(p, const)

        expected = torch.tensor([2.5, 5.0, 7.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_batch_dimensions(self):
        """Test with batch dimensions."""
        # Batch of 3 polynomials
        p = polynomial(torch.randn(3, 4, dtype=torch.float64))
        q = polynomial(torch.randn(3, 5, dtype=torch.float64))

        result = polynomial_multiply_fft(p, q)

        assert result.shape == (3, 8)  # 4 + 5 - 1 = 8

        # Verify each batch element
        for i in range(3):
            expected = polynomial_multiply(polynomial(p[i]), polynomial(q[i]))
            torch.testing.assert_close(result[i], expected)

    def test_broadcast_batch(self):
        """Test batch broadcasting."""
        # (2, 1, 3) broadcast with (4, 3)
        p = polynomial(torch.randn(2, 1, 3, dtype=torch.float64))
        q = polynomial(torch.randn(4, 3, dtype=torch.float64))

        result = polynomial_multiply_fft(p, q)

        assert result.shape == (2, 4, 5)  # 3 + 3 - 1 = 5

    def test_complex_coefficients(self):
        """Test with complex coefficients."""
        p = polynomial(
            torch.tensor([1.0 + 1j, 2.0 - 1j], dtype=torch.complex128)
        )
        q = polynomial(
            torch.tensor([1.0 - 1j, 1.0 + 1j], dtype=torch.complex128)
        )

        result = polynomial_multiply_fft(p, q)
        expected = polynomial_multiply(p, q)

        torch.testing.assert_close(result, expected)

    def test_different_dtypes(self):
        """Test with different dtypes."""
        p_f32 = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        q_f32 = polynomial(torch.tensor([4.0, 5.0], dtype=torch.float32))

        result = polynomial_multiply_fft(p_f32, q_f32)

        # Should maintain float32
        assert result.dtype == torch.float32

        expected = polynomial_multiply(p_f32, q_f32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_integer_coefficients(self):
        """Test with integer coefficients."""
        p = polynomial(torch.tensor([1, 2, 3]))
        q = polynomial(torch.tensor([4, 5]))

        result = polynomial_multiply_fft(p, q)

        # Should be converted back to integer
        expected = torch.tensor([4, 13, 22, 15])
        torch.testing.assert_close(result, expected)

    def test_gradients(self):
        """Test that gradients flow through FFT multiply."""
        p_coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        q_coeffs = torch.tensor(
            [4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        p = polynomial(p_coeffs)
        q = polynomial(q_coeffs)

        result = polynomial_multiply_fft(p, q)
        loss = result.sum()
        loss.backward()

        assert p_coeffs.grad is not None
        assert q_coeffs.grad is not None
        assert p_coeffs.grad.shape == p_coeffs.shape
        assert q_coeffs.grad.shape == q_coeffs.shape


class TestPolynomialMultiplyFFTAutograd:
    """Autograd tests for FFT multiplication."""

    def test_gradcheck(self):
        """torch.autograd.gradcheck for FFT multiply."""
        a_coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def mul_fn(a, b):
            return polynomial_multiply_fft(polynomial(a), polynomial(b))

        assert torch.autograd.gradcheck(mul_fn, (a_coeffs, b_coeffs))

    def test_gradgradcheck(self):
        """torch.autograd.gradgradcheck for FFT multiply."""
        a_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def mul_fn(a, b):
            return polynomial_multiply_fft(polynomial(a), polynomial(b)).sum()

        assert torch.autograd.gradgradcheck(mul_fn, (a_coeffs, b_coeffs))

    def test_gradcheck_high_degree(self):
        """torch.autograd.gradcheck for high-degree FFT multiply."""
        torch.manual_seed(42)
        a_coeffs = torch.randn(20, dtype=torch.float64, requires_grad=True)
        b_coeffs = torch.randn(15, dtype=torch.float64, requires_grad=True)

        def mul_fn(a, b):
            return polynomial_multiply_fft(polynomial(a), polynomial(b))

        assert torch.autograd.gradcheck(mul_fn, (a_coeffs, b_coeffs))


class TestPolynomialMultiplyAuto:
    """Tests for automatic algorithm selection."""

    def test_low_degree_uses_direct(self):
        """Low degree polynomials should use direct method."""
        # Degrees below threshold
        p = polynomial(torch.randn(10, dtype=torch.float64))
        q = polynomial(torch.randn(10, dtype=torch.float64))

        result = polynomial_multiply_auto(p, q)
        expected = polynomial_multiply(p, q)

        torch.testing.assert_close(result, expected)

    def test_high_degree_uses_fft(self):
        """High degree polynomials should use FFT."""
        # Degrees above threshold (64)
        p = polynomial(torch.randn(100, dtype=torch.float64))
        q = polynomial(torch.randn(100, dtype=torch.float64))

        result = polynomial_multiply_auto(p, q)
        expected = polynomial_multiply_fft(p, q)

        torch.testing.assert_close(result, expected)

    def test_correctness_across_threshold(self):
        """Results should be consistent regardless of algorithm used."""
        # Test around the threshold (64)
        for n in [50, 60, 64, 65, 70, 80]:
            p = polynomial(torch.randn(n, dtype=torch.float64))
            q = polynomial(torch.randn(n, dtype=torch.float64))

            result_auto = polynomial_multiply_auto(p, q)
            result_direct = polynomial_multiply(p, q)

            torch.testing.assert_close(
                result_auto,
                result_direct,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Mismatch at degree {n - 1}",
            )


class TestFFTPerformance:
    """Tests verifying FFT performance characteristics."""

    @pytest.mark.parametrize("degree", [32, 64, 128, 256])
    def test_result_shape(self, degree):
        """Verify result shape is correct for various degrees."""
        p = polynomial(torch.randn(degree + 1, dtype=torch.float64))
        q = polynomial(torch.randn(degree + 1, dtype=torch.float64))

        result = polynomial_multiply_fft(p, q)

        # Result degree = deg(p) + deg(q) = 2*degree
        assert result.shape[-1] == 2 * degree + 1

    def test_numerical_stability_high_degree(self):
        """Test numerical stability with high degree polynomials."""
        degree = 500
        p = polynomial(torch.randn(degree + 1, dtype=torch.float64))
        q = polynomial(torch.randn(degree + 1, dtype=torch.float64))

        result = polynomial_multiply_fft(p, q)

        # Verify at a few points
        x = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float64)
        # Convert to plain tensors for element-wise multiplication (not polynomial multiplication)
        p_vals = torch.Tensor(polynomial_evaluate(p, x))
        q_vals = torch.Tensor(polynomial_evaluate(q, x))
        result_vals = torch.Tensor(polynomial_evaluate(result, x))

        # At x near 0, should be accurate
        torch.testing.assert_close(
            p_vals * q_vals, result_vals, rtol=1e-8, atol=1e-8
        )
