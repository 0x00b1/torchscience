"""Tests for adaptive polynomial multiplication dispatch."""

from unittest.mock import patch

import torch
from torch import Tensor

from torchscience.polynomial import polynomial, polynomial_multiply


class TestPolynomialMultiplyAdaptive:
    """Tests for adaptive multiplication dispatch."""

    def test_multiply_low_degree_uses_direct(self):
        """Low-degree multiplication uses direct convolution."""
        a = polynomial(torch.randn(10))  # degree 9
        b = polynomial(torch.randn(10))

        with patch(
            "torchscience.polynomial._polynomial._polynomial_multiply._multiply_direct"
        ) as mock_direct:
            mock_direct.return_value = polynomial(torch.randn(19))
            polynomial_multiply(a, b)
            mock_direct.assert_called_once()

    def test_multiply_high_degree_uses_fft(self):
        """High-degree multiplication uses FFT."""
        a = polynomial(torch.randn(100))  # degree 99
        b = polynomial(torch.randn(100))

        with patch(
            "torchscience.polynomial._polynomial._polynomial_multiply._multiply_fft"
        ) as mock_fft:
            mock_fft.return_value = polynomial(torch.randn(199))
            polynomial_multiply(a, b)
            mock_fft.assert_called_once()

    def test_multiply_correctness_across_threshold(self):
        """Multiplication is correct regardless of method used."""
        torch.manual_seed(42)
        for degree in [10, 50, 100, 200]:
            a = polynomial(torch.randn(degree + 1))
            b = polynomial(torch.randn(degree + 1))
            result = polynomial_multiply(a, b)

            # Verify by evaluation
            # Convert to plain tensors for element-wise multiplication
            # (Polynomial.__mul__ does polynomial multiplication, not element-wise)
            x = torch.tensor([0.1, 0.5])
            a_eval = a(x).as_subclass(Tensor)
            b_eval = b(x).as_subclass(Tensor)
            result_eval = result(x).as_subclass(Tensor)
            torch.testing.assert_close(
                result_eval, a_eval * b_eval, atol=1e-4, rtol=1e-4
            )

    def test_multiply_at_threshold_boundary(self):
        """Test behavior exactly at the FFT threshold."""
        # Output size = n_p + n_q - 1
        # For threshold = 64, we need n_out >= 64 to use FFT

        # Just below threshold: output size 63 -> uses direct
        a = polynomial(torch.randn(32))  # 32 + 32 - 1 = 63 < 64
        b = polynomial(torch.randn(32))
        with patch(
            "torchscience.polynomial._polynomial._polynomial_multiply._multiply_direct"
        ) as mock_direct:
            mock_direct.return_value = polynomial(torch.randn(63))
            polynomial_multiply(a, b)
            mock_direct.assert_called_once()

        # At threshold: output size 64 -> uses FFT
        a = polynomial(torch.randn(33))  # 33 + 32 - 1 = 64 >= 64
        b = polynomial(torch.randn(32))
        with patch(
            "torchscience.polynomial._polynomial._polynomial_multiply._multiply_fft"
        ) as mock_fft:
            mock_fft.return_value = polynomial(torch.randn(64))
            polynomial_multiply(a, b)
            mock_fft.assert_called_once()

    def test_multiply_preserves_dtype(self):
        """Multiplication preserves dtype across both methods."""
        for degree in [10, 100]:  # Below and above threshold
            for dtype in [torch.float32, torch.float64]:
                a = polynomial(torch.randn(degree + 1, dtype=dtype))
                b = polynomial(torch.randn(degree + 1, dtype=dtype))
                result = polynomial_multiply(a, b)
                assert result.dtype == dtype

    def test_multiply_preserves_device(self):
        """Multiplication preserves device across both methods."""
        for degree in [10, 100]:  # Below and above threshold
            a = polynomial(torch.randn(degree + 1))
            b = polynomial(torch.randn(degree + 1))
            result = polynomial_multiply(a, b)
            assert result.device == a.device

    def test_multiply_batched_polynomials(self):
        """Batched multiplication works with adaptive dispatch."""
        torch.manual_seed(42)
        for degree in [10, 100]:  # Below and above threshold
            a = polynomial(torch.randn(3, degree + 1))
            b = polynomial(torch.randn(3, degree + 1))
            result = polynomial_multiply(a, b)

            assert result.shape == (3, 2 * degree + 1)

            # Verify correctness
            # Convert to plain tensors for element-wise multiplication
            # (Polynomial.__mul__ does polynomial multiplication, not element-wise)
            x = torch.tensor([0.5])
            for i in range(3):
                a_eval = a[i](x).as_subclass(Tensor)
                b_eval = b[i](x).as_subclass(Tensor)
                expected = a_eval * b_eval
                actual = result[i](x).as_subclass(Tensor)
                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )

    def test_multiply_empty_polynomial(self):
        """Empty polynomial handling works with adaptive dispatch."""
        a = polynomial(torch.tensor([1.0, 2.0]))

        # Create an empty-ish polynomial (1 coefficient = degree 0)
        b = polynomial(torch.tensor([0.0]))
        result = polynomial_multiply(a, b)

        # Result should be zero polynomial
        torch.testing.assert_close(
            result, polynomial(torch.tensor([0.0, 0.0])), atol=1e-7, rtol=1e-7
        )
