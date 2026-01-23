import pytest
import torch

from torchscience.root_finding import aberth_ehrlich


class TestAberthEhrlich:
    """Tests for Aberth-Ehrlich polynomial root finding."""

    def test_quadratic_roots(self):
        """Find roots of x^2 - 5x + 6 = (x-2)(x-3)."""
        # Coefficients in ascending order: c_0 + c_1*x + c_2*x^2
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)

        # Sort for comparison
        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_cubic_roots(self):
        """Find roots of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6."""
        coeffs = torch.tensor([-6.0, 11.0, -6.0, 1.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)

        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_complex_roots(self):
        """Find roots of x^2 + 1 = 0 (roots are +/- i)."""
        coeffs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)

        # Check |roots| == 1 (on unit circle)
        torch.testing.assert_close(
            roots.abs(),
            torch.ones(2, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        # Check imaginary parts are +/- 1
        imag_sorted = torch.sort(roots.imag)[0]
        torch.testing.assert_close(
            imag_sorted,
            torch.tensor([-1.0, 1.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_high_degree(self):
        """Aberth-Ehrlich handles high-degree polynomials."""
        # Chebyshev polynomial of degree 20 has 20 real roots in [-1, 1]
        n = 20
        # Build from roots for verification
        indices = torch.arange(n, dtype=torch.float64)
        roots_true = torch.cos(torch.pi * (2 * indices + 1) / (2 * n))

        # Build polynomial from roots (monic)
        coeffs = torch.tensor([1.0], dtype=torch.float64)
        for r in roots_true:
            # Multiply by (x - r)
            new_coeffs = torch.zeros(len(coeffs) + 1, dtype=torch.float64)
            new_coeffs[1:] = coeffs
            new_coeffs[:-1] -= r * coeffs
            coeffs = new_coeffs

        roots = aberth_ehrlich(coeffs)
        roots_sorted = torch.sort(roots.real)[0]
        roots_true_sorted = torch.sort(roots_true)[0]

        torch.testing.assert_close(
            roots_sorted, roots_true_sorted, atol=1e-8, rtol=1e-8
        )

    def test_batched(self):
        """Aberth-Ehrlich works with batched coefficients."""
        # Batch of 3 quadratics
        coeffs = torch.tensor(
            [
                [2.0, -3.0, 1.0],  # (x-1)(x-2)
                [6.0, -5.0, 1.0],  # (x-2)(x-3)
                [12.0, -7.0, 1.0],  # (x-3)(x-4)
            ],
            dtype=torch.float64,
        )

        roots = aberth_ehrlich(coeffs)
        assert roots.shape == (3, 2)  # 3 batches, 2 roots each

        # Check first batch
        roots_0 = torch.sort(roots[0].real)[0]
        torch.testing.assert_close(
            roots_0,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_linear_polynomial(self):
        """Find root of ax + b = 0."""
        # 2x + 4 = 0 -> x = -2
        coeffs = torch.tensor([4.0, 2.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)

        assert roots.shape == (1,)
        torch.testing.assert_close(
            roots.real,
            torch.tensor([-2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_quartic_roots(self):
        """Find roots of (x-1)(x-2)(x-3)(x-4)."""
        coeffs = torch.tensor(
            [24.0, -50.0, 35.0, -10.0, 1.0], dtype=torch.float64
        )
        roots = aberth_ehrlich(coeffs)

        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-9, rtol=1e-9
        )

    def test_repeated_roots(self):
        """Handle polynomial with repeated roots: (x-2)^2 = x^2 - 4x + 4."""
        coeffs = torch.tensor([4.0, -4.0, 1.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)

        # Both roots should be approximately 2
        torch.testing.assert_close(
            roots.real,
            torch.tensor([2.0, 2.0], dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_float32_precision(self):
        """Works with float32 input."""
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float32)
        roots = aberth_ehrlich(coeffs)

        # Output should be complex64
        assert roots.dtype == torch.complex64

        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([2.0, 3.0], dtype=torch.float32)
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-5, rtol=1e-5
        )

    def test_complex_coefficients(self):
        """Works with complex coefficients."""
        # (x - (1+i))(x - (1-i)) = x^2 - 2x + 2
        coeffs = torch.tensor([2.0, -2.0, 1.0], dtype=torch.complex128)
        roots = aberth_ehrlich(coeffs)

        # Sort by real then imaginary
        roots_sorted = roots[torch.argsort(roots.imag)]
        expected = torch.tensor(
            [1.0 - 1.0j, 1.0 + 1.0j], dtype=torch.complex128
        )
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_degree_error(self):
        """Raises error for constant polynomial."""
        coeffs = torch.tensor([5.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="degree >= 1"):
            aberth_ehrlich(coeffs)

    def test_device_preservation(self):
        """Output is on same device as input."""
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
        roots = aberth_ehrlich(coeffs)
        assert roots.device == coeffs.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_support(self):
        """Works on CUDA device."""
        coeffs = torch.tensor(
            [6.0, -5.0, 1.0], dtype=torch.float64, device="cuda"
        )
        roots = aberth_ehrlich(coeffs)

        assert roots.device.type == "cuda"
        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([2.0, 3.0], dtype=torch.float64, device="cuda")
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )
