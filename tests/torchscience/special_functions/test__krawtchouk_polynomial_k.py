import math

import pytest
import scipy.special
import torch

import torchscience.special_functions


def scipy_krawtchouk(n: int, x: float, p: float, N: int) -> float:
    """Reference using scipy.special.hyp2f1.

    K_n(x; p, N) = 2F1(-n, -x; -N; 1/p)

    Note: scipy's hyp2f1 can return NaN when c is a negative integer,
    even when the series terminates due to a or b being non-positive integers.
    """
    # SciPy doesn't have direct Krawtchouk, so use hyp2f1
    return float(scipy.special.hyp2f1(-n, -x, -N, 1.0 / p))


class TestKrawtchoukPolynomialK:
    """Tests for Krawtchouk polynomial K_n(x; p, N)."""

    def test_k0_equals_one(self):
        """Test that K_0(x; p, N) = 1 for all valid parameters."""
        n = torch.tensor([0.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for p_val in [0.2, 0.5, 0.8]:
                for N_val in [5.0, 10.0]:
                    x = torch.tensor([x_val], dtype=torch.float64)
                    p = torch.tensor([p_val], dtype=torch.float64)
                    N = torch.tensor([N_val], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.krawtchouk_polynomial_k(
                            n, x, p, N
                        )
                    )
                    torch.testing.assert_close(
                        result, torch.tensor([1.0], dtype=torch.float64)
                    )

    def test_k1_formula(self):
        """Test that K_1(x; p, N) = 1 - x/(N*p)."""
        n = torch.tensor([1.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for p_val in [0.2, 0.5, 0.8]:
                for N_val in [5.0, 10.0]:
                    x = torch.tensor([x_val], dtype=torch.float64)
                    p = torch.tensor([p_val], dtype=torch.float64)
                    N = torch.tensor([N_val], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.krawtchouk_polynomial_k(
                            n, x, p, N
                        )
                    )
                    expected_val = 1.0 - x_val / (N_val * p_val)
                    expected = torch.tensor(
                        [expected_val], dtype=torch.float64
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-8, atol=1e-8
                    )

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy hyp2f1 for integer orders.

        Note: scipy's hyp2f1 can return NaN when c=-N is a negative integer,
        even though the series terminates. We skip those cases.
        """
        for n_val in range(6):
            for x_val in range(6):
                for p_val in [0.3, 0.5, 0.7]:
                    N_val = 10
                    n = torch.tensor([float(n_val)], dtype=torch.float64)
                    x = torch.tensor([float(x_val)], dtype=torch.float64)
                    p = torch.tensor([p_val], dtype=torch.float64)
                    N = torch.tensor([float(N_val)], dtype=torch.float64)

                    result = (
                        torchscience.special_functions.krawtchouk_polynomial_k(
                            n, x, p, N
                        )
                    )
                    expected_val = scipy_krawtchouk(
                        n_val, float(x_val), p_val, N_val
                    )

                    # Skip cases where scipy returns NaN (due to c being negative integer)
                    if math.isnan(expected_val):
                        continue

                    expected = torch.tensor(
                        [expected_val], dtype=torch.float64
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-6, atol=1e-6
                    )

    def test_k_at_x_zero(self):
        """Test K_n(0; p, N) values."""
        # K_n(0; p, N) = 2F1(-n, 0; -N; 1/p) = 1 (since b=0 makes all terms after first vanish)
        x = torch.tensor([0.0], dtype=torch.float64)
        p = torch.tensor([0.5], dtype=torch.float64)
        N = torch.tensor([10.0], dtype=torch.float64)

        for n_val in range(6):
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.krawtchouk_polynomial_k(
                n, x, p, N
            )
            expected = torch.tensor(
                [scipy_krawtchouk(n_val, 0.0, 0.5, 10)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_gradcheck_p_parameter(self):
        """Test gradients with torch.autograd.gradcheck for p parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        p = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        N = torch.tensor([5.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda p_: torchscience.special_functions.krawtchouk_polynomial_k(
                n, x, p_, N
            ),
            (p,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_x_parameter(self):
        """Test gradients with torch.autograd.gradcheck for x parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([0.5], dtype=torch.float64)
        N = torch.tensor([5.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda x_: torchscience.special_functions.krawtchouk_polynomial_k(
                n, x_, p, N
            ),
            (x,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_complex_input(self):
        """Test with complex inputs."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        x = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        p = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        N = torch.tensor([5.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.krawtchouk_polynomial_k(
            n, x, p, N
        )
        # For real inputs, result should be real (imaginary part ~0)
        assert result.is_complex()
        expected_real = scipy_krawtchouk(2, 1.0, 0.5, 5)
        torch.testing.assert_close(
            result.real,
            torch.tensor([expected_real], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        p = torch.tensor([0.5], dtype=torch.float64)
        N = torch.tensor([5.0], dtype=torch.float64)

        result = torchscience.special_functions.krawtchouk_polynomial_k(
            n, x, p, N
        )
        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        x = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        p = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        N = torch.tensor([5.0], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.krawtchouk_polynomial_k(
            n, x, p, N
        )
        expected = scipy_krawtchouk(2, 1.0, 0.5, 5)
        torch.testing.assert_close(
            result.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        n = torch.tensor([2.0], dtype=torch.float64, device="meta")
        x = torch.tensor([1.0], dtype=torch.float64, device="meta")
        p = torch.tensor([0.5], dtype=torch.float64, device="meta")
        N = torch.tensor([5.0], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.krawtchouk_polynomial_k(
            n, x, p, N
        )
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_symmetry_property(self):
        """Test the symmetry property K_n(x; p, N) = K_x(n; p, N) for integer n, x."""
        # Krawtchouk polynomials have symmetry in n and x
        for n_val in range(5):
            for x_val in range(5):
                p_val = 0.5
                N_val = 10

                n = torch.tensor([float(n_val)], dtype=torch.float64)
                x = torch.tensor([float(x_val)], dtype=torch.float64)
                p = torch.tensor([p_val], dtype=torch.float64)
                N = torch.tensor([float(N_val)], dtype=torch.float64)

                K_nx = torchscience.special_functions.krawtchouk_polynomial_k(
                    n, x, p, N
                )
                K_xn = torchscience.special_functions.krawtchouk_polynomial_k(
                    x, n, p, N
                )
                torch.testing.assert_close(K_nx, K_xn, rtol=1e-6, atol=1e-6)

    def test_different_p_values(self):
        """Test with different probability parameters."""
        n = torch.tensor([3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        N = torch.tensor([10.0], dtype=torch.float64)

        for p_val in [0.1, 0.25, 0.5, 0.75, 0.9]:
            p = torch.tensor([p_val], dtype=torch.float64)
            result = torchscience.special_functions.krawtchouk_polynomial_k(
                n, x, p, N
            )
            expected = torch.tensor(
                [scipy_krawtchouk(3, 2.0, p_val, 10)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
