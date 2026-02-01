import math

import pytest
import torch

import torchscience.special_functions


def scipy_hahn(n: int, x: float, alpha: float, beta: float, N: int) -> float:
    """Reference using scipy.special.hyp2f1 to compute 3F2.

    Q_n(x; alpha, beta, N) = 3F2(-n, n+alpha+beta+1, -x; alpha+1, -N; 1)

    For a terminating 3F2 (when -n is a non-positive integer), we can compute
    it as a finite sum.
    """
    if n == 0:
        return 1.0

    # Compute via explicit sum for the polynomial case
    result = 0.0
    for k in range(n + 1):
        # Compute Pochhammer symbols
        neg_n_k = pochhammer(-n, k)
        n_ab1_k = pochhammer(n + alpha + beta + 1, k)
        neg_x_k = pochhammer(-x, k)
        alpha1_k = pochhammer(alpha + 1, k)
        neg_N_k = pochhammer(-N, k)
        k_factorial = math.factorial(k)

        if abs(alpha1_k * neg_N_k * k_factorial) < 1e-15:
            if abs(neg_n_k * n_ab1_k * neg_x_k) < 1e-15:
                continue
            return float("inf")

        term = (neg_n_k * n_ab1_k * neg_x_k) / (
            alpha1_k * neg_N_k * k_factorial
        )
        result += term

    return result


def pochhammer(a: float, k: int) -> float:
    """Compute the Pochhammer symbol (a)_k = a(a+1)...(a+k-1)."""
    if k <= 0:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= a + i
    return result


class TestHahnPolynomialQ:
    """Tests for Hahn polynomial Q_n(x; alpha, beta, N)."""

    def test_q0_equals_one(self):
        """Test that Q_0(x; alpha, beta, N) = 1 for all valid parameters."""
        n = torch.tensor([0.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for alpha_val in [0.5, 1.0, 2.0]:
                for beta_val in [0.5, 1.0, 2.0]:
                    for N_val in [5.0, 10.0]:
                        x = torch.tensor([x_val], dtype=torch.float64)
                        alpha = torch.tensor([alpha_val], dtype=torch.float64)
                        beta = torch.tensor([beta_val], dtype=torch.float64)
                        N = torch.tensor([N_val], dtype=torch.float64)
                        result = (
                            torchscience.special_functions.hahn_polynomial_q(
                                n, x, alpha, beta, N
                            )
                        )
                        torch.testing.assert_close(
                            result, torch.tensor([1.0], dtype=torch.float64)
                        )

    def test_q1_formula(self):
        """Test that Q_1(x; alpha, beta, N) = 1 - (alpha+beta+2)*x / ((alpha+1)*N)."""
        n = torch.tensor([1.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for alpha_val in [0.5, 1.0, 2.0]:
                for beta_val in [0.5, 1.0, 2.0]:
                    for N_val in [5.0, 10.0]:
                        x = torch.tensor([x_val], dtype=torch.float64)
                        alpha = torch.tensor([alpha_val], dtype=torch.float64)
                        beta = torch.tensor([beta_val], dtype=torch.float64)
                        N = torch.tensor([N_val], dtype=torch.float64)
                        result = (
                            torchscience.special_functions.hahn_polynomial_q(
                                n, x, alpha, beta, N
                            )
                        )
                        expected_val = 1.0 - (
                            alpha_val + beta_val + 2
                        ) * x_val / ((alpha_val + 1) * N_val)
                        expected = torch.tensor(
                            [expected_val], dtype=torch.float64
                        )
                        torch.testing.assert_close(
                            result, expected, rtol=1e-8, atol=1e-8
                        )

    def test_reference_agreement_integer_orders(self):
        """Test agreement with reference implementation for integer orders."""
        for n_val in range(5):
            for x_val in range(5):
                for alpha_val in [0.5, 1.0]:
                    for beta_val in [0.5, 1.0]:
                        N_val = 10
                        n = torch.tensor([float(n_val)], dtype=torch.float64)
                        x = torch.tensor([float(x_val)], dtype=torch.float64)
                        alpha = torch.tensor([alpha_val], dtype=torch.float64)
                        beta = torch.tensor([beta_val], dtype=torch.float64)
                        N = torch.tensor([float(N_val)], dtype=torch.float64)

                        result = (
                            torchscience.special_functions.hahn_polynomial_q(
                                n, x, alpha, beta, N
                            )
                        )
                        expected_val = scipy_hahn(
                            n_val, float(x_val), alpha_val, beta_val, N_val
                        )

                        # Skip cases with infinity or NaN
                        if math.isnan(expected_val) or math.isinf(
                            expected_val
                        ):
                            continue

                        expected = torch.tensor(
                            [expected_val], dtype=torch.float64
                        )
                        torch.testing.assert_close(
                            result, expected, rtol=1e-6, atol=1e-6
                        )

    def test_q_at_x_zero(self):
        """Test Q_n(0; alpha, beta, N) values."""
        # When x=0, the (-x)_k term is 1 for k=0 and 0 for k>0
        # So Q_n(0; alpha, beta, N) = 1
        x = torch.tensor([0.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([1.0], dtype=torch.float64)
        N = torch.tensor([10.0], dtype=torch.float64)

        for n_val in range(6):
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha, beta, N
            )
            expected = torch.tensor(
                [scipy_hahn(n_val, 0.0, 1.0, 1.0, 10)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_gradcheck_x_parameter(self):
        """Test gradients with torch.autograd.gradcheck for x parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([1.0], dtype=torch.float64)
        N = torch.tensor([5.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda x_: torchscience.special_functions.hahn_polynomial_q(
                n, x_, alpha, beta, N
            ),
            (x,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_alpha_parameter(self):
        """Test gradients with torch.autograd.gradcheck for alpha parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        beta = torch.tensor([1.0], dtype=torch.float64)
        N = torch.tensor([5.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda alpha_: torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha_, beta, N
            ),
            (alpha,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_beta_parameter(self):
        """Test gradients with torch.autograd.gradcheck for beta parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        N = torch.tensor([5.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda beta_: torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha, beta_, N
            ),
            (beta,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.skip(
        reason="Complex number support for quinary operators not yet implemented"
    )
    def test_complex_input(self):
        """Test with complex inputs."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        x = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        alpha = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        beta = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        N = torch.tensor([5.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.hahn_polynomial_q(
            n, x, alpha, beta, N
        )
        # For real inputs, result should be real (imaginary part ~0)
        assert result.is_complex()
        expected_real = scipy_hahn(2, 1.0, 1.0, 1.0, 5)
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
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([1.0], dtype=torch.float64)
        N = torch.tensor([5.0], dtype=torch.float64)

        result = torchscience.special_functions.hahn_polynomial_q(
            n, x, alpha, beta, N
        )
        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        x = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        alpha = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        beta = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        N = torch.tensor([5.0], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.hahn_polynomial_q(
            n, x, alpha, beta, N
        )
        expected = scipy_hahn(2, 1.0, 1.0, 1.0, 5)
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
        alpha = torch.tensor([1.0], dtype=torch.float64, device="meta")
        beta = torch.tensor([1.0], dtype=torch.float64, device="meta")
        N = torch.tensor([5.0], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.hahn_polynomial_q(
            n, x, alpha, beta, N
        )
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_symmetric_alpha_beta(self):
        """Test behavior when alpha = beta (symmetric case)."""
        n = torch.tensor([3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        N = torch.tensor([10.0], dtype=torch.float64)

        for param_val in [0.5, 1.0, 2.0]:
            alpha = torch.tensor([param_val], dtype=torch.float64)
            beta = torch.tensor([param_val], dtype=torch.float64)
            result = torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha, beta, N
            )
            expected = torch.tensor(
                [scipy_hahn(3, 2.0, param_val, param_val, 10)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_different_alpha_beta_values(self):
        """Test with different alpha and beta parameter values."""
        n = torch.tensor([3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        N = torch.tensor([10.0], dtype=torch.float64)

        for alpha_val, beta_val in [(0.5, 1.0), (1.0, 2.0), (2.0, 0.5)]:
            alpha = torch.tensor([alpha_val], dtype=torch.float64)
            beta = torch.tensor([beta_val], dtype=torch.float64)
            result = torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha, beta, N
            )
            expected = torch.tensor(
                [scipy_hahn(3, 2.0, alpha_val, beta_val, 10)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_higher_degree_polynomials(self):
        """Test higher degree polynomials (n > 2)."""
        x = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([1.0], dtype=torch.float64)
        N = torch.tensor([15.0], dtype=torch.float64)

        for n_val in [3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hahn_polynomial_q(
                n, x, alpha, beta, N
            )
            expected = torch.tensor(
                [scipy_hahn(n_val, 2.0, 1.0, 1.0, 15)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
