import pytest
import scipy.special
import torch

import torchscience.special_functions


def scipy_jacobi_p(n: float, alpha: float, beta: float, z: float) -> float:
    """Reference using scipy.special.eval_jacobi."""
    return float(scipy.special.eval_jacobi(n, alpha, beta, z))


class TestJacobiPolynomialP:
    """Tests for Jacobi polynomial P_n^(alpha,beta)(z)."""

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy for integer orders."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        for n in range(11):
            for alpha in [0.0, 0.5, 1.0]:
                for beta in [0.0, 0.5, 1.0]:
                    n_t = torch.tensor([float(n)], dtype=torch.float64)
                    alpha_t = torch.tensor([alpha], dtype=torch.float64)
                    beta_t = torch.tensor([beta], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.jacobi_polynomial_p(
                            n_t, alpha_t, beta_t, z
                        )
                    )
                    expected = torch.tensor(
                        [
                            scipy_jacobi_p(n, alpha, beta, float(zi))
                            for zi in z
                        ],
                        dtype=torch.float64,
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-8, atol=1e-8
                    )

    def test_scipy_agreement_noninteger_orders(self):
        """Test agreement with scipy for non-integer orders."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7], dtype=torch.float64)
        for n in [0.5, 1.5, 2.5]:
            for alpha in [0.5, 1.0]:
                for beta in [0.5, 1.0]:
                    n_t = torch.tensor([n], dtype=torch.float64)
                    alpha_t = torch.tensor([alpha], dtype=torch.float64)
                    beta_t = torch.tensor([beta], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.jacobi_polynomial_p(
                            n_t, alpha_t, beta_t, z
                        )
                    )
                    expected = torch.tensor(
                        [
                            scipy_jacobi_p(n, alpha, beta, float(zi))
                            for zi in z
                        ],
                        dtype=torch.float64,
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-5, atol=1e-5
                    )

    def test_special_values(self):
        """Test special values of Jacobi polynomials."""
        # P_0^(alpha,beta)(z) = 1
        n = torch.tensor([0.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

        # P_1^(alpha,beta)(z) = (alpha - beta)/2 + (alpha + beta + 2)*z/2
        n = torch.tensor([1.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        expected_val = (1.0 - 0.5) / 2 + (1.0 + 0.5 + 2) * 0.5 / 2
        expected = torch.tensor([expected_val], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_legendre_relation(self):
        """Test that P_n^(0,0)(z) = P_n(z) (Legendre polynomial)."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7], dtype=torch.float64)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        beta = torch.tensor([0.0], dtype=torch.float64)
        for n in range(6):
            n_t = torch.tensor([float(n)], dtype=torch.float64)
            jacobi = torchscience.special_functions.jacobi_polynomial_p(
                n_t, alpha, beta, z
            )
            legendre = torchscience.special_functions.legendre_polynomial_p(
                n_t, z
            )
            torch.testing.assert_close(jacobi, legendre, rtol=1e-8, atol=1e-8)

    def test_gradcheck(self):
        """Test gradients with torch.autograd.gradcheck."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        beta = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            torchscience.special_functions.jacobi_polynomial_p,
            (n, alpha, beta, z),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_complex_input(self):
        """Test with complex inputs."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        alpha = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        beta = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        # For real n, alpha, beta, z the result should be real (imaginary part ~0)
        assert result.is_complex()
        expected_real = scipy_jacobi_p(2.0, 0.5, 0.5, 0.5)
        torch.testing.assert_close(
            result.real,
            torch.tensor([expected_real], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        beta = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        alpha = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        beta = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        expected = scipy_jacobi_p(2.0, 0.5, 0.5, 0.5)
        torch.testing.assert_close(
            result.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        n = torch.tensor([2.0], dtype=torch.float64, device="meta")
        alpha = torch.tensor([0.5], dtype=torch.float64, device="meta")
        beta = torch.tensor([0.5], dtype=torch.float64, device="meta")
        z = torch.tensor([0.5], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_gegenbauer_relation(self):
        """Test that Jacobi with alpha=beta relates to Gegenbauer."""
        # P_n^(lambda-1/2, lambda-1/2)(z) = const * C_n^lambda(z)
        # For lambda = 1 (so alpha = beta = 0.5):
        # P_n^(0.5, 0.5)(z) = Gamma(n+1.5) / (Gamma(1.5) * Gamma(n+1)) * C_n^1(z) / (Gamma(n+2) / (Gamma(2) * Gamma(n+1)))
        # This is more complex, so we just check numerical agreement with scipy
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        for n in range(5):
            n_t = torch.tensor([float(n)], dtype=torch.float64)
            alpha = torch.tensor([0.5], dtype=torch.float64)
            beta = torch.tensor([0.5], dtype=torch.float64)

            jacobi = torchscience.special_functions.jacobi_polynomial_p(
                n_t, alpha, beta, z
            )
            # Just verify against scipy
            expected = torch.tensor(
                [scipy_jacobi_p(n, 0.5, 0.5, float(zi)) for zi in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(jacobi, expected, rtol=1e-8, atol=1e-8)

    def test_negative_n(self):
        """Test with negative n values."""
        n = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        beta = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        # Should not raise an error
        result = torchscience.special_functions.jacobi_polynomial_p(
            n, alpha, beta, z
        )
        assert result.shape == (3,)
