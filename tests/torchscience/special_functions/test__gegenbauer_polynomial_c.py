import pytest
import scipy.special
import torch

import torchscience.special_functions


def scipy_gegenbauer_c(n: float, lam: float, z: float) -> float:
    """Reference using scipy.special.eval_gegenbauer."""
    return float(scipy.special.eval_gegenbauer(n, lam, z))


class TestGegenbauerPolynomialC:
    """Tests for Gegenbauer polynomial C_n^lambda(z)."""

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy for integer orders."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        for n in range(11):
            for lam in [0.5, 1.0, 1.5, 2.0]:
                n_t = torch.tensor([float(n)], dtype=torch.float64)
                lam_t = torch.tensor([lam], dtype=torch.float64)
                result = (
                    torchscience.special_functions.gegenbauer_polynomial_c(
                        n_t, lam_t, z
                    )
                )
                expected = torch.tensor(
                    [scipy_gegenbauer_c(n, lam, float(zi)) for zi in z],
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-8, atol=1e-8
                )

    def test_scipy_agreement_noninteger_orders(self):
        """Test agreement with scipy for non-integer orders."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7], dtype=torch.float64)
        for n in [0.5, 1.5, 2.5, 3.7]:
            for lam in [0.5, 1.0, 1.5]:
                n_t = torch.tensor([n], dtype=torch.float64)
                lam_t = torch.tensor([lam], dtype=torch.float64)
                result = (
                    torchscience.special_functions.gegenbauer_polynomial_c(
                        n_t, lam_t, z
                    )
                )
                expected = torch.tensor(
                    [scipy_gegenbauer_c(n, lam, float(zi)) for zi in z],
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-5, atol=1e-5
                )

    def test_special_values(self):
        """Test special values of Gegenbauer polynomials."""
        # C_0^lambda(z) = 1
        n = torch.tensor([0.0], dtype=torch.float64)
        lam = torch.tensor([1.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

        # C_1^lambda(z) = 2*lambda*z
        n = torch.tensor([1.0], dtype=torch.float64)
        lam = torch.tensor([1.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        expected = torch.tensor([2 * 1.5 * 0.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_legendre_relation(self):
        """Test that C_n^(1/2)(z) = P_n(z) (Legendre polynomial)."""
        z = torch.tensor([0.0, 0.3, 0.5, 0.7], dtype=torch.float64)
        lam = torch.tensor([0.5], dtype=torch.float64)
        for n in range(6):
            n_t = torch.tensor([float(n)], dtype=torch.float64)
            gegenbauer = (
                torchscience.special_functions.gegenbauer_polynomial_c(
                    n_t, lam, z
                )
            )
            legendre = torchscience.special_functions.legendre_polynomial_p(
                n_t, z
            )
            torch.testing.assert_close(
                gegenbauer, legendre, rtol=1e-8, atol=1e-8
            )

    def test_recurrence_relation(self):
        """Test recurrence: n*C_n = 2(n+lambda-1)*z*C_{n-1} - (n+2*lambda-2)*C_{n-2}."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        lam = torch.tensor([1.5], dtype=torch.float64)

        for n in range(2, 10):
            n_f = float(n)
            n_t = torch.tensor([n_f], dtype=torch.float64)
            n_t_1 = torch.tensor([n_f - 1], dtype=torch.float64)
            n_t_2 = torch.tensor([n_f - 2], dtype=torch.float64)

            C_n = torchscience.special_functions.gegenbauer_polynomial_c(
                n_t, lam, z
            )
            C_n1 = torchscience.special_functions.gegenbauer_polynomial_c(
                n_t_1, lam, z
            )
            C_n2 = torchscience.special_functions.gegenbauer_polynomial_c(
                n_t_2, lam, z
            )

            lam_val = 1.5
            left = n_f * C_n
            right = (
                2 * (n_f + lam_val - 1) * z * C_n1
                - (n_f + 2 * lam_val - 2) * C_n2
            )
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-8)

    def test_gradcheck(self):
        """Test gradients with torch.autograd.gradcheck."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        lam = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            torchscience.special_functions.gegenbauer_polynomial_c,
            (n, lam, z),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_complex_input(self):
        """Test with complex inputs."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        lam = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        # For real n, lambda, z the result should be real (imaginary part ~0)
        assert result.is_complex()
        expected_real = scipy_gegenbauer_c(2.0, 1.0, 0.5)
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
        lam = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        lam = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        expected = scipy_gegenbauer_c(2.0, 1.0, 0.5)
        torch.testing.assert_close(
            result.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        n = torch.tensor([2.0], dtype=torch.float64, device="meta")
        lam = torch.tensor([1.0], dtype=torch.float64, device="meta")
        z = torch.tensor([0.5], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_negative_n(self):
        """Test with negative n values."""
        n = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64)
        lam = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        # Should not raise an error
        result = torchscience.special_functions.gegenbauer_polynomial_c(
            n, lam, z
        )
        assert result.shape == (3,)
