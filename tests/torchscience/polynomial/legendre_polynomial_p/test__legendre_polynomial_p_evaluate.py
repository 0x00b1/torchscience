# tests/torchscience/polynomial/legendre_polynomial_p/test__legendre_polynomial_p_evaluate.py
import pytest
import scipy.special
import torch

from torchscience.polynomial import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)
from torchscience.polynomial._legendre_polynomial_p._legendre_polynomial_p_evaluate import (
    legendre_polynomial_p_evaluate,
)


class TestLegendrePolynomialPEvaluate:
    def test_evaluate_degree_0(self):
        """P_0(x) = 1"""
        p = legendre_polynomial_p(torch.tensor([2.0]))  # 2 * P_0
        x = torch.tensor([0.0, 0.5, 1.0])
        result = legendre_polynomial_p_evaluate(p, x)
        expected = torch.tensor([2.0, 2.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_evaluate_degree_1(self):
        """P_1(x) = x"""
        p = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        x = torch.tensor([0.0, 0.5, 1.0])
        result = legendre_polynomial_p_evaluate(p, x)
        expected = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(result, expected)

    def test_evaluate_degree_2(self):
        """P_2(x) = (3x^2 - 1) / 2"""
        p = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
        x = torch.tensor([0.0, 1.0, -1.0])
        result = legendre_polynomial_p_evaluate(p, x)
        expected = torch.tensor([-0.5, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_evaluate_vs_scipy(self):
        """Compare against scipy.special.eval_legendre"""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        p = legendre_polynomial_p(coeffs)
        x = torch.linspace(-1, 1, 50)

        result = legendre_polynomial_p_evaluate(p, x)

        # Reference using scipy
        expected = sum(
            c * scipy.special.eval_legendre(k, x.numpy())
            for k, c in enumerate(coeffs.numpy())
        )

        torch.testing.assert_close(result, torch.from_numpy(expected).float())

    def test_evaluate_at_endpoints(self):
        """P_n(1) = 1 and P_n(-1) = (-1)^n for all n"""
        coeffs = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0]
        )  # P_0 + P_1 + P_2 + P_3 + P_4
        p = legendre_polynomial_p(coeffs)

        # At x=1: all P_n(1) = 1, so sum = 5
        result_plus = legendre_polynomial_p_evaluate(p, torch.tensor([1.0]))
        torch.testing.assert_close(result_plus, torch.tensor([5.0]))

        # At x=-1: P_n(-1) = (-1)^n, so sum = 1 - 1 + 1 - 1 + 1 = 1
        result_minus = legendre_polynomial_p_evaluate(p, torch.tensor([-1.0]))
        torch.testing.assert_close(result_minus, torch.tensor([1.0]))

    def test_evaluate_warns_outside_domain(self):
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        x = torch.tensor([2.0])

        with pytest.warns(UserWarning, match="outside natural domain"):
            legendre_polynomial_p_evaluate(p, x)

    def test_evaluate_no_warning_inside_domain(self):
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.5])

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            legendre_polynomial_p_evaluate(p, x)

    def test_evaluate_batched(self):
        """Test batched coefficients"""
        coeffs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # batch of [P_0, P_1]
        p = LegendrePolynomialP(coeffs)
        x = torch.tensor([0.5])

        result = legendre_polynomial_p_evaluate(p, x)
        # First batch: 1*P_0(0.5) = 1
        # Second batch: 1*P_1(0.5) = 0.5
        expected = torch.tensor([[1.0], [0.5]])
        torch.testing.assert_close(result, expected)

    def test_evaluate_callable(self):
        """Test __call__ method"""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.5])
        result = p(x)
        expected = legendre_polynomial_p_evaluate(p, x)
        torch.testing.assert_close(result, expected)

    def test_evaluate_gradient(self):
        """Test autograd compatibility"""
        coeffs = torch.randn(5, dtype=torch.float64, requires_grad=True)
        x = torch.linspace(-0.9, 0.9, 10, dtype=torch.float64)

        def func(c):
            return legendre_polynomial_p_evaluate(LegendrePolynomialP(c), x)

        torch.autograd.gradcheck(func, coeffs, raise_exception=True)
