import pytest
import torch

from torchscience.polynomial import (
    LegendrePolynomialP,
    PolynomialError,
    legendre_polynomial_p,
)


class TestLegendrePolynomialP:
    def test_create_from_coeffs(self):
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        p = LegendrePolynomialP(coeffs)
        torch.testing.assert_close(p.as_subclass(torch.Tensor), coeffs)

    def test_constructor_function(self):
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        p = legendre_polynomial_p(coeffs)
        torch.testing.assert_close(p.as_subclass(torch.Tensor), coeffs)

    def test_domain_constant(self):
        assert LegendrePolynomialP.DOMAIN == (-1.0, 1.0)

    def test_empty_coeffs_raises(self):
        with pytest.raises(PolynomialError):
            legendre_polynomial_p(torch.tensor([]))

    def test_batched_coeffs(self):
        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # batch of 2
        p = LegendrePolynomialP(coeffs)
        assert p.shape == (2, 2)
