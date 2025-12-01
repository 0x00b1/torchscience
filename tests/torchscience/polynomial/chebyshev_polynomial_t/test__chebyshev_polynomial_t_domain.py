# tests/torchscience/polynomial/chebyshev_polynomial_t/test__chebyshev_polynomial_t_domain.py
"""Tests for domain validation in ChebyshevPolynomialT."""

import warnings

import pytest
import torch

from torchscience.polynomial import (
    ChebyshevPolynomialT,
    DomainError,
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_fit,
    chebyshev_polynomial_t_integral,
    chebyshev_polynomial_t_weight,
)


class TestChebyshevPolynomialTDomain:
    """Tests for domain validation in ChebyshevPolynomialT operations."""

    def test_evaluate_warns_outside_domain(self):
        """Evaluating outside [-1, 1] should warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0]))
        x = torch.tensor([2.0])  # Outside [-1, 1]

        with pytest.warns(UserWarning, match="outside natural domain"):
            chebyshev_polynomial_t_evaluate(p, x)

    def test_evaluate_warns_negative_outside_domain(self):
        """Evaluating at negative value outside [-1, 1] should warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0]))
        x = torch.tensor([-2.0])  # Outside [-1, 1]

        with pytest.warns(UserWarning, match="outside natural domain"):
            chebyshev_polynomial_t_evaluate(p, x)

    def test_evaluate_warns_mixed_values(self):
        """Evaluating with some values outside [-1, 1] should warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.5, 1.5])  # One inside, one outside

        with pytest.warns(UserWarning, match="outside natural domain"):
            chebyshev_polynomial_t_evaluate(p, x)

    def test_evaluate_no_warning_inside_domain(self):
        """Evaluating inside [-1, 1] should not warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.5])  # Inside [-1, 1]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_evaluate(p, x)  # Should not warn

    def test_evaluate_no_warning_at_boundary(self):
        """Evaluating at domain boundaries [-1, 1] should not warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0]))
        x = torch.tensor([-1.0, 1.0])  # At boundaries

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_evaluate(p, x)  # Should not warn

    def test_evaluate_no_warning_all_inside(self):
        """Evaluating with all values inside [-1, 1] should not warn."""
        p = ChebyshevPolynomialT(coeffs=torch.tensor([1.0, 2.0, 3.0]))
        x = torch.linspace(-1, 1, 10)  # All inside

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_evaluate(p, x)  # Should not warn

    def test_fit_errors_outside_domain(self):
        """Fitting with points outside [-1, 1] should raise DomainError."""
        x = torch.tensor([2.0, 3.0])  # Outside [-1, 1]
        y = torch.tensor([1.0, 2.0])

        with pytest.raises(DomainError):
            chebyshev_polynomial_t_fit(x, y, degree=3)

    def test_fit_errors_negative_outside_domain(self):
        """Fitting with negative points outside [-1, 1] should raise DomainError."""
        x = torch.tensor([-2.0, -3.0])  # Outside [-1, 1]
        y = torch.tensor([1.0, 2.0])

        with pytest.raises(DomainError):
            chebyshev_polynomial_t_fit(x, y, degree=3)

    def test_fit_errors_mixed_values(self):
        """Fitting with some points outside [-1, 1] should raise DomainError."""
        x = torch.tensor([0.5, 1.5])  # One inside, one outside
        y = torch.tensor([1.0, 2.0])

        with pytest.raises(DomainError):
            chebyshev_polynomial_t_fit(x, y, degree=1)

    def test_fit_succeeds_inside_domain(self):
        """Fitting with all points inside [-1, 1] should succeed."""
        x = torch.tensor([-0.5, 0.0, 0.5])  # Inside [-1, 1]
        y = torch.tensor([1.0, 2.0, 3.0])

        # Should not raise
        result = chebyshev_polynomial_t_fit(x, y, degree=2)
        assert result is not None
        assert result.coeffs.shape[-1] == 3  # degree + 1 coefficients

    def test_fit_succeeds_at_boundaries(self):
        """Fitting with points at domain boundaries should succeed."""
        x = torch.tensor([-1.0, 0.0, 1.0])  # At boundaries
        y = torch.tensor([1.0, 0.0, 1.0])

        # Should not raise
        result = chebyshev_polynomial_t_fit(x, y, degree=2)
        assert result is not None

    def test_domain_constant_value(self):
        """Verify DOMAIN constant has correct value."""
        assert ChebyshevPolynomialT.DOMAIN == (-1.0, 1.0)


class TestChebyshevPolynomialTIntegralDomain:
    """Tests for domain validation in chebyshev_polynomial_t_integral."""

    def test_integral_warns_outside_domain(self):
        """Integration bounds outside [-1, 1] should warn."""
        p = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))

        with pytest.warns(UserWarning, match="Integration bounds"):
            chebyshev_polynomial_t_integral(
                p, lower=torch.tensor(-2.0), upper=torch.tensor(2.0)
            )

    def test_integral_warns_lower_outside_domain(self):
        """Integration lower bound outside [-1, 1] should warn."""
        p = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))

        with pytest.warns(UserWarning, match="Integration bounds"):
            chebyshev_polynomial_t_integral(
                p, lower=torch.tensor(-2.0), upper=torch.tensor(0.5)
            )

    def test_integral_warns_upper_outside_domain(self):
        """Integration upper bound outside [-1, 1] should warn."""
        p = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))

        with pytest.warns(UserWarning, match="Integration bounds"):
            chebyshev_polynomial_t_integral(
                p, lower=torch.tensor(-0.5), upper=torch.tensor(2.0)
            )

    def test_integral_no_warning_inside_domain(self):
        """Integration inside [-1, 1] should not warn."""
        p = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_integral(
                p, lower=torch.tensor(-1.0), upper=torch.tensor(1.0)
            )

    def test_integral_no_warning_partial_domain(self):
        """Integration over partial domain should not warn."""
        p = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_integral(
                p, lower=torch.tensor(-0.5), upper=torch.tensor(0.5)
            )


class TestChebyshevPolynomialTWeightDomain:
    """Tests for domain validation in chebyshev_polynomial_t_weight."""

    def test_weight_warns_outside_domain(self):
        """Evaluating weight outside [-1, 1] should warn."""
        x = torch.tensor([2.0])  # Outside [-1, 1]

        with pytest.warns(UserWarning, match="weight function outside"):
            chebyshev_polynomial_t_weight(x)

    def test_weight_warns_negative_outside_domain(self):
        """Evaluating weight at negative value outside [-1, 1] should warn."""
        x = torch.tensor([-2.0])  # Outside [-1, 1]

        with pytest.warns(UserWarning, match="weight function outside"):
            chebyshev_polynomial_t_weight(x)

    def test_weight_warns_mixed_values(self):
        """Evaluating weight with some values outside [-1, 1] should warn."""
        x = torch.tensor([0.5, 1.5])  # One inside, one outside

        with pytest.warns(UserWarning, match="weight function outside"):
            chebyshev_polynomial_t_weight(x)

    def test_weight_no_warning_inside_domain(self):
        """Evaluating weight inside (-1, 1) should not warn."""
        x = torch.tensor([0.5])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_weight(x)

    def test_weight_no_warning_all_inside(self):
        """Evaluating weight with all values inside (-1, 1) should not warn."""
        x = torch.linspace(-0.9, 0.9, 10)  # All inside, avoiding boundary

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chebyshev_polynomial_t_weight(x)
