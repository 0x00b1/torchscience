"""Tests for polynomial exception hierarchy."""

import pytest

from torchscience.polynomial import (
    DegreeError,
    DomainError,
    ParameterError,
    ParameterMismatchError,
    PolynomialError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from PolynomialError."""

    def test_domain_error_is_polynomial_error(self):
        with pytest.raises(PolynomialError):
            raise DomainError("test")

    def test_parameter_error_is_polynomial_error(self):
        with pytest.raises(PolynomialError):
            raise ParameterError("test")

    def test_parameter_mismatch_error_is_polynomial_error(self):
        with pytest.raises(PolynomialError):
            raise ParameterMismatchError("test")

    def test_degree_error_is_polynomial_error(self):
        with pytest.raises(PolynomialError):
            raise DegreeError("test")


class TestExceptionMessages:
    """Test that exceptions preserve their messages."""

    def test_domain_error_message(self):
        with pytest.raises(DomainError, match="outside valid domain"):
            raise DomainError("Value outside valid domain")

    def test_parameter_error_message(self):
        with pytest.raises(ParameterError, match="alpha must be"):
            raise ParameterError("alpha must be greater than -1")

    def test_parameter_mismatch_error_message(self):
        with pytest.raises(
            ParameterMismatchError, match="different parameters"
        ):
            raise ParameterMismatchError(
                "Cannot add polynomials with different parameters"
            )
