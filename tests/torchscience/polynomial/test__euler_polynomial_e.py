"""Tests for Euler polynomial series."""

import torch

from torchscience.polynomial._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
    euler_polynomial_e_add,
    euler_polynomial_e_antiderivative,
    euler_polynomial_e_degree,
    euler_polynomial_e_derivative,
    euler_polynomial_e_equal,
    euler_polynomial_e_evaluate,
    euler_polynomial_e_multiply,
    euler_polynomial_e_negate,
    euler_polynomial_e_pow,
    euler_polynomial_e_scale,
    euler_polynomial_e_subtract,
    euler_polynomial_e_to_polynomial,
    euler_polynomial_e_trim,
    polynomial_to_euler_polynomial_e,
)


class TestEulerPolynomialE:
    """Tests for EulerPolynomialE class."""

    def test_creation(self):
        """Test basic creation of Euler polynomial series."""
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        e = euler_polynomial_e(coeffs)
        assert isinstance(e, EulerPolynomialE)
        assert e.shape[-1] == 3

    def test_repr(self):
        """Test string representation."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        assert "EulerPolynomialE" in repr(e)


class TestEulerPolynomialEEvaluate:
    """Tests for Euler polynomial evaluation."""

    def test_e0_is_one(self):
        """E_0(x) = 1 for all x."""
        e = euler_polynomial_e(torch.tensor([1.0, 0.0], dtype=torch.float64))
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = euler_polynomial_e_evaluate(e, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_e1_is_x_minus_half(self):
        """E_1(x) = x - 1/2."""
        e = euler_polynomial_e(torch.tensor([0.0, 1.0], dtype=torch.float64))
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = euler_polynomial_e_evaluate(e, x)
        expected = x - 0.5
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_e2(self):
        """E_2(x) = x^2 - x."""
        e = euler_polynomial_e(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = euler_polynomial_e_evaluate(e, x)
        expected = x**2 - x
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


class TestEulerPolynomialEArithmetic:
    """Tests for Euler polynomial arithmetic operations."""

    def test_add(self):
        """Test addition."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        b = euler_polynomial_e(torch.tensor([3.0, 4.0]))
        c = euler_polynomial_e_add(a, b)
        expected = torch.tensor([4.0, 6.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_subtract(self):
        """Test subtraction."""
        a = euler_polynomial_e(torch.tensor([3.0, 4.0]))
        b = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = euler_polynomial_e_subtract(a, b)
        expected = torch.tensor([2.0, 2.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_negate(self):
        """Test negation."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = euler_polynomial_e_negate(a)
        expected = torch.tensor([-1.0, -2.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_scale(self):
        """Test scaling."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = euler_polynomial_e_scale(a, torch.tensor(3.0))
        expected = torch.tensor([3.0, 6.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_multiply_by_one(self):
        """Multiplying by E_0 (which is 1) should preserve the polynomial."""
        a = euler_polynomial_e(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )  # E_1
        one = euler_polynomial_e(
            torch.tensor([1.0, 0.0], dtype=torch.float64)
        )  # E_0 = 1
        c = euler_polynomial_e_multiply(a, one)
        # Should equal E_1(x) = x - 1/2
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = euler_polynomial_e_evaluate(c, x)
        expected = x - 0.5
        torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-8)

    def test_pow_zero(self):
        """p^0 = 1."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = euler_polynomial_e_pow(a, 0)
        # Result should be E_0
        assert c.shape[-1] == 1
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_pow_one(self):
        """p^1 = p."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = euler_polynomial_e_pow(a, 1)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0]),
            atol=1e-10,
            rtol=1e-10,
        )


class TestEulerPolynomialEDerivative:
    """Tests for Euler polynomial derivative."""

    def test_derivative_e1(self):
        """E'_1(x) = 1 = 1*E_0."""
        e = euler_polynomial_e(torch.tensor([0.0, 1.0], dtype=torch.float64))
        de = euler_polynomial_e_derivative(e)
        # E'_1 = 1 * E_0
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            de.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_derivative_e2(self):
        """E'_2(x) = 2*E_1(x)."""
        e = euler_polynomial_e(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        de = euler_polynomial_e_derivative(e)
        # E'_2 = 2 * E_1, so coeffs should be [0.0, 2.0]
        expected = torch.tensor([0.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(
            de.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_derivative_zero_order(self):
        """Zero-order derivative returns original."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        de = euler_polynomial_e_derivative(e, order=0)
        torch.testing.assert_close(
            de.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0]),
            atol=1e-10,
            rtol=1e-10,
        )


class TestEulerPolynomialEAntiderivative:
    """Tests for Euler polynomial antiderivative."""

    def test_antiderivative_e0(self):
        """Integral of E_0 = 1 is E_1 / 1 = E_1."""
        e = euler_polynomial_e(torch.tensor([1.0], dtype=torch.float64))
        E = euler_polynomial_e_antiderivative(e, constant=0.0)
        # integral(1*E_0) = 1/1 * E_1 = E_1
        expected = torch.tensor([0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            E.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_antiderivative_roundtrip(self):
        """Derivative of antiderivative should give original (up to constant)."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0], dtype=torch.float64))
        E = euler_polynomial_e_antiderivative(e, constant=0.0)
        e_back = euler_polynomial_e_derivative(E)
        # Should match original
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(
            e_back.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )


class TestEulerPolynomialEConversion:
    """Tests for conversion to/from standard polynomial."""

    def test_e0_to_polynomial(self):
        """E_0(x) = 1 should convert to polynomial [1]."""
        e = euler_polynomial_e(torch.tensor([1.0], dtype=torch.float64))
        p = euler_polynomial_e_to_polynomial(e)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_e1_to_polynomial(self):
        """E_1(x) = x - 1/2 should convert to polynomial [-1/2, 1]."""
        e = euler_polynomial_e(torch.tensor([0.0, 1.0], dtype=torch.float64))
        p = euler_polynomial_e_to_polynomial(e)
        expected = torch.tensor([-0.5, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_roundtrip_conversion(self):
        """Converting to polynomial and back should preserve series."""
        from torchscience.polynomial import polynomial

        # Start with a simple polynomial: p(x) = 1 + 2x
        p = polynomial(torch.tensor([1.0, 2.0], dtype=torch.float64))
        e = polynomial_to_euler_polynomial_e(p)
        p_back = euler_polynomial_e_to_polynomial(e)

        torch.testing.assert_close(
            p_back.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )


class TestEulerPolynomialEUtilities:
    """Tests for utility functions."""

    def test_degree(self):
        """Test degree computation."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0, 3.0]))
        assert euler_polynomial_e_degree(e) == 2

    def test_degree_with_trailing_zeros(self):
        """Degree should ignore trailing zeros."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0, 0.0]))
        assert euler_polynomial_e_degree(e) == 1

    def test_trim(self):
        """Test trimming trailing zeros."""
        e = euler_polynomial_e(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        e_trimmed = euler_polynomial_e_trim(e)
        assert e_trimmed.shape[-1] == 2

    def test_equal(self):
        """Test equality check."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        b = euler_polynomial_e(torch.tensor([1.0, 2.0, 0.0]))
        assert euler_polynomial_e_equal(a, b)

    def test_not_equal(self):
        """Test inequality."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        b = euler_polynomial_e(torch.tensor([1.0, 3.0]))
        assert not euler_polynomial_e_equal(a, b)


class TestEulerPolynomialEOperatorOverloading:
    """Tests for operator overloading."""

    def test_add_operator(self):
        """Test + operator."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        b = euler_polynomial_e(torch.tensor([3.0, 4.0]))
        c = a + b
        assert isinstance(c, EulerPolynomialE)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([4.0, 6.0]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_sub_operator(self):
        """Test - operator."""
        a = euler_polynomial_e(torch.tensor([3.0, 4.0]))
        b = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = a - b
        assert isinstance(c, EulerPolynomialE)

    def test_neg_operator(self):
        """Test unary - operator."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = -a
        assert isinstance(c, EulerPolynomialE)

    def test_mul_scalar_operator(self):
        """Test * operator with scalar."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = a * torch.tensor(2.0)
        assert isinstance(c, EulerPolynomialE)

    def test_call_operator(self):
        """Test () operator for evaluation."""
        e = euler_polynomial_e(torch.tensor([1.0, 0.0], dtype=torch.float64))
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = e(x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_pow_operator(self):
        """Test ** operator."""
        a = euler_polynomial_e(torch.tensor([1.0, 2.0]))
        c = a**0
        assert isinstance(c, EulerPolynomialE)
