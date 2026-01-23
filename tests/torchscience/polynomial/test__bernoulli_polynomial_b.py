"""Tests for Bernoulli polynomial series."""

import torch

from torchscience.polynomial._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
    bernoulli_polynomial_b_add,
    bernoulli_polynomial_b_antiderivative,
    bernoulli_polynomial_b_degree,
    bernoulli_polynomial_b_derivative,
    bernoulli_polynomial_b_equal,
    bernoulli_polynomial_b_evaluate,
    bernoulli_polynomial_b_multiply,
    bernoulli_polynomial_b_negate,
    bernoulli_polynomial_b_pow,
    bernoulli_polynomial_b_scale,
    bernoulli_polynomial_b_subtract,
    bernoulli_polynomial_b_to_polynomial,
    bernoulli_polynomial_b_trim,
    polynomial_to_bernoulli_polynomial_b,
)


class TestBernoulliPolynomialB:
    """Tests for BernoulliPolynomialB class."""

    def test_creation(self):
        """Test basic creation of Bernoulli polynomial series."""
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        b = bernoulli_polynomial_b(coeffs)
        assert isinstance(b, BernoulliPolynomialB)
        assert b.shape[-1] == 3

    def test_repr(self):
        """Test string representation."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        assert "BernoulliPolynomialB" in repr(b)


class TestBernoulliPolynomialBEvaluate:
    """Tests for Bernoulli polynomial evaluation."""

    def test_b0_is_one(self):
        """B_0(x) = 1 for all x."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 0.0]))  # 1*B_0 + 0*B_1
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = bernoulli_polynomial_b_evaluate(b.to(torch.float64), x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_b1_is_x_minus_half(self):
        """B_1(x) = x - 1/2."""
        b = bernoulli_polynomial_b(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = bernoulli_polynomial_b_evaluate(b, x)
        expected = x - 0.5
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_b2(self):
        """B_2(x) = x^2 - x + 1/6."""
        b = bernoulli_polynomial_b(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = bernoulli_polynomial_b_evaluate(b, x)
        expected = x**2 - x + 1 / 6
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


class TestBernoulliPolynomialBArithmetic:
    """Tests for Bernoulli polynomial arithmetic operations."""

    def test_add(self):
        """Test addition."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        b = bernoulli_polynomial_b(torch.tensor([3.0, 4.0]))
        c = bernoulli_polynomial_b_add(a, b)
        expected = torch.tensor([4.0, 6.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_subtract(self):
        """Test subtraction."""
        a = bernoulli_polynomial_b(torch.tensor([3.0, 4.0]))
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = bernoulli_polynomial_b_subtract(a, b)
        expected = torch.tensor([2.0, 2.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_negate(self):
        """Test negation."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = bernoulli_polynomial_b_negate(a)
        expected = torch.tensor([-1.0, -2.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_scale(self):
        """Test scaling."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = bernoulli_polynomial_b_scale(a, torch.tensor(3.0))
        expected = torch.tensor([3.0, 6.0])
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_multiply_by_one(self):
        """Multiplying by B_0 (which is 1) should preserve the polynomial."""
        a = bernoulli_polynomial_b(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )  # B_1
        one = bernoulli_polynomial_b(
            torch.tensor([1.0, 0.0], dtype=torch.float64)
        )  # B_0 = 1
        c = bernoulli_polynomial_b_multiply(a, one)
        # Should equal B_1(x) = x - 1/2
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = bernoulli_polynomial_b_evaluate(c, x)
        expected = x - 0.5
        torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-8)

    def test_pow_zero(self):
        """p^0 = 1."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = bernoulli_polynomial_b_pow(a, 0)
        # Result should be B_0
        assert c.shape[-1] == 1
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_pow_one(self):
        """p^1 = p."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = bernoulli_polynomial_b_pow(a, 1)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0]),
            atol=1e-10,
            rtol=1e-10,
        )


class TestBernoulliPolynomialBDerivative:
    """Tests for Bernoulli polynomial derivative."""

    def test_derivative_b1(self):
        """B'_1(x) = 1 = 1*B_0."""
        b = bernoulli_polynomial_b(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )
        db = bernoulli_polynomial_b_derivative(b)
        # B'_1 = 1 * B_0
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            db.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_derivative_b2(self):
        """B'_2(x) = 2*B_1(x)."""
        b = bernoulli_polynomial_b(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        db = bernoulli_polynomial_b_derivative(b)
        # B'_2 = 2 * B_1, so coeffs should be [0.0, 2.0]
        expected = torch.tensor([0.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(
            db.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_derivative_zero_order(self):
        """Zero-order derivative returns original."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        db = bernoulli_polynomial_b_derivative(b, order=0)
        torch.testing.assert_close(
            db.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0]),
            atol=1e-10,
            rtol=1e-10,
        )


class TestBernoulliPolynomialBAntiderivative:
    """Tests for Bernoulli polynomial antiderivative."""

    def test_antiderivative_b0(self):
        """Integral of B_0 = 1 is B_1 / 1 = B_1."""
        b = bernoulli_polynomial_b(torch.tensor([1.0], dtype=torch.float64))
        B = bernoulli_polynomial_b_antiderivative(b, constant=0.0)
        # integral(1*B_0) = 1/1 * B_1 = B_1
        expected = torch.tensor([0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            B.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_antiderivative_roundtrip(self):
        """Derivative of antiderivative should give original (up to constant)."""
        b = bernoulli_polynomial_b(
            torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        B = bernoulli_polynomial_b_antiderivative(b, constant=0.0)
        b_back = bernoulli_polynomial_b_derivative(B)
        # Should match original
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(
            b_back.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )


class TestBernoulliPolynomialBConversion:
    """Tests for conversion to/from standard polynomial."""

    def test_b0_to_polynomial(self):
        """B_0(x) = 1 should convert to polynomial [1]."""
        b = bernoulli_polynomial_b(torch.tensor([1.0], dtype=torch.float64))
        p = bernoulli_polynomial_b_to_polynomial(b)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_b1_to_polynomial(self):
        """B_1(x) = x - 1/2 should convert to polynomial [-1/2, 1]."""
        b = bernoulli_polynomial_b(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )
        p = bernoulli_polynomial_b_to_polynomial(b)
        expected = torch.tensor([-0.5, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), expected, atol=1e-10, rtol=1e-10
        )

    def test_roundtrip_conversion(self):
        """Converting to polynomial and back should preserve series."""
        from torchscience.polynomial import polynomial

        # Start with a simple polynomial: p(x) = 1 + 2x
        p = polynomial(torch.tensor([1.0, 2.0], dtype=torch.float64))
        b = polynomial_to_bernoulli_polynomial_b(p)
        p_back = bernoulli_polynomial_b_to_polynomial(b)

        torch.testing.assert_close(
            p_back.as_subclass(torch.Tensor),
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )


class TestBernoulliPolynomialBUtilities:
    """Tests for utility functions."""

    def test_degree(self):
        """Test degree computation."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 3.0]))
        assert bernoulli_polynomial_b_degree(b) == 2

    def test_degree_with_trailing_zeros(self):
        """Degree should ignore trailing zeros."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0]))
        assert bernoulli_polynomial_b_degree(b) == 1

    def test_trim(self):
        """Test trimming trailing zeros."""
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        b_trimmed = bernoulli_polynomial_b_trim(b)
        assert b_trimmed.shape[-1] == 2

    def test_equal(self):
        """Test equality check."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0]))
        assert bernoulli_polynomial_b_equal(a, b)

    def test_not_equal(self):
        """Test inequality."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        b = bernoulli_polynomial_b(torch.tensor([1.0, 3.0]))
        assert not bernoulli_polynomial_b_equal(a, b)


class TestBernoulliPolynomialBOperatorOverloading:
    """Tests for operator overloading."""

    def test_add_operator(self):
        """Test + operator."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        b = bernoulli_polynomial_b(torch.tensor([3.0, 4.0]))
        c = a + b
        assert isinstance(c, BernoulliPolynomialB)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([4.0, 6.0]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_sub_operator(self):
        """Test - operator."""
        a = bernoulli_polynomial_b(torch.tensor([3.0, 4.0]))
        b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = a - b
        assert isinstance(c, BernoulliPolynomialB)

    def test_neg_operator(self):
        """Test unary - operator."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = -a
        assert isinstance(c, BernoulliPolynomialB)

    def test_mul_scalar_operator(self):
        """Test * operator with scalar."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = a * torch.tensor(2.0)
        assert isinstance(c, BernoulliPolynomialB)

    def test_call_operator(self):
        """Test () operator for evaluation."""
        b = bernoulli_polynomial_b(
            torch.tensor([1.0, 0.0], dtype=torch.float64)
        )
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = b(x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_pow_operator(self):
        """Test ** operator."""
        a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
        c = a**0
        assert isinstance(c, BernoulliPolynomialB)
