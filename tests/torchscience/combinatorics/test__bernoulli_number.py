"""Tests for Bernoulli numbers."""

from fractions import Fraction

import pytest
import torch

from torchscience.combinatorics._bernoulli_number import (
    _bernoulli_number_exact,
    bernoulli_number,
    bernoulli_number_all,
)


class TestBernoulliNumberExact:
    """Tests for exact Bernoulli number computation."""

    def test_b0(self):
        assert _bernoulli_number_exact(0) == Fraction(1)

    def test_b1(self):
        assert _bernoulli_number_exact(1) == Fraction(-1, 2)

    def test_b2(self):
        assert _bernoulli_number_exact(2) == Fraction(1, 6)

    def test_b4(self):
        assert _bernoulli_number_exact(4) == Fraction(-1, 30)

    def test_b6(self):
        assert _bernoulli_number_exact(6) == Fraction(1, 42)

    def test_b8(self):
        assert _bernoulli_number_exact(8) == Fraction(-1, 30)

    def test_b10(self):
        assert _bernoulli_number_exact(10) == Fraction(5, 66)

    def test_odd_bernoulli_numbers_are_zero(self):
        """B_n = 0 for odd n >= 3."""
        for n in [3, 5, 7, 9, 11, 13]:
            assert _bernoulli_number_exact(n) == Fraction(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _bernoulli_number_exact(-1)


class TestBernoulliNumber:
    """Tests for bernoulli_number function."""

    def test_b0(self):
        result = bernoulli_number(0)
        torch.testing.assert_close(
            result, torch.tensor(1.0, dtype=torch.float64)
        )

    def test_b1(self):
        result = bernoulli_number(1)
        torch.testing.assert_close(
            result, torch.tensor(-0.5, dtype=torch.float64)
        )

    def test_b2(self):
        result = bernoulli_number(2)
        torch.testing.assert_close(
            result, torch.tensor(1.0 / 6.0, dtype=torch.float64)
        )

    def test_dtype(self):
        result = bernoulli_number(2, dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_device(self):
        result = bernoulli_number(2, device="cpu")
        assert result.device == torch.device("cpu")


class TestBernoulliNumberAll:
    """Tests for bernoulli_number_all function."""

    def test_first_seven(self):
        result = bernoulli_number_all(6)
        expected = torch.tensor(
            [1.0, -0.5, 1 / 6, 0.0, -1 / 30, 0.0, 1 / 42], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected)

    def test_shape(self):
        result = bernoulli_number_all(10)
        assert result.shape == (11,)

    def test_dtype(self):
        result = bernoulli_number_all(5, dtype=torch.float32)
        assert result.dtype == torch.float32
