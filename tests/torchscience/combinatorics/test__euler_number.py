"""Tests for Euler numbers."""

import pytest
import torch

from torchscience.combinatorics._euler_number import (
    _euler_number_exact,
    euler_number,
    euler_number_all,
)


class TestEulerNumberExact:
    """Tests for exact Euler number computation."""

    def test_e0(self):
        assert _euler_number_exact(0) == 1

    def test_e2(self):
        assert _euler_number_exact(2) == -1

    def test_e4(self):
        assert _euler_number_exact(4) == 5

    def test_e6(self):
        assert _euler_number_exact(6) == -61

    def test_e8(self):
        assert _euler_number_exact(8) == 1385

    def test_e10(self):
        assert _euler_number_exact(10) == -50521

    def test_odd_euler_numbers_are_zero(self):
        """E_n = 0 for odd n."""
        for n in [1, 3, 5, 7, 9, 11, 13]:
            assert _euler_number_exact(n) == 0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _euler_number_exact(-1)


class TestEulerNumber:
    """Tests for euler_number function."""

    def test_e0(self):
        result = euler_number(0)
        torch.testing.assert_close(
            result, torch.tensor(1.0, dtype=torch.float64)
        )

    def test_e2(self):
        result = euler_number(2)
        torch.testing.assert_close(
            result, torch.tensor(-1.0, dtype=torch.float64)
        )

    def test_e4(self):
        result = euler_number(4)
        torch.testing.assert_close(
            result, torch.tensor(5.0, dtype=torch.float64)
        )

    def test_e6(self):
        result = euler_number(6)
        torch.testing.assert_close(
            result, torch.tensor(-61.0, dtype=torch.float64)
        )

    def test_dtype(self):
        result = euler_number(2, dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_device(self):
        result = euler_number(2, device="cpu")
        assert result.device == torch.device("cpu")


class TestEulerNumberAll:
    """Tests for euler_number_all function."""

    def test_first_nine(self):
        result = euler_number_all(8)
        expected = torch.tensor(
            [1.0, 0.0, -1.0, 0.0, 5.0, 0.0, -61.0, 0.0, 1385.0],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected)

    def test_shape(self):
        result = euler_number_all(10)
        assert result.shape == (11,)

    def test_dtype(self):
        result = euler_number_all(5, dtype=torch.float32)
        assert result.dtype == torch.float32
