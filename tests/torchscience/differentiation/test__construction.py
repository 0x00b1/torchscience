"""Tests for stencil construction functions."""

import pytest
import torch

from torchscience.differentiation import (
    biharmonic_stencil,
    finite_difference_stencil,
    gradient_stencils,
    laplacian_stencil,
)


class TestFornbergAlgorithm:
    """Tests for Fornberg's algorithm implementation."""

    def test_first_derivative_central_accuracy_2(self):
        """Central first derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=2, kind="central"
        )
        # Should be [-1/2, 0, 1/2]
        expected_coeffs = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        expected_offsets = torch.tensor([[-1], [0], [1]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_second_derivative_central_accuracy_2(self):
        """Central second derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=2, accuracy=2, kind="central"
        )
        # Should be [1, -2, 1]
        expected_coeffs = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_first_derivative_forward_accuracy_1(self):
        """Forward first derivative with 1st order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=1, kind="forward"
        )
        # Should be [-1, 1]
        expected_coeffs = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        expected_offsets = torch.tensor([[0], [1]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_first_derivative_backward_accuracy_1(self):
        """Backward first derivative with 1st order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=1, kind="backward"
        )
        # Should be [-1, 1] with offsets [-1, 0]
        expected_coeffs = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        expected_offsets = torch.tensor([[-1], [0]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_fourth_derivative_central_accuracy_2(self):
        """Central fourth derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=4, accuracy=2, kind="central"
        )
        # Should be [1, -4, 6, -4, 1]
        expected_coeffs = torch.tensor(
            [1.0, -4.0, 6.0, -4.0, 1.0], dtype=torch.float64
        )
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_higher_order_accuracy(self):
        """Central first derivative with 4th order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=4, kind="central"
        )
        # Should be [1/12, -2/3, 0, 2/3, -1/12]
        expected_coeffs = torch.tensor(
            [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=torch.float64
        )
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_derivative_tuple_1d(self):
        """Derivative specified as tuple for 1D."""
        stencil = finite_difference_stencil(
            derivative=(2,), accuracy=2, kind="central"
        )
        assert stencil.derivative == (2,)
        assert stencil.ndim == 1

    def test_derivative_tuple_2d_mixed(self):
        """Mixed partial derivative d^2/dxdy."""
        stencil = finite_difference_stencil(
            derivative=(1, 1), accuracy=2, kind="central"
        )
        assert stencil.derivative == (1, 1)
        assert stencil.ndim == 2
        # Mixed partial should have 4 corners
        assert stencil.n_points == 4

    def test_derivative_tuple_2d_second_x(self):
        """Second derivative in x: d^2/dx^2."""
        stencil = finite_difference_stencil(
            derivative=(2, 0), accuracy=2, kind="central"
        )
        assert stencil.derivative == (2, 0)
        assert stencil.ndim == 2
        # Should only vary in first dimension
        assert torch.all(stencil.offsets[:, 1] == 0)


class TestStencilValidation:
    """Tests for stencil validation."""

    def test_invalid_kind_raises(self):
        """Invalid kind raises error."""
        with pytest.raises(ValueError, match="kind"):
            finite_difference_stencil(derivative=1, accuracy=2, kind="invalid")

    def test_zero_derivative_raises(self):
        """Zero derivative order raises error."""
        with pytest.raises(ValueError, match="derivative"):
            finite_difference_stencil(derivative=0, accuracy=2, kind="central")

    def test_zero_accuracy_raises(self):
        """Zero accuracy raises error."""
        with pytest.raises(ValueError, match="accuracy"):
            finite_difference_stencil(derivative=1, accuracy=0, kind="central")

    def test_negative_derivative_raises(self):
        """Negative derivative raises error."""
        with pytest.raises(ValueError, match="derivative"):
            finite_difference_stencil(
                derivative=-1, accuracy=2, kind="central"
            )


class TestPrebuiltStencils:
    """Tests for pre-built stencil functions."""

    def test_laplacian_stencil_2d(self):
        """2D Laplacian stencil."""

        stencil = laplacian_stencil(ndim=2, accuracy=2)
        assert stencil.ndim == 2
        # 5-point stencil for 2D Laplacian
        assert stencil.n_points == 5

    def test_laplacian_stencil_3d(self):
        """3D Laplacian stencil."""
        from torchscience.differentiation import laplacian_stencil

        stencil = laplacian_stencil(ndim=3, accuracy=2)
        assert stencil.ndim == 3
        # 7-point stencil for 3D Laplacian
        assert stencil.n_points == 7

    def test_laplacian_stencil_1d(self):
        """1D Laplacian stencil (just second derivative)."""
        from torchscience.differentiation import laplacian_stencil

        stencil = laplacian_stencil(ndim=1, accuracy=2)
        assert stencil.ndim == 1
        # 3-point stencil for 1D Laplacian
        assert stencil.n_points == 3
        # Coefficients should be [1, -2, 1]
        expected_coeffs = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_laplacian_stencil_2d_values(self):
        """2D Laplacian stencil has correct coefficients."""
        from torchscience.differentiation import laplacian_stencil

        stencil = laplacian_stencil(ndim=2, accuracy=2)
        # Center should have coefficient -4, and 4 neighbors should have +1
        assert stencil.coeffs.sum().abs() < 1e-10  # Should sum to 0

    def test_gradient_stencils_2d(self):
        """2D gradient stencils."""

        stencils = gradient_stencils(ndim=2, accuracy=2)
        assert len(stencils) == 2
        assert stencils[0].derivative == (1, 0)
        assert stencils[1].derivative == (0, 1)

    def test_gradient_stencils_3d(self):
        """3D gradient stencils."""
        from torchscience.differentiation import gradient_stencils

        stencils = gradient_stencils(ndim=3, accuracy=2)
        assert len(stencils) == 3
        assert stencils[0].derivative == (1, 0, 0)
        assert stencils[1].derivative == (0, 1, 0)
        assert stencils[2].derivative == (0, 0, 1)

    def test_gradient_stencils_1d(self):
        """1D gradient stencils."""
        from torchscience.differentiation import gradient_stencils

        stencils = gradient_stencils(ndim=1, accuracy=2)
        assert len(stencils) == 1
        assert stencils[0].derivative == (1,)

    def test_gradient_stencils_forward(self):
        """Gradient stencils with forward difference."""
        from torchscience.differentiation import gradient_stencils

        stencils = gradient_stencils(ndim=2, accuracy=1, kind="forward")
        assert len(stencils) == 2
        # Forward stencil should have offsets starting at 0
        assert all(stencil.offsets.min() >= 0 for stencil in stencils)

    def test_biharmonic_stencil_2d(self):
        """2D biharmonic stencil."""

        stencil = biharmonic_stencil(ndim=2, accuracy=2)
        assert stencil.ndim == 2
        # Biharmonic is 13-point in 2D
        assert stencil.n_points == 13

    def test_biharmonic_stencil_1d(self):
        """1D biharmonic stencil."""
        from torchscience.differentiation import biharmonic_stencil

        stencil = biharmonic_stencil(ndim=1, accuracy=2)
        assert stencil.ndim == 1
        # Biharmonic in 1D is fourth derivative: 5 points
        assert stencil.n_points == 5
        # Coefficients should be [1, -4, 6, -4, 1]
        expected_coeffs = torch.tensor(
            [1.0, -4.0, 6.0, -4.0, 1.0], dtype=torch.float64
        )
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_biharmonic_stencil_2d_values(self):
        """2D biharmonic stencil sums to zero."""
        from torchscience.differentiation import biharmonic_stencil

        stencil = biharmonic_stencil(ndim=2, accuracy=2)
        # Biharmonic stencil coefficients should sum to 0
        assert stencil.coeffs.sum().abs() < 1e-10
