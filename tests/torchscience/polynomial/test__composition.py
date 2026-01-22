"""Tests for polynomial composition."""

import torch

from torchscience.polynomial import (
    polynomial,
    polynomial_compose,
    polynomial_equal,
    polynomial_evaluate,
)


class TestPolynomialCompose:
    """Tests for polynomial_compose."""

    def test_compose_identity(self):
        """p(x) composed with x gives p(x)."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
        x = polynomial(torch.tensor([0.0, 1.0]))  # x

        result = polynomial_compose(p, x)

        assert polynomial_equal(result, p, tol=1e-6)

    def test_compose_constant(self):
        """p(c) where c is constant gives constant polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
        c = polynomial(torch.tensor([2.0]))  # constant 2

        result = polynomial_compose(p, c)

        # p(2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        expected = polynomial(torch.tensor([17.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_compose_square(self):
        """(x^2) composed with (x + 1) gives x^2 + 2x + 1."""
        p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
        q = polynomial(torch.tensor([1.0, 1.0]))  # x + 1

        result = polynomial_compose(p, q)

        # (x+1)^2 = x^2 + 2x + 1
        expected = polynomial(torch.tensor([1.0, 2.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_compose_linear_in_quadratic(self):
        """(2x + 3) composed with (x^2) gives 2x^2 + 3."""
        p = polynomial(torch.tensor([3.0, 2.0]))  # 3 + 2x
        q = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2

        result = polynomial_compose(p, q)

        # 3 + 2*(x^2) = 3 + 2x^2
        expected = polynomial(torch.tensor([3.0, 0.0, 2.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_compose_cubic(self):
        """Complex composition test."""
        p = polynomial(torch.tensor([1.0, 1.0, 1.0]))  # 1 + x + x^2
        q = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x

        result = polynomial_compose(p, q)

        # p(1+x) = 1 + (1+x) + (1+x)^2
        #        = 1 + 1 + x + 1 + 2x + x^2
        #        = 3 + 3x + x^2
        expected = polynomial(torch.tensor([3.0, 3.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_compose_evaluation_equivalence(self):
        """p(q(x)) evaluated at x should equal p evaluated at q(x)."""
        p = polynomial(torch.tensor([1.0, -2.0, 3.0]))
        q = polynomial(torch.tensor([2.0, 1.0, -1.0]))

        composed = polynomial_compose(p, q)

        # Test at several points
        x_vals = torch.tensor([0.0, 1.0, -1.0, 2.0])
        q_of_x = polynomial_evaluate(q, x_vals)
        p_of_q_of_x = polynomial_evaluate(p, q_of_x)
        composed_of_x = polynomial_evaluate(composed, x_vals)

        assert torch.allclose(p_of_q_of_x, composed_of_x, atol=1e-5)


class TestComposeAutograd:
    """Tests for composition autograd."""

    def test_compose_gradcheck(self):
        """Verify gradients through polynomial_compose."""
        p_coeffs = torch.tensor(
            [1.0, 2.0], requires_grad=True, dtype=torch.float64
        )
        q_coeffs = torch.tensor(
            [1.0, 1.0], requires_grad=True, dtype=torch.float64
        )

        def compose_sum(p_c, q_c):
            p = polynomial(p_c)
            q = polynomial(q_c)
            result = polynomial_compose(p, q)
            return result.sum()

        assert torch.autograd.gradcheck(
            compose_sum, (p_coeffs, q_coeffs), eps=1e-6
        )

    def test_compose_gradgradcheck(self):
        """Verify second-order gradients through polynomial_compose."""
        p_coeffs = torch.tensor(
            [1.0, 2.0], requires_grad=True, dtype=torch.float64
        )
        q_coeffs = torch.tensor(
            [1.0, 1.0], requires_grad=True, dtype=torch.float64
        )

        def compose_sum(p_c, q_c):
            p = polynomial(p_c)
            q = polynomial(q_c)
            result = polynomial_compose(p, q)
            return result.sum()

        assert torch.autograd.gradgradcheck(
            compose_sum, (p_coeffs, q_coeffs), eps=1e-6
        )


class TestComposeBatched:
    """Tests for batched composition."""

    def test_batched_compose(self):
        """Composition with batch dimensions."""
        # Batch of 2 polynomials for p
        p_coeffs = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # 1+x, 2+2x
        p = polynomial(p_coeffs)
        q = polynomial(torch.tensor([1.0, 1.0]))  # 1+x

        result = polynomial_compose(p, q)

        # First: (1+x) composed with (1+x) = 1 + (1+x) = 2 + x
        # Second: (2+2x) composed with (1+x) = 2 + 2(1+x) = 4 + 2x
        assert result.shape[0] == 2

        # Verify values
        expected_0 = polynomial(torch.tensor([2.0, 1.0]))
        expected_1 = polynomial(torch.tensor([4.0, 2.0]))
        assert polynomial_equal(polynomial(result[0]), expected_0, tol=1e-6)
        assert polynomial_equal(polynomial(result[1]), expected_1, tol=1e-6)
