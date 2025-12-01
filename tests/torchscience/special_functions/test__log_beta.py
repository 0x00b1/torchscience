import math

import pytest
import torch
import torch.testing
from hypothesis import given, settings
from hypothesis import strategies as st

import torchscience.special_functions


class TestLogBeta:
    """Tests for the log_beta function."""

    def test_known_values(self):
        """Test log_beta at known values."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a, b)

        # log B(1,1) = 0, log B(2,2) = log(1/6), log B(3,3) = log(1/30)
        expected = torch.tensor(
            [
                0.0,  # log B(1,1) = log(1) = 0
                math.log(1 / 6),  # log B(2,2)
                math.log(1 / 30),  # log B(3,3)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_log_beta_half_half_equals_log_pi(self):
        """Test log B(0.5, 0.5) = log(pi)."""
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a, b)
        expected = torch.tensor([math.log(math.pi)], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_log_beta_one_n_equals_minus_log_n(self):
        """Test log B(1, n) = -log(n)."""
        n = torch.tensor([2.0, 3.0, 4.0, 5.0, 10.0], dtype=torch.float64)
        a = torch.ones_like(n)
        result = torchscience.special_functions.log_beta(a, n)
        expected = -torch.log(n)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_symmetry(self):
        """Test log B(a, b) = log B(b, a)."""
        a = torch.tensor([1.5, 2.0, 3.5, 5.0], dtype=torch.float64)
        b = torch.tensor([2.5, 4.0, 1.5, 7.0], dtype=torch.float64)
        result_ab = torchscience.special_functions.log_beta(a, b)
        result_ba = torchscience.special_functions.log_beta(b, a)
        torch.testing.assert_close(
            result_ab, result_ba, rtol=1e-12, atol=1e-12
        )

    def test_consistency_with_beta(self):
        """Test log_beta(a, b) = log(beta(a, b))."""
        a = torch.tensor([1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 4.0, 2.0], dtype=torch.float64)
        log_beta_result = torchscience.special_functions.log_beta(a, b)
        beta_result = torchscience.special_functions.beta(a, b)
        expected = torch.log(beta_result)
        torch.testing.assert_close(
            log_beta_result, expected, rtol=1e-8, atol=1e-8
        )

    def test_consistency_with_lgamma(self):
        """Test log_beta(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
        a = torch.tensor([1.5, 2.0, 3.5, 10.0], dtype=torch.float64)
        b = torch.tensor([2.5, 4.0, 1.5, 7.0], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a, b)
        expected = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_poles_return_inf(self):
        """Test that log_beta at poles returns inf or -inf."""
        # log B(0, x) = lgamma(0) + lgamma(x) - lgamma(x) = inf
        # log B(-1, 1) = lgamma(-1) + lgamma(1) - lgamma(0) = inf + 0 - inf = -inf
        a_poles = torch.tensor([0.0, -1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a_poles, b)
        assert torch.isinf(result).all()

    def test_gradient_formula(self):
        """Test d/da log_beta(a,b) = psi(a) - psi(a+b)."""
        a = torch.tensor(
            [2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor([3.0, 4.0, 2.0], dtype=torch.float64)
        y = torchscience.special_functions.log_beta(a, b)
        y.sum().backward()

        # Expected: psi(a) - psi(a+b)
        expected = torch.special.digamma(a.detach()) - torch.special.digamma(
            a.detach() + b
        )
        torch.testing.assert_close(a.grad, expected, rtol=1e-6, atol=1e-6)

    def test_gradient_b(self):
        """Test d/db log_beta(a,b) = psi(b) - psi(a+b)."""
        a = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        b = torch.tensor(
            [3.0, 4.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.log_beta(a, b)
        y.sum().backward()

        # Expected: psi(b) - psi(a+b)
        expected = torch.special.digamma(b.detach()) - torch.special.digamma(
            a + b.detach()
        )
        torch.testing.assert_close(b.grad, expected, rtol=1e-6, atol=1e-6)

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        a = torch.tensor(
            [1.5, 2.5, 5.0], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.log_beta,
            (a, b),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.log_beta,
            (a, b),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.log_beta(a, b)

        (grad_a,) = torch.autograd.grad(y, a, create_graph=True)
        (grad2_a,) = torch.autograd.grad(grad_a, a)

        assert torch.isfinite(grad2_a).all()
        # d²/da² log_beta = psi'(a) - psi'(a+b) = trigamma(a) - trigamma(a+b)
        expected = torch.special.polygamma(
            1, a.detach()
        ) - torch.special.polygamma(1, a.detach() + b.detach())
        torch.testing.assert_close(grad2_a, expected, rtol=1e-6, atol=1e-6)

    def test_broadcasting(self):
        """Test broadcasting between a and b."""
        a = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a, b)
        assert result.shape == (3, 4)

    def test_dtypes(self):
        """Test various dtypes."""
        for dtype in [torch.float32, torch.float64]:
            a = torch.tensor([2.0, 3.0], dtype=dtype)
            b = torch.tensor([3.0, 4.0], dtype=dtype)
            result = torchscience.special_functions.log_beta(a, b)
            assert result.dtype == dtype

    @pytest.mark.skip(reason="Complex log_beta not yet fully implemented")
    def test_complex_input(self):
        """Test complex input."""
        a = torch.tensor([1.0 + 1.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.log_beta(a, b)
        # Verify against lgamma
        expected = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    @given(
        a=st.floats(
            min_value=0.1,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        b=st.floats(
            min_value=0.1,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_symmetry(self, a, b):
        """Property: log_beta(a, b) = log_beta(b, a)."""
        a_tensor = torch.tensor([a], dtype=torch.float64)
        b_tensor = torch.tensor([b], dtype=torch.float64)
        result_ab = torchscience.special_functions.log_beta(a_tensor, b_tensor)
        result_ba = torchscience.special_functions.log_beta(b_tensor, a_tensor)
        if torch.isfinite(result_ab).all() and torch.isfinite(result_ba).all():
            torch.testing.assert_close(
                result_ab, result_ba, rtol=1e-10, atol=1e-10
            )

    @given(
        a=st.floats(
            min_value=0.1,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        b=st.floats(
            min_value=0.1,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_lgamma_identity(self, a, b):
        """Property: log_beta = lgamma(a) + lgamma(b) - lgamma(a+b)."""
        a_tensor = torch.tensor([a], dtype=torch.float64)
        b_tensor = torch.tensor([b], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a_tensor, b_tensor)
        expected = (
            torch.lgamma(a_tensor)
            + torch.lgamma(b_tensor)
            - torch.lgamma(a_tensor + b_tensor)
        )
        if torch.isfinite(result).all() and torch.isfinite(expected).all():
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_recurrence_relation(self):
        """Test recurrence: log_beta(a+1, b) = log_beta(a, b) + log(a) - log(a+b)."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 4.0, 2.0], dtype=torch.float64)

        left = torchscience.special_functions.log_beta(a + 1, b)
        right = (
            torchscience.special_functions.log_beta(a, b)
            + torch.log(a)
            - torch.log(a + b)
        )
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_comparison_with_scipy(self):
        """Test agreement with scipy.special.betaln if available."""
        pytest.importorskip("scipy")
        import scipy.special

        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0, 3.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.log_beta(a, b)
        expected = torch.tensor(
            scipy.special.betaln(a.numpy(), b.numpy()), dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
