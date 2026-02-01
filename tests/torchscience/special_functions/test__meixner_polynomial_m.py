import math

import pytest
import scipy.special
import torch

import torchscience.special_functions


def scipy_meixner(n: int, x: float, beta: float, c: float) -> float:
    """Reference using scipy.special.hyp2f1.

    M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)

    Note: scipy's hyp2f1 can return NaN in some edge cases.
    """
    z = 1.0 - 1.0 / c
    return float(scipy.special.hyp2f1(-n, -x, beta, z))


class TestMeixnerPolynomialM:
    """Tests for Meixner polynomial M_n(x; beta, c)."""

    def test_m0_equals_one(self):
        """Test that M_0(x; beta, c) = 1 for all valid parameters."""
        n = torch.tensor([0.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for beta_val in [1.0, 2.0, 5.0]:
                for c_val in [0.2, 0.5, 0.8]:
                    x = torch.tensor([x_val], dtype=torch.float64)
                    beta = torch.tensor([beta_val], dtype=torch.float64)
                    c = torch.tensor([c_val], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.meixner_polynomial_m(
                            n, x, beta, c
                        )
                    )
                    torch.testing.assert_close(
                        result, torch.tensor([1.0], dtype=torch.float64)
                    )

    def test_m1_formula(self):
        """Test M_1(x; beta, c) against scipy reference.

        M_1(x; beta, c) = 2F1(-1, -x; beta; 1-1/c) = 1 + x*(c-1)/(c*beta)
        """
        n = torch.tensor([1.0], dtype=torch.float64)
        for x_val in [0.0, 1.0, 2.0, 3.0]:
            for beta_val in [1.0, 2.0, 5.0]:
                for c_val in [0.2, 0.5, 0.8]:
                    x = torch.tensor([x_val], dtype=torch.float64)
                    beta = torch.tensor([beta_val], dtype=torch.float64)
                    c = torch.tensor([c_val], dtype=torch.float64)
                    result = (
                        torchscience.special_functions.meixner_polynomial_m(
                            n, x, beta, c
                        )
                    )
                    # Use scipy as reference
                    expected_val = scipy_meixner(1, x_val, beta_val, c_val)
                    expected = torch.tensor(
                        [expected_val], dtype=torch.float64
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-8, atol=1e-8
                    )

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy hyp2f1 for integer orders."""
        for n_val in range(6):
            for x_val in range(6):
                for beta_val in [1.0, 2.0, 3.0]:
                    for c_val in [0.3, 0.5, 0.7]:
                        n = torch.tensor([float(n_val)], dtype=torch.float64)
                        x = torch.tensor([float(x_val)], dtype=torch.float64)
                        beta = torch.tensor([beta_val], dtype=torch.float64)
                        c = torch.tensor([c_val], dtype=torch.float64)

                        result = torchscience.special_functions.meixner_polynomial_m(
                            n, x, beta, c
                        )
                        expected_val = scipy_meixner(
                            n_val, float(x_val), beta_val, c_val
                        )

                        # Skip cases where scipy returns NaN
                        if math.isnan(expected_val) or math.isinf(
                            expected_val
                        ):
                            continue

                        expected = torch.tensor(
                            [expected_val], dtype=torch.float64
                        )
                        torch.testing.assert_close(
                            result, expected, rtol=1e-6, atol=1e-6
                        )

    def test_m_at_x_zero(self):
        """Test M_n(0; beta, c) values.

        M_n(0; beta, c) = 2F1(-n, 0; beta; z) = 1
        since b=0 makes all terms after first vanish.
        """
        x = torch.tensor([0.0], dtype=torch.float64)
        beta = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([0.5], dtype=torch.float64)

        for n_val in range(6):
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.meixner_polynomial_m(
                n, x, beta, c
            )
            expected = torch.tensor([1.0], dtype=torch.float64)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_gradcheck_c_parameter(self):
        """Test gradients with torch.autograd.gradcheck for c parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda c_: torchscience.special_functions.meixner_polynomial_m(
                n, x, beta, c_
            ),
            (c,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_x_parameter(self):
        """Test gradients with torch.autograd.gradcheck for x parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        beta = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([0.5], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda x_: torchscience.special_functions.meixner_polynomial_m(
                n, x_, beta, c
            ),
            (x,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_beta_parameter(self):
        """Test gradients with torch.autograd.gradcheck for beta parameter."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        beta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([0.5], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda beta_: torchscience.special_functions.meixner_polynomial_m(
                n, x, beta_, c
            ),
            (beta,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_complex_input(self):
        """Test with complex inputs."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        x = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        beta = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        c = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.meixner_polynomial_m(
            n, x, beta, c
        )
        # For real inputs, result should be real (imaginary part ~0)
        assert result.is_complex()
        expected_real = scipy_meixner(2, 1.0, 2.0, 0.5)
        torch.testing.assert_close(
            result.real,
            torch.tensor([expected_real], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        beta = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.meixner_polynomial_m(
            n, x, beta, c
        )
        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        x = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        beta = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        c = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.meixner_polynomial_m(
            n, x, beta, c
        )
        expected = scipy_meixner(2, 1.0, 2.0, 0.5)
        torch.testing.assert_close(
            result.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        n = torch.tensor([2.0], dtype=torch.float64, device="meta")
        x = torch.tensor([1.0], dtype=torch.float64, device="meta")
        beta = torch.tensor([2.0], dtype=torch.float64, device="meta")
        c = torch.tensor([0.5], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.meixner_polynomial_m(
            n, x, beta, c
        )
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_different_beta_values(self):
        """Test with different beta parameters."""
        n = torch.tensor([3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([0.5], dtype=torch.float64)

        for beta_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
            beta = torch.tensor([beta_val], dtype=torch.float64)
            result = torchscience.special_functions.meixner_polynomial_m(
                n, x, beta, c
            )
            expected = torch.tensor(
                [scipy_meixner(3, 2.0, beta_val, 0.5)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_different_c_values(self):
        """Test with different c parameters."""
        n = torch.tensor([3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        beta = torch.tensor([2.0], dtype=torch.float64)

        for c_val in [0.1, 0.25, 0.5, 0.75, 0.9]:
            c = torch.tensor([c_val], dtype=torch.float64)
            result = torchscience.special_functions.meixner_polynomial_m(
                n, x, beta, c
            )
            expected = torch.tensor(
                [scipy_meixner(3, 2.0, 2.0, c_val)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
