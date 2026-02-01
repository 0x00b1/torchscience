import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


def scipy_legendre_q(n: int, x: float) -> float:
    """Reference implementation using scipy.

    scipy.special.lqn returns (Q_0, Q_1, ..., Q_n) and their derivatives.
    We just return Q_n.
    """
    if abs(x) >= 1.0:
        return float("nan")
    Qn, _ = scipy.special.lqn(n, x)
    return Qn[n]


class TestLegendrePolynomialQ:
    """Tests for Legendre function of the second kind."""

    def test_forward_q0(self):
        """Test Q_0(x) = arctanh(x)."""
        x = torch.tensor([0.0, 0.5, -0.5, 0.9, -0.9], dtype=torch.float64)
        n = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        expected = torch.arctanh(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_q1(self):
        """Test Q_1(x) = x * arctanh(x) - 1."""
        x = torch.tensor([0.0, 0.5, -0.5, 0.9, -0.9], dtype=torch.float64)
        n = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        expected = x * torch.arctanh(x) - 1
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.lqn."""
        x_vals = torch.linspace(-0.95, 0.95, 15, dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_q(
                x_vals, n
            )

            expected_list = [
                scipy_legendre_q(n_val, xi.item()) for xi in x_vals
            ]
            expected = torch.tensor(expected_list, dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Mismatch for n={n_val}",
            )

    def test_q0_at_zero(self):
        """Test Q_0(0) = 0."""
        x = torch.tensor([0.0], dtype=torch.float64)
        n = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_q1_at_zero(self):
        """Test Q_1(0) = -1."""
        x = torch.tensor([0.0], dtype=torch.float64)
        n = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        expected = torch.tensor([-1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_parity_property(self):
        """Test Q_n(-x) = (-1)^{n+1} Q_n(x) for integer n."""
        x_vals = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result_pos = torchscience.special_functions.legendre_polynomial_q(
                x_vals, n
            )
            result_neg = torchscience.special_functions.legendre_polynomial_q(
                -x_vals, n
            )

            expected_relation = ((-1.0) ** (n_val + 1)) * result_pos
            torch.testing.assert_close(
                result_neg,
                expected_relation,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Parity property failed for n={n_val}",
            )

    def test_recurrence_relation(self):
        """Test (n+1)Q_{n+1}(x) = (2n+1)*x*Q_n(x) - n*Q_{n-1}(x)."""
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        for n in range(1, 8):
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
            n_plus_1 = torch.tensor([float(n + 1)], dtype=torch.float64)

            Q_n = torchscience.special_functions.legendre_polynomial_q(
                x, n_tensor
            )
            Q_n_minus_1 = torchscience.special_functions.legendre_polynomial_q(
                x, n_minus_1
            )
            Q_n_plus_1 = torchscience.special_functions.legendre_polynomial_q(
                x, n_plus_1
            )

            left = (n + 1) * Q_n_plus_1
            right = (2 * n + 1) * x * Q_n - n * Q_n_minus_1

            torch.testing.assert_close(
                left,
                right,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Recurrence failed at n={n}",
            )

    def test_gradcheck_x(self):
        """Test gradient correctness with respect to x."""
        x = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.legendre_polynomial_q(x, n)

        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_n(self):
        """Test gradient correctness with respect to n."""
        x = torch.tensor([0.5], dtype=torch.float64)
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(n):
            return torchscience.special_functions.legendre_polynomial_q(x, n)

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_gradient_q0_analytical(self):
        """Test that dQ_0/dx = 1/(1-x^2)."""
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        result.backward()

        expected_grad = 1 / (1 - x.detach() ** 2)  # dQ_0/dx = 1/(1-x^2)
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.empty(3, 1, device="meta", dtype=torch.float64)
        n = torch.empty(1, 5, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)

        # Should broadcast to (3, 5) shape
        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between x and n."""
        x = torch.tensor([[0.3], [0.5], [0.7]], dtype=torch.float64)  # (3, 1)
        n = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)  # (1, 3)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        assert result.shape == (3, 3)

        # Verify corner values
        # Q_0(0.3) = arctanh(0.3)
        torch.testing.assert_close(
            result[0, 0],
            torch.arctanh(torch.tensor(0.3, dtype=torch.float64)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        x = torch.tensor([0.5], dtype=torch.float32)
        n = torch.tensor([0.0], dtype=torch.float32)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        assert result.dtype == torch.float32

        expected = torch.arctanh(x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_low_precision_dtype_bfloat16(self):
        """Test with bfloat16 dtype.

        Note: float16 is not supported because Q_n involves log((1+x)/(1-x))
        which has precision issues in float16 due to limited mantissa bits,
        resulting in NaN for some inputs.
        """
        dtype = torch.bfloat16
        x = torch.tensor([0.5], dtype=dtype)
        n = torch.tensor([0.0], dtype=dtype)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        assert result.dtype == dtype

        # Compare to float32 reference
        x_f32 = x.to(torch.float32)
        n_f32 = n.to(torch.float32)
        expected = torchscience.special_functions.legendre_polynomial_q(
            x_f32, n_f32
        )

        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=5e-2, atol=5e-2
        )

    def test_large_degree(self):
        """Test with larger polynomial degrees."""
        x = torch.tensor([0.5], dtype=torch.float64)

        for n_val in [5, 8, 10]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_q(x, n)
            expected = torch.tensor(
                [scipy_legendre_q(n_val, 0.5)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Large degree n={n_val} failed",
            )

    def test_singularity_behavior(self):
        """Test behavior near singularities at x = +/- 1."""
        n = torch.tensor([0.0], dtype=torch.float64)

        # Values approaching 1
        x_near_1 = torch.tensor([0.99, 0.999], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_q(
            x_near_1, n
        )

        # Should be large positive values (approaching +inf)
        assert torch.all(result > 0)
        assert result[1] > result[0]  # Should increase as we approach 1

        # Values approaching -1
        x_near_minus_1 = torch.tensor([-0.99, -0.999], dtype=torch.float64)
        result_neg = torchscience.special_functions.legendre_polynomial_q(
            x_near_minus_1, n
        )

        # Should be large negative values (approaching -inf)
        assert torch.all(result_neg < 0)
        assert (
            result_neg[1] < result_neg[0]
        )  # Should decrease (more negative) as we approach -1

    def test_wronskian_relation(self):
        """Test the Wronskian: W[P_n, Q_n] = P_n * Q_n' - P_n' * Q_n = 1/(1-x^2).

        We verify this using finite differences for the derivatives.
        """
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        eps = 1e-7

        for n_val in [0, 1, 2, 3]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)

            P_n = torchscience.special_functions.legendre_polynomial_p(n, x)
            Q_n = torchscience.special_functions.legendre_polynomial_q(x, n)

            # Numerical derivatives
            P_n_plus = torchscience.special_functions.legendre_polynomial_p(
                n, x + eps
            )
            P_n_minus = torchscience.special_functions.legendre_polynomial_p(
                n, x - eps
            )
            dP_dx = (P_n_plus - P_n_minus) / (2 * eps)

            Q_n_plus = torchscience.special_functions.legendre_polynomial_q(
                x + eps, n
            )
            Q_n_minus = torchscience.special_functions.legendre_polynomial_q(
                x - eps, n
            )
            dQ_dx = (Q_n_plus - Q_n_minus) / (2 * eps)

            wronskian = P_n * dQ_dx - Q_n * dP_dx
            expected = 1 / (1 - x**2)

            torch.testing.assert_close(
                wronskian,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Wronskian failed for n={n_val}",
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, device="cuda")
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.legendre_polynomial_q(
            x.cpu(), n.cpu()
        )
        torch.testing.assert_close(
            result.cpu(), result_cpu, rtol=1e-12, atol=1e-12
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward(self):
        """Test backward pass on CUDA."""
        x = torch.tensor(
            [0.3, 0.5, 0.7],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.legendre_polynomial_q(x, n)
        result.sum().backward()

        # Compare gradient to CPU
        x_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.legendre_polynomial_q(
            x_cpu, n.cpu()
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            x.grad.cpu(), x_cpu.grad, rtol=1e-10, atol=1e-10
        )

    def test_gradgradcheck_x(self):
        """Test second-order gradient correctness with respect to x.

        Note: The backward_backward kernel uses finite differences internally,
        so we use relaxed tolerances here.
        """
        x = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.legendre_polynomial_q(x, n)

        # Second-order gradients - relaxed tolerance due to finite difference approximation
        assert torch.autograd.gradgradcheck(
            func, (x,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_legendre_q(x, n):
            return torchscience.special_functions.legendre_polynomial_q(x, n)

        x = torch.tensor([0.5], dtype=torch.float64)
        n = torch.tensor([0.0], dtype=torch.float64)

        result = compiled_legendre_q(x, n)
        expected = torch.arctanh(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_q2_explicit(self):
        """Test Q_2(x) against explicit formula.

        Q_2(x) = P_2(x) * Q_0(x) - (3/2) * x
               = (3x^2-1)/2 * arctanh(x) - 3x/2
        """
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        n = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)

        Q0 = torch.arctanh(x)
        P2 = (3 * x**2 - 1) / 2
        expected = P2 * Q0 - 3 * x / 2

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_q3_explicit(self):
        """Test Q_3(x) against explicit formula.

        Q_3(x) = P_3(x) * Q_0(x) - (5/2) * x^2 + (2/3)
        where P_3(x) = (5x^3 - 3x)/2
        """
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        n = torch.tensor([3.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_q(x, n)

        Q0 = torch.arctanh(x)
        P3 = (5 * x**3 - 3 * x) / 2
        # W_2(x) = (1/1)*P_0*P_2 + (1/2)*P_1*P_1
        # P_0 = 1, P_1 = x, P_2 = (3x^2-1)/2
        # W_2 = (3x^2-1)/2 + x^2/2 = (3x^2 - 1 + x^2)/2 = (4x^2 - 1)/2 = 2x^2 - 0.5
        # But simpler to use recurrence result:
        # Q_3 = (5 * 0.5 * Q_2 - 2 * Q_1) / 3 for x=0.5
        # Let's just compare to scipy
        expected_list = [scipy_legendre_q(3, xi.item()) for xi in x]
        expected = torch.tensor(expected_list, dtype=torch.float64)

        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)
