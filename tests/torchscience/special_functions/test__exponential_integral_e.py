import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestExponentialIntegralE:
    """Tests for the generalized exponential integral E_n(x)."""

    # =========================================================================
    # Forward correctness tests - E_0
    # =========================================================================

    def test_e_0_matches_direct_formula(self):
        """Test E_0(x) = e^{-x} / x."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.zeros_like(x)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        expected = torch.exp(-x) / x
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Forward correctness tests - E_1
    # =========================================================================

    def test_e_1_matches_specialized_function(self):
        """Test E_1(x) via exponential_integral_e(1, x) matches exponential_integral_e_1(x)."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.ones_like(x)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        expected = torchscience.special_functions.exponential_integral_e_1(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Forward correctness tests - scipy agreement
    # =========================================================================

    def test_scipy_agreement_various_orders(self):
        """Test agreement with scipy.special.expn for various orders."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0, 1, 2, 3, 5, 10]:
            n = torch.full_like(x, float(order))
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            expected = torch.tensor(
                [scipy.special.expn(order, xi.item()) for xi in x],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_small_x(self):
        """Test agreement with scipy for small x values."""
        x = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5], dtype=torch.float64)
        for order in [1, 2, 3, 4]:
            n = torch.full_like(x, float(order))
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            expected = torch.tensor(
                [scipy.special.expn(order, xi.item()) for xi in x],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_large_x(self):
        """Test agreement with scipy for large x values."""
        x = torch.tensor([5.0, 10.0, 20.0, 50.0], dtype=torch.float64)
        for order in [0, 1, 2, 5]:
            n = torch.full_like(x, float(order))
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            expected = torch.tensor(
                [scipy.special.expn(order, xi.item()) for xi in x],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-12,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Special values tests
    # =========================================================================

    def test_at_zero_e_n_for_n_geq_2(self):
        """Test E_n(0) = 1/(n-1) for n >= 2."""
        for order in [2, 3, 4, 5, 10]:
            n = torch.tensor([float(order)], dtype=torch.float64)
            x = torch.tensor([0.0], dtype=torch.float64)
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            expected = torch.tensor([1.0 / (order - 1)], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for E_{order}(0)",
            )

    def test_at_zero_e_0_and_e_1(self):
        """Test E_0(0) = +inf and E_1(0) = +inf."""
        x = torch.tensor([0.0], dtype=torch.float64)
        for order in [0, 1]:
            n = torch.tensor([float(order)], dtype=torch.float64)
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            assert result.isinf().all() and result[0] > 0, (
                f"E_{order}(0) should be +inf"
            )

    def test_negative_x_returns_nan(self):
        """Test E_n(x < 0) = NaN for real inputs."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([-1.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.isnan().all()

    def test_non_integer_n_returns_nan(self):
        """Test E_n(x) = NaN for non-integer n."""
        n = torch.tensor([1.5], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.isnan().all()

    def test_negative_n_returns_nan(self):
        """Test E_n(x) = NaN for n < 0."""
        n = torch.tensor([-1.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.isnan().all()

    def test_nan_input(self):
        """Test E_n(NaN) = NaN and E_NaN(x) = NaN."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.isnan().all()

        n = torch.tensor([float("nan")], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.isnan().all()

    # =========================================================================
    # Recurrence relation tests
    # =========================================================================

    def test_recurrence_relation(self):
        """Test recurrence: E_n(x) = (e^{-x} - x * E_{n-1}(x)) / (n-1)."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [2, 3, 4, 5]:
            n = torch.full_like(x, float(order))
            n_minus = torch.full_like(x, float(order - 1))

            E_n = torchscience.special_functions.exponential_integral_e(n, x)
            E_nm1 = torchscience.special_functions.exponential_integral_e(
                n_minus, x
            )

            expected = (torch.exp(-x) - x * E_nm1) / (order - 1)
            torch.testing.assert_close(
                E_n,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Recurrence failed for n={order}",
            )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck_x(self):
        """Test first-order gradient correctness for x."""
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(x_):
            return torchscience.special_functions.exponential_integral_e(n, x_)

        assert torch.autograd.gradcheck(
            func, x, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_small_x(self):
        """Test gradient correctness for small x values."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor(
            [0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True
        )

        def func(x_):
            return torchscience.special_functions.exponential_integral_e(n, x_)

        assert torch.autograd.gradcheck(
            func, x, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_large_x(self):
        """Test gradient correctness for large x values."""
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor(
            [5.0, 10.0, 20.0], dtype=torch.float64, requires_grad=True
        )

        def func(x_):
            return torchscience.special_functions.exponential_integral_e(n, x_)

        assert torch.autograd.gradcheck(
            func, x, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck_x(self):
        """Test second-order gradient correctness for x."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(x_):
            return torchscience.special_functions.exponential_integral_e(n, x_)

        assert torch.autograd.gradgradcheck(
            func, x, eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradient_x_identity(self):
        """Verify d/dx E_n(x) = -E_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        y = torchscience.special_functions.exponential_integral_e(n, x)
        grad = torch.autograd.grad(y.sum(), x)[0]

        n_minus = n - 1
        E_nm1 = torchscience.special_functions.exponential_integral_e(
            n_minus, x.detach()
        )
        expected = -E_nm1

        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-10)

    def test_gradient_n_is_zero(self):
        """Verify gradient w.r.t. n is zero."""
        n = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        y = torchscience.special_functions.exponential_integral_e(n, x)
        grad = torch.autograd.grad(y.sum(), n)[0]

        expected = torch.zeros_like(n)
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting_n_scalar(self):
        """Test broadcasting when n is scalar-like."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.shape == (4,)

    def test_broadcasting_x_scalar(self):
        """Test broadcasting when x is scalar-like."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.shape == (4,)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)  # (4,)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.shape == (3, 4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        x = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j, 3.0 + 0.2j], dtype=dtype)
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.dtype == dtype

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real E_n."""
        n_real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x_real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        n_complex = n_real.to(torch.complex128)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.exponential_integral_e(
            n_real, x_real
        )
        result_complex = torchscience.special_functions.exponential_integral_e(
            n_complex, x_complex
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_complex.imag),
            rtol=1e-10,
            atol=1e-10,
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.randn(10, device="meta")
        x = torch.randn(10, device="meta")
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.shape == (10,)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor shape inference with broadcasting."""
        n = torch.randn(3, 1, device="meta")
        x = torch.randn(1, 4, device="meta")
        result = torchscience.special_functions.exponential_integral_e(n, x)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.exponential_integral_e(n, x)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.rand(5, 3, dtype=torch.float64) + 0.5  # Ensure x > 0

        def fn(x_row):
            return torchscience.special_functions.exponential_integral_e(
                n, x_row
            )

        result = torch.vmap(fn)(x)
        expected = torchscience.special_functions.exponential_integral_e(
            n.unsqueeze(0), x
        )
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.exponential_integral_e
        )
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.rand(3, dtype=torch.float64) + 0.5
        result = compiled_fn(n, x)
        expected = torchscience.special_functions.exponential_integral_e(n, x)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.exponential_integral_e
        )
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor(
            [0.6, 1.2, 0.8], dtype=torch.float64, requires_grad=True
        )
        result = compiled_fn(n, x)
        result.sum().backward()
        assert x.grad is not None
        # Verify gradient matches uncompiled version
        x2 = x.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.exponential_integral_e(n, x2)
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x2.grad)

    # =========================================================================
    # Monotonicity and asymptotic tests
    # =========================================================================

    def test_monotonically_decreasing_in_x(self):
        """Test E_n(x) is monotonically decreasing in x for x > 0."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0, 1, 2, 3]:
            n = torch.full_like(x, float(order))
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            # Check that each value is less than the previous
            diffs = result[1:] - result[:-1]
            assert (diffs < 0).all(), f"E_{order} not monotonically decreasing"

    def test_ordering_in_n(self):
        """Test E_n(x) < E_m(x) for n > m and x > 0."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in range(1, 5):
            n_lower = torch.full_like(x, float(order - 1))
            n_higher = torch.full_like(x, float(order))
            E_lower = torchscience.special_functions.exponential_integral_e(
                n_lower, x
            )
            E_higher = torchscience.special_functions.exponential_integral_e(
                n_higher, x
            )
            # E_n(x) < E_{n-1}(x) for the same x
            assert (E_higher < E_lower).all(), (
                f"E_{order} not less than E_{order - 1}"
            )

    def test_asymptotic_large_x(self):
        """Test asymptotic behavior E_n(x) ~ e^{-x}/x for large x."""
        x = torch.tensor([50.0, 100.0], dtype=torch.float64)
        for order in [0, 1, 2, 3]:
            n = torch.full_like(x, float(order))
            result = torchscience.special_functions.exponential_integral_e(
                n, x
            )
            # For large x, E_n(x) ~ e^{-x}/x
            asymptotic = torch.exp(-x) / x
            # The ratio should approach 1 for large x
            ratio = result / asymptotic
            assert (ratio > 0.5).all() and (ratio < 2.0).all(), (
                f"Asymptotic behavior not correct for n={order}"
            )
