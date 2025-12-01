import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestModifiedBesselK:
    """Tests for the modified Bessel function of the second kind K_n(z) of general order n."""

    # =========================================================================
    # Forward correctness tests - integer orders
    # =========================================================================

    def test_order_zero_matches_k0(self):
        """Test K_0(z) via modified_bessel_k(0, z) matches modified_bessel_k_0(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.zeros_like(z)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        expected = torchscience.special_functions.modified_bessel_k_0(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_order_one_matches_k1(self):
        """Test K_1(z) via modified_bessel_k(1, z) matches modified_bessel_k_1(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.ones_like(z)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        expected = torchscience.special_functions.modified_bessel_k_1(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy.special.kv for integer orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0, 1, 2, 3, 5, 10]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.modified_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.kv(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_negative_integer_orders(self):
        """Test agreement with scipy for negative integer orders: K_{-n}(z) = K_n(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [-1, -2, -3, -5]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.modified_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.kv(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Forward correctness tests - non-integer orders
    # =========================================================================

    def test_scipy_agreement_half_order(self):
        """Test agreement with scipy for half-integer orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0.5, 1.5, 2.5, -0.5, -1.5]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.modified_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.kv(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_fractional_order(self):
        """Test agreement with scipy for general fractional orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0.3, 1.7, 2.25, 3.8]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.modified_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.kv(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Special values tests
    # =========================================================================

    def test_at_zero_is_infinity(self):
        """Test K_n(0) = +infinity for all n."""
        z = torch.tensor([0.0], dtype=torch.float64)
        for order in [0.0, 1.0, 2.0, 0.5, -1.0]:
            n = torch.tensor([order], dtype=torch.float64)
            result = torchscience.special_functions.modified_bessel_k(n, z)
            assert torch.isinf(result).all(), (
                f"K_{order}(0) should be infinite"
            )
            assert (result > 0).all(), (
                f"K_{order}(0) should be positive infinity"
            )

    def test_positive_for_positive_z(self):
        """Test K_n(z) > 0 for z > 0."""
        z = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0.0, 1.0, 2.0, 0.5, -1.5]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.modified_bessel_k(n, z)
            assert (result > 0).all(), (
                f"K_{order}(z) should be positive for z > 0"
            )

    def test_at_infinity_is_zero(self):
        """Test K_n(+infinity) = 0 (exponential decay)."""
        z = torch.tensor([float("inf")], dtype=torch.float64)
        for order in [0.0, 1.0, 2.0]:
            n = torch.tensor([order], dtype=torch.float64)
            result = torchscience.special_functions.modified_bessel_k(n, z)
            torch.testing.assert_close(
                result,
                torch.tensor([0.0], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_nan_input(self):
        """Test K_n(NaN) = NaN."""
        z = torch.tensor([float("nan")], dtype=torch.float64)
        n = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert torch.isnan(result).all()

    # =========================================================================
    # Symmetry tests
    # =========================================================================

    def test_negative_order_symmetry(self):
        """Test K_{-n}(z) = K_n(z) for all n."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0.0, 1.0, 2.0, 0.5, 1.5, 3.0]:
            n_pos = torch.full_like(z, order)
            n_neg = torch.full_like(z, -order)
            result_pos = torchscience.special_functions.modified_bessel_k(
                n_pos, z
            )
            result_neg = torchscience.special_functions.modified_bessel_k(
                n_neg, z
            )
            torch.testing.assert_close(
                result_neg,
                result_pos,
                rtol=1e-10,
                atol=1e-10,
                msg=f"K_{{-{order}}}(z) != K_{order}(z)",
            )

    # =========================================================================
    # Recurrence relation tests
    # =========================================================================

    def test_recurrence_relation(self):
        """Test recurrence: K_{n+1}(z) = K_{n-1}(z) + (2n/z) * K_n(z)."""
        z = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [1.0, 2.0, 3.0, 0.5, 1.5]:
            n = torch.full_like(z, order)
            n_minus = torch.full_like(z, order - 1)
            n_plus = torch.full_like(z, order + 1)

            k_n = torchscience.special_functions.modified_bessel_k(n, z)
            k_nm1 = torchscience.special_functions.modified_bessel_k(
                n_minus, z
            )
            k_np1 = torchscience.special_functions.modified_bessel_k(n_plus, z)

            lhs = k_np1
            rhs = k_nm1 + (2 * order / z) * k_n
            torch.testing.assert_close(
                lhs,
                rhs,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Recurrence failed for n={order}",
            )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck_z_integer_order(self):
        """Test first-order gradient correctness for integer order."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.modified_bessel_k(n, z_)

        assert torch.autograd.gradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z_fractional_order(self):
        """Test first-order gradient correctness for fractional order."""
        n = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.modified_bessel_k(n, z_)

        assert torch.autograd.gradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_both_inputs(self):
        """Test first-order gradient correctness for both n and z."""
        n = torch.tensor(
            [0.5, 1.5, 2.5], dtype=torch.float64, requires_grad=True
        )
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(n_, z_):
            return torchscience.special_functions.modified_bessel_k(n_, z_)

        # Note: gradient w.r.t. n is computed numerically; may need relaxed tolerance
        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness for z."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.modified_bessel_k(n, z_)

        assert torch.autograd.gradgradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_identity(self):
        """Verify d/dz K_n(z) = -(K_{n-1}(z) + K_{n+1}(z)) / 2."""
        n = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        y = torchscience.special_functions.modified_bessel_k(n, z)
        grad = torch.autograd.grad(y.sum(), z)[0]

        n_minus = n - 1
        n_plus = n + 1
        k_nm1 = torchscience.special_functions.modified_bessel_k(
            n_minus, z.detach()
        )
        k_np1 = torchscience.special_functions.modified_bessel_k(
            n_plus, z.detach()
        )
        expected = -(k_nm1 + k_np1) / 2

        torch.testing.assert_close(grad, expected, rtol=1e-8, atol=1e-10)

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting_n_scalar(self):
        """Test broadcasting when n is scalar-like."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.shape == (4,)

    def test_broadcasting_z_scalar(self):
        """Test broadcasting when z is scalar-like."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.shape == (4,)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)  # (4,)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.shape == (3, 4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j, 3.0 + 0.2j], dtype=dtype)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.dtype == dtype

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_near_real_accuracy(self):
        """Validate complex accuracy against scipy near real axis."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
        z = torch.tensor(
            [1.0 + 0.1j, 2.0 - 0.1j, 3.0 + 0.05j], dtype=torch.complex128
        )
        result = torchscience.special_functions.modified_bessel_k(n, z)
        expected = torch.tensor(
            [
                scipy.special.kv(n[i].real.item(), z[i].item())
                for i in range(len(z))
            ],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real K_n."""
        n_real = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z_real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        n_complex = n_real.to(torch.complex128)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.modified_bessel_k(
            n_real, z_real
        )
        result_complex = torchscience.special_functions.modified_bessel_k(
            n_complex, z_complex
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
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.shape == (10,)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor shape inference with broadcasting."""
        n = torch.randn(3, 1, device="meta")
        z = torch.randn(1, 4, device="meta")
        result = torchscience.special_functions.modified_bessel_k(n, z)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.modified_bessel_k(n, z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.modified_bessel_k(n, z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = (
            torch.abs(torch.randn(5, 3, dtype=torch.float64)) + 0.1
        )  # positive z

        def fn(z_row):
            return torchscience.special_functions.modified_bessel_k(n, z_row)

        result = torch.vmap(fn)(z)
        expected = torchscience.special_functions.modified_bessel_k(
            n.unsqueeze(0), z
        )
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_k
        )
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.abs(torch.randn(3, dtype=torch.float64)) + 0.1
        result = compiled_fn(n, z)
        expected = torchscience.special_functions.modified_bessel_k(n, z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.modified_bessel_k
        )
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = (
            torch.abs(torch.randn(3, dtype=torch.float64)) + 0.1
        ).requires_grad_(True)
        result = compiled_fn(n, z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.modified_bessel_k(n, z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    # =========================================================================
    # Large argument tests
    # =========================================================================

    def test_large_z_asymptotic(self):
        """Test large z values where asymptotic expansion is used."""
        n = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
        z = torch.tensor([50.0, 100.0, 200.0, 500.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k(
            n.unsqueeze(1), z
        )
        expected = torch.tensor(
            [[scipy.special.kv(ni.item(), zi.item()) for zi in z] for ni in n],
            dtype=torch.float64,
        )
        # Slightly relaxed tolerance for asymptotic region
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_large_order(self):
        """Test large order values."""
        n = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        z = torch.tensor([15.0, 25.0, 55.0], dtype=torch.float64)
        result = torchscience.special_functions.modified_bessel_k(n, z)
        expected = torch.tensor(
            [
                scipy.special.kv(n[i].item(), z[i].item())
                for i in range(len(z))
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-10)

    # =========================================================================
    # Exponential decay tests
    # =========================================================================

    def test_exponential_decay_for_large_z(self):
        """Test that K_n(z) decays exponentially for large z."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z1 = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        z2 = torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64)

        k1 = torchscience.special_functions.modified_bessel_k(n, z1)
        k2 = torchscience.special_functions.modified_bessel_k(n, z2)

        # K_n(2x) / K_n(x) ~ e^{-x} * sqrt(x/(2x)) * (1 + O(1/x)) for large x
        # The asymptotic form includes a sqrt(x) factor, so the ratio is approximately
        # e^{-x} * sqrt(1/2). This is a qualitative test of exponential decay.
        ratio = k2 / k1
        expected_ratio = torch.exp(-(z2 - z1)) * torch.sqrt(z1 / z2)

        # Relaxed tolerance because asymptotic formula has corrections
        torch.testing.assert_close(ratio, expected_ratio, rtol=0.5, atol=1e-6)
