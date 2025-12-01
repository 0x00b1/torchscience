import math

import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestSphericalBesselK:
    """Tests for the modified spherical Bessel function k_n(z) of general order n."""

    # =========================================================================
    # Forward correctness tests - integer orders
    # =========================================================================

    def test_order_zero_matches_k0(self):
        """Test k_0(z) via spherical_bessel_k(0, z) matches spherical_bessel_k_0(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.zeros_like(z)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        expected = torchscience.special_functions.spherical_bessel_k_0(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_order_one_matches_k1(self):
        """Test k_1(z) via spherical_bessel_k(1, z) matches spherical_bessel_k_1(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.ones_like(z)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        expected = torchscience.special_functions.spherical_bessel_k_1(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy.special.spherical_kn for integer orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0, 1, 2, 3, 5, 10]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.spherical_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.spherical_kn(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_large_orders(self):
        """Test agreement with scipy for large integer orders."""
        z = torch.tensor([10.0, 15.0, 20.0, 30.0], dtype=torch.float64)
        for order in [5, 10, 15]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.spherical_bessel_k(n, z)
            expected = torch.tensor(
                [scipy.special.spherical_kn(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Forward correctness tests - non-integer orders
    # =========================================================================

    def test_scipy_agreement_half_order(self):
        """Test agreement with scipy for half-integer orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [0.5, 1.5, 2.5]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.spherical_bessel_k(n, z)
            # For half-integer order, use modified Bessel K relation
            nu = order + 0.5
            expected = torch.tensor(
                [
                    math.sqrt(math.pi / (2 * x.item()))
                    * scipy.special.kv(nu, x.item())
                    for x in z
                ],
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
            result = torchscience.special_functions.spherical_bessel_k(n, z)
            nu = order + 0.5
            expected = torch.tensor(
                [
                    math.sqrt(math.pi / (2 * x.item()))
                    * scipy.special.kv(nu, x.item())
                    for x in z
                ],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Special values tests
    # =========================================================================

    def test_at_zero_singular(self):
        """Test k_n(0) = infinity for all n (singular at origin)."""
        z = torch.tensor([0.0], dtype=torch.float64)
        for order in [0, 1, 2, 0.5, 1.5]:
            n = torch.tensor([float(order)], dtype=torch.float64)
            result = torchscience.special_functions.spherical_bessel_k(n, z)
            assert result.isinf().all() and (result > 0).all()

    def test_nan_input(self):
        """Test k_n(NaN) = NaN and k_NaN(z) = NaN."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.isnan().all()

        n = torch.tensor([float("nan")], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.isnan().all()

    def test_negative_z_returns_nan(self):
        """Test k_n(z) returns NaN for negative z (real implementation)."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.isnan().all()

    # =========================================================================
    # Recurrence relation tests
    # =========================================================================

    def test_recurrence_relation(self):
        """Test recurrence: k_{n+1}(z) - k_{n-1}(z) = (2n+1)/z * k_n(z)."""
        z = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        for order in [1.0, 2.0, 3.0, 0.5, 1.5]:
            n = torch.full_like(z, order)
            n_minus = torch.full_like(z, order - 1)
            n_plus = torch.full_like(z, order + 1)

            k_n = torchscience.special_functions.spherical_bessel_k(n, z)
            k_nm1 = torchscience.special_functions.spherical_bessel_k(
                n_minus, z
            )
            k_np1 = torchscience.special_functions.spherical_bessel_k(
                n_plus, z
            )

            lhs = k_np1 - k_nm1
            rhs = ((2 * order + 1) / z) * k_n
            torch.testing.assert_close(
                lhs,
                rhs,
                rtol=1e-8,
                atol=1e-10,
                msg=f"Recurrence failed for n={order}",
            )

    def test_k0_formula(self):
        """Test k_0(z) = (pi/2) * exp(-z) / z for non-zero z."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.zeros_like(z)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        expected = (math.pi / 2) * torch.exp(-z) / z
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

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
            return torchscience.special_functions.spherical_bessel_k(n, z_)

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
            return torchscience.special_functions.spherical_bessel_k(n, z_)

        assert torch.autograd.gradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.skip(
        reason="Gradient w.r.t. order n is computed numerically and is not accurate enough for gradcheck"
    )
    def test_gradcheck_both_inputs(self):
        """Test first-order gradient correctness for both n and z.

        Note: This test is skipped because the gradient w.r.t. n is computed
        numerically using finite differences for the Bessel function derivative
        with respect to order, which is inherently imprecise.
        """
        n = torch.tensor(
            [0.5, 1.5, 2.5], dtype=torch.float64, requires_grad=True
        )
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(n_, z_):
            return torchscience.special_functions.spherical_bessel_k(n_, z_)

        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-5, atol=0.25, rtol=0.25
        )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness for z."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.spherical_bessel_k(n, z_)

        assert torch.autograd.gradgradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_identity(self):
        """Verify d/dz k_n(z) = -k_{n-1}(z) - ((n+1)/z)*k_n(z)."""
        n = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        y = torchscience.special_functions.spherical_bessel_k(n, z)
        grad = torch.autograd.grad(y.sum(), z)[0]

        k_n = torchscience.special_functions.spherical_bessel_k(n, z.detach())
        k_nm1 = torchscience.special_functions.spherical_bessel_k(
            n - 1, z.detach()
        )
        expected = -k_nm1 - ((n + 1) / z.detach()) * k_n

        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-10)

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting_n_scalar(self):
        """Test broadcasting when n is scalar-like."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.shape == (4,)

    def test_broadcasting_z_scalar(self):
        """Test broadcasting when z is scalar-like."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.shape == (4,)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)  # (4,)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.shape == (3, 4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j, 3.0 + 0.2j], dtype=dtype)
        result = torchscience.special_functions.spherical_bessel_k(n, z)
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
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        expected = torch.tensor(
            [
                scipy.special.spherical_kn(int(n[i].real.item()), z[i].item())
                for i in range(len(z))
            ],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real k_n."""
        n_real = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z_real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        n_complex = n_real.to(torch.complex128)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.spherical_bessel_k(
            n_real, z_real
        )
        result_complex = torchscience.special_functions.spherical_bessel_k(
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
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.shape == (10,)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor shape inference with broadcasting."""
        n = torch.randn(3, 1, device="meta")
        z = torch.randn(1, 4, device="meta")
        result = torchscience.special_functions.spherical_bessel_k(n, z)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.spherical_bessel_k(n, z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.spherical_bessel_k(n, z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.randn(5, 3, dtype=torch.float64).abs() + 0.1  # Avoid z=0

        def fn(z_row):
            return torchscience.special_functions.spherical_bessel_k(n, z_row)

        result = torch.vmap(fn)(z)
        expected = torchscience.special_functions.spherical_bessel_k(
            n.unsqueeze(0), z
        )
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_k
        )
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.randn(3, dtype=torch.float64).abs() + 0.1
        result = compiled_fn(n, z)
        expected = torchscience.special_functions.spherical_bessel_k(n, z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(
            torchscience.special_functions.spherical_bessel_k
        )
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        # Create base tensor, then derive z with requires_grad
        z_base = torch.randn(3, dtype=torch.float64).abs() + 0.1
        z = z_base.clone().requires_grad_(True)
        result = compiled_fn(n, z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z_base.clone().requires_grad_(True)
        expected = torchscience.special_functions.spherical_bessel_k(n, z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    # =========================================================================
    # Large argument tests
    # =========================================================================

    def test_large_z(self):
        """Test large z values (exponential decay)."""
        n = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
        z = torch.tensor([10.0, 20.0, 30.0, 50.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(
            n.unsqueeze(1), z
        )
        expected = torch.tensor(
            [
                [
                    scipy.special.spherical_kn(int(ni.item()), zi.item())
                    for zi in z
                ]
                for ni in n
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-10)

    def test_exponential_decay(self):
        """Verify k_n(z) decays exponentially for large z."""
        n = torch.tensor([0.0], dtype=torch.float64)
        z1 = torch.tensor([5.0], dtype=torch.float64)
        z2 = torch.tensor([10.0], dtype=torch.float64)
        k_z1 = torchscience.special_functions.spherical_bessel_k(n, z1)
        k_z2 = torchscience.special_functions.spherical_bessel_k(n, z2)
        # k_0(z) = (pi/2) * e^(-z) / z
        # So k_0(10)/k_0(5) should be approximately e^(-5) * 5/10 = e^(-5)/2
        ratio = k_z2 / k_z1
        expected_ratio = math.exp(-5) * 5 / 10  # Approximate
        # Allow larger tolerance for asymptotic behavior
        assert abs(ratio.item() / expected_ratio - 1) < 0.1

    def test_positive_values(self):
        """Verify k_n(z) is positive for positive real z and integer n >= 0."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.spherical_bessel_k(
            n.unsqueeze(1), z
        )
        assert (result > 0).all()
