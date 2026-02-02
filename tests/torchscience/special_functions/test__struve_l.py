import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestStruveL:
    """Tests for the modified Struve function L_n(z) of general order n."""

    # =========================================================================
    # Forward correctness tests - integer orders
    # =========================================================================

    def test_order_zero_matches_l0(self):
        """Test L_0(z) via struve_l(0, z) matches struve_l_0(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.zeros_like(z)
        result = torchscience.special_functions.struve_l(n, z)
        expected = torchscience.special_functions.struve_l_0(z)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_order_one_matches_l1(self):
        """Test L_1(z) via struve_l(1, z) matches struve_l_1(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = torch.ones_like(z)
        result = torchscience.special_functions.struve_l(n, z)
        expected = torchscience.special_functions.struve_l_1(z)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_scipy_agreement_integer_orders(self):
        """Test agreement with scipy.special.modstruve for integer orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0, 1, 2, 3]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.struve_l(n, z)
            expected = torch.tensor(
                [scipy.special.modstruve(order, x.item()) for x in z],
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
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0.5, 1.5, 2.5]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.struve_l(n, z)
            expected = torch.tensor(
                [scipy.special.modstruve(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-8,
                msg=f"Failed for order n={order}",
            )

    def test_scipy_agreement_fractional_order(self):
        """Test agreement with scipy for general fractional orders."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0.3, 1.7, 2.25]:
            n = torch.full_like(z, order)
            result = torchscience.special_functions.struve_l(n, z)
            expected = torch.tensor(
                [scipy.special.modstruve(order, x.item()) for x in z],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-8,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Special values tests
    # =========================================================================

    def test_at_zero(self):
        """Test L_n(0) = 0 for n >= -1."""
        z = torch.tensor([0.0], dtype=torch.float64)
        for order in [0.0, 0.5, 1.0, 2.0, 5.0]:
            n = torch.tensor([order], dtype=torch.float64)
            result = torchscience.special_functions.struve_l(n, z)
            torch.testing.assert_close(
                result,
                torch.tensor([0.0], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_nan_input(self):
        """Test that NaN inputs produce NaN outputs."""
        n = torch.tensor([float("nan")], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.struve_l(n, z)
        assert torch.isnan(result).all()

        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.struve_l(n, z)
        assert torch.isnan(result).all()

    # =========================================================================
    # Symmetry tests
    # =========================================================================

    def test_negative_z_integer_order(self):
        """Test L_n(-z) = (-1)^(n+1) * L_n(z) for integer n."""
        z = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        for order in [0, 1, 2, 3, 4]:
            n = torch.full_like(z, float(order))
            result_pos = torchscience.special_functions.struve_l(n, z)
            result_neg = torchscience.special_functions.struve_l(n, -z)
            expected_sign = (-1) ** (order + 1)
            torch.testing.assert_close(
                result_neg,
                expected_sign * result_pos,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for n={order}",
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
            return torchscience.special_functions.struve_l(n, z_)

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
            return torchscience.special_functions.struve_l(n, z_)

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
            return torchscience.special_functions.struve_l(n_, z_)

        # Gradient w.r.t. n is numerical; may need relaxed tolerance
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
            return torchscience.special_functions.struve_l(n, z_)

        assert torch.autograd.gradgradcheck(
            func, z, eps=1e-6, atol=1e-3, rtol=1e-3
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting_n_scalar(self):
        """Test broadcasting when n is scalar-like."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.struve_l(n, z)
        assert result.shape == (4,)

    def test_broadcasting_z_scalar(self):
        """Test broadcasting when z is scalar-like."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.struve_l(n, z)
        assert result.shape == (4,)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)  # (4,)
        result = torchscience.special_functions.struve_l(n, z)
        assert result.shape == (3, 4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.struve_l(n, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j, 3.0 + 0.2j], dtype=dtype)
        result = torchscience.special_functions.struve_l(n, z)
        assert result.dtype == dtype

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real L_n."""
        n_real = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z_real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        n_complex = n_real.to(torch.complex128)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.struve_l(n_real, z_real)
        result_complex = torchscience.special_functions.struve_l(
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
        result = torchscience.special_functions.struve_l(n, z)
        assert result.shape == (10,)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor shape inference with broadcasting."""
        n = torch.randn(3, 1, device="meta")
        z = torch.randn(1, 4, device="meta")
        result = torchscience.special_functions.struve_l(n, z)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.struve_l(n, z)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.struve_l(n, z)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.randn(5, 3, dtype=torch.float64).abs() + 0.1

        def fn(z_row):
            return torchscience.special_functions.struve_l(n, z_row)

        result = torch.vmap(fn)(z)
        expected = torchscience.special_functions.struve_l(n.unsqueeze(0), z)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.struve_l)
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.randn(3, dtype=torch.float64).abs() + 0.1
        result = compiled_fn(n, z)
        expected = torchscience.special_functions.struve_l(n, z)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.struve_l)
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = (torch.randn(3, dtype=torch.float64).abs() + 0.1).requires_grad_(
            True
        )
        result = compiled_fn(n, z)
        result.sum().backward()
        assert z.grad is not None
        # Verify gradient matches uncompiled version
        z2 = z.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.struve_l(n, z2)
        expected.sum().backward()
        torch.testing.assert_close(z.grad, z2.grad)

    # =========================================================================
    # Relationship to struve_h tests
    # =========================================================================

    def test_positive_values_grow(self):
        """Test that L_n(z) grows for positive z (unlike oscillating H_n)."""
        n = torch.tensor([0.0, 1.0], dtype=torch.float64)
        z1 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        z2 = torch.tensor([5.0, 5.0], dtype=torch.float64)
        l1 = torchscience.special_functions.struve_l(n, z1)
        l2 = torchscience.special_functions.struve_l(n, z2)
        # L_n grows with z for positive z
        assert (l2 > l1).all()
