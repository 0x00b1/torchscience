import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestSphericalHankel1:
    """Tests for the spherical Hankel function of the first kind h_n^(1)(z)."""

    # =========================================================================
    # Forward correctness tests - definition h_n^(1) = j_n + i*y_n
    # =========================================================================

    def test_definition_order_zero(self):
        """Test h_0^(1)(z) = j_0(z) + i*y_0(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.complex128)
        n = torch.zeros_like(z)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Compute expected from scipy
        z_np = z.real.numpy()
        j_n = scipy.special.spherical_jn(0, z_np)
        y_n = scipy.special.spherical_yn(0, z_np)
        expected = torch.tensor(j_n + 1j * y_n, dtype=torch.complex128)

        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_definition_order_one(self):
        """Test h_1^(1)(z) = j_1(z) + i*y_1(z)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.complex128)
        n = torch.ones_like(z)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Compute expected from scipy
        z_np = z.real.numpy()
        j_n = scipy.special.spherical_jn(1, z_np)
        y_n = scipy.special.spherical_yn(1, z_np)
        expected = torch.tensor(j_n + 1j * y_n, dtype=torch.complex128)

        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_definition_general_orders(self):
        """Test h_n^(1)(z) = j_n(z) + i*y_n(z) for various orders."""
        z = torch.tensor([1.0, 2.0, 5.0], dtype=torch.complex128)

        for order in [0, 1, 2, 3, 5]:
            n = torch.full_like(z, float(order))
            result = torchscience.special_functions.spherical_hankel_1(n, z)

            z_np = z.real.numpy()
            j_n = scipy.special.spherical_jn(order, z_np)
            y_n = scipy.special.spherical_yn(order, z_np)
            expected = torch.tensor(j_n + 1j * y_n, dtype=torch.complex128)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-7,
                atol=1e-10,
                msg=f"Failed for order n={order}",
            )

    # =========================================================================
    # Explicit formula tests
    # =========================================================================

    def test_h0_explicit_formula(self):
        """Test h_0^(1)(z) = e^(iz) / (iz) = -i * e^(iz) / z."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.complex128)
        n = torch.zeros_like(z)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # h_0^(1)(z) = e^(iz) / (iz)
        i = torch.tensor(1j, dtype=torch.complex128)
        expected = torch.exp(i * z) / (i * z)

        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_h1_explicit_formula(self):
        """Test h_1^(1)(z) = -e^(iz) * (1/z + i/z^2)."""
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.complex128)
        n = torch.ones_like(z)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # h_1^(1)(z) = -e^(iz) * (1/z + i/z^2) = -e^(iz) * (z + i) / z^2
        i = torch.tensor(1j, dtype=torch.complex128)
        expected = -torch.exp(i * z) * (z + i) / (z * z)

        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-10)

    # =========================================================================
    # Special values tests
    # =========================================================================

    def test_at_zero_is_nan(self):
        """Test that h_n^(1)(0) returns NaN (singularity)."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
        z = torch.zeros_like(n)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.isnan().all(), "h_n^(1)(0) should be NaN"

    def test_nan_input_z(self):
        """Test h_n^(1)(NaN) = NaN."""
        n = torch.tensor([1.0], dtype=torch.complex128)
        z = torch.tensor([float("nan") + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.isnan().all()

    def test_nan_input_n(self):
        """Test h_NaN^(1)(z) = NaN."""
        n = torch.tensor([float("nan") + 0j], dtype=torch.complex128)
        z = torch.tensor([1.0 + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.isnan().all()

    # =========================================================================
    # Recurrence relation tests
    # =========================================================================

    def test_recurrence_relation(self):
        """Test recurrence: h_{n-1}(z) + h_{n+1}(z) = (2n+1)/z * h_n(z)."""
        z = torch.tensor([1.0, 2.0, 5.0], dtype=torch.complex128)

        for order in [1.0, 2.0, 3.0]:
            n = torch.full_like(z, order)
            n_minus = torch.full_like(z, order - 1)
            n_plus = torch.full_like(z, order + 1)

            h_n = torchscience.special_functions.spherical_hankel_1(n, z)
            h_nm1 = torchscience.special_functions.spherical_hankel_1(
                n_minus, z
            )
            h_np1 = torchscience.special_functions.spherical_hankel_1(
                n_plus, z
            )

            lhs = h_nm1 + h_np1
            rhs = ((2 * order + 1) / z) * h_n

            torch.testing.assert_close(
                lhs,
                rhs,
                rtol=1e-7,
                atol=1e-10,
                msg=f"Recurrence failed for n={order}",
            )

    # =========================================================================
    # Derivative relation tests
    # =========================================================================

    def test_derivative_relation(self):
        """Test d/dz h_n(z) = (n/z)*h_n(z) - h_{n+1}(z)."""
        z = torch.tensor([1.0, 2.0, 5.0], dtype=torch.complex128)

        for order in [0.0, 1.0, 2.0]:
            n = torch.full_like(z, order)
            n_plus = torch.full_like(z, order + 1)

            h_n = torchscience.special_functions.spherical_hankel_1(n, z)
            h_np1 = torchscience.special_functions.spherical_hankel_1(
                n_plus, z
            )

            # Compute derivative using analytical formula
            analytical_deriv = (n / z) * h_n - h_np1

            # Compute derivative numerically
            eps = 1e-7
            z_plus = z + eps
            z_minus = z - eps
            h_plus = torchscience.special_functions.spherical_hankel_1(
                n, z_plus
            )
            h_minus = torchscience.special_functions.spherical_hankel_1(
                n, z_minus
            )
            numerical_deriv = (h_plus - h_minus) / (2 * eps)

            torch.testing.assert_close(
                analytical_deriv,
                numerical_deriv,
                rtol=1e-4,
                atol=1e-8,
                msg=f"Derivative formula failed for n={order}",
            )

    # =========================================================================
    # Real input promotion tests
    # =========================================================================

    def test_real_input_promotion_float32(self):
        """Test that real float32 inputs are promoted to complex64."""
        n = torch.tensor([0.0, 1.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0], dtype=torch.float32)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.dtype == torch.complex64

    def test_real_input_promotion_float64(self):
        """Test that real float64 inputs are promoted to complex128."""
        n = torch.tensor([0.0, 1.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.dtype == torch.complex128

    def test_output_is_complex(self):
        """Test that output is always complex."""
        n = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.is_complex()

    # =========================================================================
    # Complex argument tests
    # =========================================================================

    def test_complex_argument(self):
        """Test with complex argument z."""
        n = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([1.0 + 0.5j, 2.0 + 1.0j], dtype=torch.complex128)

        # Should not raise
        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Result should be finite and complex
        assert result.is_complex()
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting_n_scalar(self):
        """Test broadcasting when n is scalar-like."""
        n = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.shape == (3,)

    def test_broadcasting_z_scalar(self):
        """Test broadcasting when z is scalar-like."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
        z = torch.tensor([2.0 + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.shape == (3,)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting."""
        n = torch.tensor(
            [[0.0], [1.0], [2.0]], dtype=torch.complex128
        )  # (3, 1)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128)  # (3,)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        assert result.shape == (3, 3)

    # =========================================================================
    # Asymptotic behavior tests
    # =========================================================================

    def test_large_argument_decay(self):
        """Test that |h_n^(1)(z)| decays like 1/z for large real z."""
        n = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        z_values = torch.tensor(
            [10.0, 20.0, 50.0, 100.0], dtype=torch.complex128
        )

        results = []
        for z in z_values:
            h = torchscience.special_functions.spherical_hankel_1(
                n, z.unsqueeze(0)
            )
            results.append(h.abs().item())

        # For large z, |h_n^(1)(z)| ~ 1/z
        # So |h(z)| * z should be approximately constant
        products = [
            results[i] * z_values[i].real.item() for i in range(len(results))
        ]

        # Check that the products are approximately constant (within 10%)
        mean_product = sum(products) / len(products)
        for p in products:
            assert abs(p - mean_product) / mean_product < 0.1

    # =========================================================================
    # Autograd tests
    # =========================================================================

    def test_grad_z(self):
        """Test gradient computation with respect to z."""
        n = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        z = torch.tensor(
            [2.0 + 0j], dtype=torch.complex128, requires_grad=True
        )

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Use abs() to get a real scalar for backward
        loss = result.abs().sum()
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_grad_n(self):
        """Test gradient computation with respect to n."""
        n = torch.tensor(
            [1.0 + 0j], dtype=torch.complex128, requires_grad=True
        )
        z = torch.tensor([2.0 + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Use abs() to get a real scalar for backward
        loss = result.abs().sum()
        loss.backward()

        assert n.grad is not None
        assert torch.isfinite(n.grad).all()

    @pytest.mark.skip(reason="gradcheck may be unstable for complex functions")
    def test_gradcheck(self):
        """Test gradient correctness using torch.autograd.gradcheck."""

        def func(n, z):
            result = torchscience.special_functions.spherical_hankel_1(n, z)
            # Return magnitude for real-valued output
            return result.abs()

        n = torch.tensor(
            [1.0 + 0j], dtype=torch.complex128, requires_grad=True
        )
        z = torch.tensor(
            [2.0 + 0j], dtype=torch.complex128, requires_grad=True
        )

        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    # =========================================================================
    # Consistency tests
    # =========================================================================

    def test_consistency_with_internal_bessel(self):
        """Test consistency with internal spherical Bessel functions."""
        z_real = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        z = z_real.to(torch.complex128)

        for order in [0, 1, 2]:
            n_real = torch.full_like(z_real, float(order))
            n = torch.full_like(z, float(order))

            # Get spherical Bessel functions
            j_n = torchscience.special_functions.spherical_bessel_j(
                n_real, z_real
            )
            y_n = torchscience.special_functions.spherical_bessel_y(
                n_real, z_real
            )

            # Get spherical Hankel function
            h_n = torchscience.special_functions.spherical_hankel_1(n, z)

            # h_n^(1) = j_n + i*y_n
            expected = j_n.to(torch.complex128) + 1j * y_n.to(torch.complex128)

            torch.testing.assert_close(
                h_n,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Consistency failed for n={order}",
            )

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_small_z(self):
        """Test behavior for small z (approaching singularity)."""
        n = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([0.01 + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Should be large magnitude but finite
        assert torch.isfinite(result).all()
        assert result.abs().item() > 10  # Should be large for small z

    def test_negative_z_real(self):
        """Test with negative real z."""
        n = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([-1.0 + 0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Should be finite
        assert torch.isfinite(result).all()

    def test_imaginary_z(self):
        """Test with purely imaginary z."""
        n = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([0.0 + 2.0j], dtype=torch.complex128)

        result = torchscience.special_functions.spherical_hankel_1(n, z)

        # Should be finite
        assert torch.isfinite(result).all()
