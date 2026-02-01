import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestDawson:
    """Tests for Dawson's integral D(z) = exp(-z^2) * integral from 0 to z of exp(t^2) dt."""

    def test_dawson_at_origin(self):
        """Test D(0) = 0."""
        z = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.dawson(z)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_dawson_odd_function(self):
        """Test D(-z) = -D(z) (odd function)."""
        z_vals = [0.5, 1.0, 2.0, 3.0]
        for z_val in z_vals:
            z = torch.tensor([z_val], dtype=torch.float64)
            result_pos = torchscience.special_functions.dawson(z)
            result_neg = torchscience.special_functions.dawson(-z)
            torch.testing.assert_close(
                result_neg, -result_pos, rtol=1e-10, atol=1e-12
            )

    def test_dawson_maximum(self):
        """Test that D(x) has maximum at x ~ 0.924 where D(x) ~ 0.541."""
        # Check that D(0.924) ~ 0.541
        z = torch.tensor([0.924], dtype=torch.float64)
        result = torchscience.special_functions.dawson(z)
        expected = 0.541
        assert abs(result.item() - expected) < 0.001, (
            f"Expected D(0.924) ~ 0.541, got {result.item()}"
        )

    def test_dawson_asymptotic(self):
        """Test asymptotic behavior: D(x) -> 1/(2x) as x -> infinity."""
        x_vals = [10.0, 20.0, 50.0, 100.0]
        for x_val in x_vals:
            z = torch.tensor([x_val], dtype=torch.float64)
            result = torchscience.special_functions.dawson(z)
            expected = 1.0 / (2.0 * x_val)
            rel_error = abs(result.item() - expected) / expected
            assert rel_error < 0.1, (
                f"Asymptotic test failed at x={x_val}: "
                f"got {result.item()}, expected ~{expected}, rel_error={rel_error}"
            )

    def test_scipy_reference(self):
        """Compare against scipy.special.dawsn at various points."""
        try:
            from scipy.special import dawsn
        except ImportError:
            pytest.skip("scipy not installed")

        x_vals = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
        for x_val in x_vals:
            z = torch.tensor([x_val], dtype=torch.float64)
            result = torchscience.special_functions.dawson(z)
            expected = dawsn(x_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_scipy_reference_negative(self):
        """Compare against scipy.special.dawsn at negative points."""
        try:
            from scipy.special import dawsn
        except ImportError:
            pytest.skip("scipy not installed")

        x_vals = [-0.5, -1.0, -2.0, -3.0]
        for x_val in x_vals:
            z = torch.tensor([x_val], dtype=torch.float64)
            result = torchscience.special_functions.dawson(z)
            expected = dawsn(x_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_gradient_formula(self):
        """Test that gradient matches analytical formula D'(z) = 1 - 2z*D(z)."""
        z_vals = [0.5, 1.0, 2.0, 3.0]

        for z_val in z_vals:
            z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)
            D_z = torchscience.special_functions.dawson(z)

            # Compute gradient via autograd
            D_z.backward()
            grad_autograd = z.grad.item()

            # Compute expected gradient: D'(z) = 1 - 2z*D(z)
            expected_grad = 1.0 - 2.0 * z_val * D_z.detach().item()

            assert (
                abs(grad_autograd - expected_grad)
                < 1e-6 * abs(expected_grad) + 1e-8
            ), (
                f"Gradient mismatch at z={z_val}: "
                f"got {grad_autograd}, expected {expected_grad}"
            )

    def test_gradcheck_real_input(self):
        """Test gradients for real input."""
        z = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            torchscience.special_functions.dawson,
            (z,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck_real_input(self):
        """Test second-order gradients for real input."""
        z = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.dawson,
            (z,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_second_derivative_formula(self):
        """Verify the second derivative formula D''(z) = (4z^2 - 2)*D(z) - 2z."""
        z_val = 1.5

        # Compute D(z) and analytical second derivative
        z = torch.tensor([z_val], dtype=torch.float64)
        D_z = torchscience.special_functions.dawson(z).item()

        # D''(z) = (4z^2 - 2)*D(z) - 2z
        D_double_prime = (4 * z_val * z_val - 2) * D_z - 2 * z_val

        # Numerical second derivative
        eps = 1e-5

        def get_D(zz):
            return torchscience.special_functions.dawson(
                torch.tensor([zz], dtype=torch.float64)
            ).item()

        num_D_double_prime = (
            get_D(z_val + eps) - 2 * get_D(z_val) + get_D(z_val - eps)
        ) / (eps * eps)

        rel_err = abs(D_double_prime - num_D_double_prime) / abs(
            num_D_double_prime
        )
        assert rel_err < 2e-3, (
            f"Second derivative formula mismatch: "
            f"analytical={D_double_prime}, numerical={num_D_double_prime}"
        )

    def test_complex_support(self):
        """Test that complex input works."""
        z = torch.tensor([1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.dawson(z)
        assert result.is_complex(), "Expected complex output for complex input"
        assert result.shape == z.shape

    def test_complex_real_axis(self):
        """Test that complex with zero imaginary matches real result."""
        x_vals = [0.5, 1.0, 2.0]
        for x_val in x_vals:
            z_real = torch.tensor([x_val], dtype=torch.float64)
            z_complex = torch.tensor([x_val + 0j], dtype=torch.complex128)

            result_real = torchscience.special_functions.dawson(z_real)
            result_complex = torchscience.special_functions.dawson(z_complex)

            torch.testing.assert_close(
                result_complex.real,
                result_real.to(torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )
            assert abs(result_complex.imag.item()) < 1e-10, (
                f"Expected zero imaginary part for real input, got {result_complex.imag.item()}"
            )

    def test_faddeeva_relation_real(self):
        """Test D(x) = sqrt(pi)/2 * Im[w(x)] for real x."""
        x_vals = [0.5, 1.0, 2.0, 3.0]
        sqrt_pi_over_2 = math.sqrt(math.pi) / 2

        for x_val in x_vals:
            x = torch.tensor([x_val], dtype=torch.float64)
            z = torch.tensor([x_val + 0j], dtype=torch.complex128)

            D_x = torchscience.special_functions.dawson(x)
            w_x = torchscience.special_functions.faddeeva_w(z)

            # D(x) = sqrt(pi)/2 * Im[w(x)]
            D_from_faddeeva = sqrt_pi_over_2 * w_x.imag

            torch.testing.assert_close(
                D_x,
                D_from_faddeeva.to(torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    def test_batch_computation(self):
        """Test batch computation matches element-wise."""
        try:
            from scipy.special import dawsn
        except ImportError:
            pytest.skip("scipy not installed")

        x_vals = [0.5, 1.0, 2.0, 3.0]
        z = torch.tensor(x_vals, dtype=torch.float64)
        result = torchscience.special_functions.dawson(z)

        for i, x_val in enumerate(x_vals):
            expected = dawsn(x_val)
            torch.testing.assert_close(
                result[i],
                torch.tensor(expected, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_meta_tensor(self):
        """Test meta tensor support."""
        z = torch.tensor([1.0], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.dawson(z)
        assert result.device.type == "meta"
        assert result.shape == z.shape
        assert result.dtype == z.dtype

    def test_compile_smoke(self):
        """Test that torch.compile works."""
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        compiled_fn = torch.compile(torchscience.special_functions.dawson)
        result = compiled_fn(z)

        expected = torchscience.special_functions.dawson(z)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_vmap(self):
        """Test vmap support."""
        z = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)

        def fn(x):
            return torchscience.special_functions.dawson(x)

        result = torch.vmap(fn)(z)

        expected = torchscience.special_functions.dawson(z)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_nan_input(self):
        """Test NaN input propagates to output."""
        z = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.dawson(z)
        assert torch.isnan(result).all()

    def test_dtype_preservation_float32(self):
        """Test float32 dtype is preserved."""
        z = torch.tensor([1.0], dtype=torch.float32)
        result = torchscience.special_functions.dawson(z)
        assert result.dtype == torch.float32

    def test_dtype_preservation_float64(self):
        """Test float64 dtype is preserved."""
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.dawson(z)
        assert result.dtype == torch.float64

    def test_gradcheck_complex_input(self):
        """Test gradients for complex input."""
        z = torch.tensor(
            [1.0 + 0.5j], dtype=torch.complex128, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.dawson,
            (z,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )
