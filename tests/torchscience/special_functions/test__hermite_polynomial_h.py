import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


def scipy_hermite_h(n: float, z: float) -> float:
    """Reference implementation using scipy."""
    return scipy.special.eval_hermite(int(n), z)


class TestHermitePolynomialH:
    """Tests for Physicists' Hermite polynomial."""

    def test_forward_integer_degrees(self):
        """Test forward pass for integer degrees 0-5."""
        z = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float64)

        # H_0(z) = 1
        n = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # H_1(z) = 2z
        n = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = 2 * z
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # H_2(z) = 4z^2 - 2
        n = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = 4 * z**2 - 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # H_3(z) = 8z^3 - 12z
        n = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = 8 * z**3 - 12 * z
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # H_4(z) = 16z^4 - 48z^2 + 12
        n = torch.tensor([4.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = 16 * z**4 - 48 * z**2 + 12
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # H_5(z) = 32z^5 - 160z^3 + 120z
        n = torch.tensor([5.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        expected = 32 * z**5 - 160 * z**3 + 120 * z
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.eval_hermite."""
        z_vals = torch.linspace(-2.0, 2.0, 20, dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5, 10]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hermite_polynomial_h(
                n, z_vals
            )

            expected_list = [scipy_hermite_h(n_val, z.item()) for z in z_vals]
            expected = torch.tensor(expected_list, dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Mismatch for n={n_val}",
            )

    def test_special_values_at_z_equals_0(self):
        """Test H_n(0) values.

        H_n(0) = 0 for odd n
        H_n(0) = (-1)^{n/2} * n! / (n/2)! for even n
        """
        z = torch.tensor([0.0], dtype=torch.float64)

        # Odd n: should be 0
        for n_val in [1, 3, 5, 7]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hermite_polynomial_h(n, z)
            torch.testing.assert_close(
                result,
                torch.zeros(1, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
                msg=f"H_{n_val}(0) should be 0",
            )

        # Even n: H_0(0)=1, H_2(0)=-2, H_4(0)=12, H_6(0)=-120
        expected_values = {0: 1.0, 2: -2.0, 4: 12.0, 6: -120.0}
        for n_val, expected_val in expected_values.items():
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hermite_polynomial_h(n, z)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"H_{n_val}(0) should be {expected_val}",
            )

    def test_parity_property(self):
        """Test H_n(-z) = (-1)^n H_n(z) for integer n."""
        z_vals = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result_pos = torchscience.special_functions.hermite_polynomial_h(
                n, z_vals
            )
            result_neg = torchscience.special_functions.hermite_polynomial_h(
                n, -z_vals
            )

            expected_relation = ((-1.0) ** n_val) * result_pos
            torch.testing.assert_close(
                result_neg,
                expected_relation,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Parity property failed for n={n_val}",
            )

    def test_recurrence_relation(self):
        """Test H_{n+1}(z) = 2z*H_n(z) - 2n*H_{n-1}(z)."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        for n in range(1, 10):
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
            n_plus_1 = torch.tensor([float(n + 1)], dtype=torch.float64)

            H_n = torchscience.special_functions.hermite_polynomial_h(
                n_tensor, z
            )
            H_n_minus_1 = torchscience.special_functions.hermite_polynomial_h(
                n_minus_1, z
            )
            H_n_plus_1 = torchscience.special_functions.hermite_polynomial_h(
                n_plus_1, z
            )

            # H_{n+1}(z) = 2z*H_n(z) - 2n*H_{n-1}(z)
            expected = 2 * z * H_n - 2 * n * H_n_minus_1

            torch.testing.assert_close(
                H_n_plus_1,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Recurrence relation failed at n={n}",
            )

    def test_gradcheck_z(self):
        """Test gradient correctness with respect to z."""
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.hermite_polynomial_h(n, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_n(self):
        """Test gradient correctness with respect to n."""
        z = torch.tensor([0.5], dtype=torch.float64)
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(n):
            return torchscience.special_functions.hermite_polynomial_h(n, z)

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_both(self):
        """Test gradient correctness with respect to both n and z."""
        z = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(n, z):
            return torchscience.special_functions.hermite_polynomial_h(n, z)

        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_analytical(self):
        """Test that dH_n/dz matches the analytical formula.

        dH_n/dz = 2n * H_{n-1}(z)
        For H_2(z) = 4z^2 - 2, we have dH_2/dz = 8z = 2*2*H_1(z) = 4*2z = 8z.
        """
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        result.backward()

        expected_grad = 8 * z.detach()  # dH_2/dz = 8z
        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        z = torch.empty(1, 5, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)

        # Should broadcast to (3, 5) shape
        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n and z."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)  # (1, 3)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.shape == (3, 3)

        # Verify some values
        # H_0(z) = 1 for all z
        torch.testing.assert_close(
            result[0, :],
            torch.ones(3, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        # H_1(z) = 2z
        torch.testing.assert_close(
            result[1, :], 2 * z.squeeze(), rtol=1e-10, atol=1e-10
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        z = torch.tensor([0.5], dtype=torch.float32)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.dtype == torch.float32

        expected = torch.tensor([-1.0], dtype=torch.float32)  # 4*0.25 - 2
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        z = torch.tensor([0.5], dtype=dtype)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        z_f32 = z.to(torch.float32)
        expected = torchscience.special_functions.hermite_polynomial_h(
            n_f32, z_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_large_degree(self):
        """Test with larger polynomial degrees."""
        z = torch.tensor([0.5], dtype=torch.float64)

        # Test up to n=10 where the implementation is accurate
        for n_val in [6, 8, 10]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.hermite_polynomial_h(n, z)
            expected = torch.tensor(
                [scipy_hermite_h(n_val, 0.5)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Large degree n={n_val} failed",
            )

    def test_large_z_values(self):
        """Test with larger z values."""
        n = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)

        # H_3(z) = 8z^3 - 12z
        expected = 8 * z**3 - 12 * z
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness with respect to z."""
        z = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.hermite_polynomial_h(n, z)

        # Second-order gradients
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_hermite(n, z):
            return torchscience.special_functions.hermite_polynomial_h(n, z)

        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = compiled_hermite(n, z)
        expected = torch.tensor([-1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize(
        "n_dtype,z_dtype,expected_dtype",
        [
            (torch.float32, torch.complex64, torch.complex64),
            (torch.float64, torch.complex128, torch.complex128),
            (torch.float64, torch.complex64, torch.complex128),  # promoted
        ],
    )
    def test_complex_dtypes(self, n_dtype, z_dtype, expected_dtype):
        """Test with complex dtypes (complex64 and complex128)."""
        # Test with complex z values
        n = torch.tensor([2.0], dtype=n_dtype)
        z = torch.tensor([0.5 + 0.1j, 0.3 - 0.2j], dtype=z_dtype)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.dtype == expected_dtype

        # Verify against analytical formula H_2(z) = 4z^2 - 2
        expected = 4 * z.to(expected_dtype) ** 2 - 2
        rtol = 1e-5 if expected_dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(result, expected, rtol=rtol, atol=rtol)

    def test_complex_n_and_z(self):
        """Test with both complex n and z."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex128)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.dtype == torch.complex128

        # Verify against analytical formula H_2(z) = 4z^2 - 2
        expected = 4 * z**2 - 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z_real = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.hermite_polynomial_h(
            n, z_real
        )
        result_complex = torchscience.special_functions.hermite_polynomial_h(
            n, z_complex
        )

        # Real part should match, imaginary part should be zero
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_negative_n_integer(self):
        """Test with negative integer n values.

        For negative integers, the Hermite polynomial is defined via the
        confluent hypergeometric function which is well-defined.
        """
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        # Just verify that the function doesn't crash for negative n
        n = torch.tensor([-1.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.shape == z.shape
        assert not torch.any(torch.isnan(result))

        n = torch.tensor([-2.0], dtype=torch.float64)
        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.shape == z.shape
        assert not torch.any(torch.isnan(result))

    def test_negative_n_non_integer(self):
        """Test with negative non-integer n values."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        n = torch.tensor([-0.5], dtype=torch.float64)

        result = torchscience.special_functions.hermite_polynomial_h(n, z)

        # Just verify that the function doesn't crash
        assert result.shape == z.shape
        assert not torch.any(torch.isnan(result))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.hermite_polynomial_h(
            n.cpu(), z.cpu()
        )
        torch.testing.assert_close(
            result.cpu(), result_cpu, rtol=1e-12, atol=1e-12
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward(self):
        """Test backward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        z = torch.tensor(
            [0.3, 0.5, 0.7],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )

        result = torchscience.special_functions.hermite_polynomial_h(n, z)
        result.sum().backward()

        # Compare gradient to CPU
        z_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.hermite_polynomial_h(
            n.cpu(), z_cpu
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            z.grad.cpu(), z_cpu.grad, rtol=1e-10, atol=1e-10
        )

    def test_derivative_recurrence(self):
        """Test that dH_n/dz = 2n * H_{n-1}(z)."""
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )

        for n_val in [1, 2, 3, 4, 5]:
            z.grad = None
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n_val - 1)], dtype=torch.float64)

            result = torchscience.special_functions.hermite_polynomial_h(n, z)
            result.sum().backward()

            H_n_minus_1 = torchscience.special_functions.hermite_polynomial_h(
                n_minus_1, z.detach()
            )
            expected_grad = 2 * n_val * H_n_minus_1

            torch.testing.assert_close(
                z.grad,
                expected_grad,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Derivative recurrence failed for n={n_val}",
            )

    def test_second_derivative(self):
        """Test that d^2H_n/dz^2 = 4n(n-1) * H_{n-2}(z)."""
        z = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([3.0], dtype=torch.float64)

        # Compute first derivative with create_graph=True to enable second derivative
        y = torchscience.special_functions.hermite_polynomial_h(n, z)
        dH_dz = torch.autograd.grad(y.sum(), z, create_graph=True)[0]

        # Compute second derivative
        d2H_dz2 = torch.autograd.grad(dH_dz.sum(), z)[0]

        # Expected: 4n(n-1) * H_{n-2}(z) = 4*3*2 * H_1(z) = 24 * 2z = 48z
        n_minus_2 = torch.tensor([1.0], dtype=torch.float64)
        H_n_minus_2 = torchscience.special_functions.hermite_polynomial_h(
            n_minus_2, z.detach()
        )
        expected = 4 * 3 * 2 * H_n_minus_2

        torch.testing.assert_close(d2H_dz2, expected, rtol=1e-4, atol=1e-4)
