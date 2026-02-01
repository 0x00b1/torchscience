import math

import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


def scipy_legendre_p(n: float, z: float) -> float:
    """Reference implementation using scipy."""
    return scipy.special.eval_legendre(n, z)


class TestLegendrePolynomialP:
    """Tests for Legendre polynomial of the first kind."""

    def test_forward_integer_degrees(self):
        """Test forward pass for integer degrees 0-5."""
        z = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float64)

        # P_0(z) = 1
        n = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # P_1(z) = z
        n = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        torch.testing.assert_close(result, z, rtol=1e-10, atol=1e-10)

        # P_2(z) = (3z^2 - 1)/2
        n = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = (3 * z**2 - 1) / 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # P_3(z) = (5z^3 - 3z)/2
        n = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = (5 * z**3 - 3 * z) / 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # P_4(z) = (35z^4 - 30z^2 + 3)/8
        n = torch.tensor([4.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = (35 * z**4 - 30 * z**2 + 3) / 8
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # P_5(z) = (63z^5 - 70z^3 + 15z)/8
        n = torch.tensor([5.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = (63 * z**5 - 70 * z**3 + 15 * z) / 8
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.eval_legendre."""
        z_vals = torch.linspace(-0.99, 0.99, 20, dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5, 10]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_p(
                n, z_vals
            )

            expected_list = [scipy_legendre_p(n_val, z.item()) for z in z_vals]
            expected = torch.tensor(expected_list, dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Mismatch for n={n_val}",
            )

    def test_special_values_at_z_equals_1(self):
        """Test P_n(1) = 1 for all n >= 0."""
        z = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.tensor([1.0], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 5, 10, 20]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_p(n, z)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"P_{n_val}(1) should be 1",
            )

    def test_special_values_at_z_equals_minus_1(self):
        """Test P_n(-1) = (-1)^n for integer n >= 0."""
        z = torch.tensor([-1.0], dtype=torch.float64)

        # For small n, we can use tight tolerances
        for n_val in [0, 1, 2, 3, 4, 5, 10]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_p(n, z)
            expected = torch.tensor([(-1.0) ** n_val], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"P_{n_val}(-1) should be (-1)^{n_val}",
            )

    def test_parity_property(self):
        """Test P_n(-z) = (-1)^n P_n(z) for integer n."""
        z_vals = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result_pos = torchscience.special_functions.legendre_polynomial_p(
                n, z_vals
            )
            result_neg = torchscience.special_functions.legendre_polynomial_p(
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

    def test_bonnet_recurrence(self):
        """Test (n+1)P_{n+1}(z) = (2n+1)zP_n(z) - nP_{n-1}(z)."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        for n in range(1, 10):
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
            n_plus_1 = torch.tensor([float(n + 1)], dtype=torch.float64)

            P_n = torchscience.special_functions.legendre_polynomial_p(
                n_tensor, z
            )
            P_n_minus_1 = torchscience.special_functions.legendre_polynomial_p(
                n_minus_1, z
            )
            P_n_plus_1 = torchscience.special_functions.legendre_polynomial_p(
                n_plus_1, z
            )

            left = (n + 1) * P_n_plus_1
            right = (2 * n + 1) * z * P_n - n * P_n_minus_1

            torch.testing.assert_close(
                left,
                right,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Bonnet recurrence failed at n={n}",
            )

    def test_gradcheck_z(self):
        """Test gradient correctness with respect to z."""
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.legendre_polynomial_p(n, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_n(self):
        """Test gradient correctness with respect to n."""
        z = torch.tensor([0.5], dtype=torch.float64)
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(n):
            return torchscience.special_functions.legendre_polynomial_p(n, z)

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_both(self):
        """Test gradient correctness with respect to both n and z."""
        z = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(n, z):
            return torchscience.special_functions.legendre_polynomial_p(n, z)

        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_analytical(self):
        """Test that dP_n/dz matches the analytical formula.

        For P_2(z) = (3z^2 - 1)/2, we have dP_2/dz = 3z.
        """
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        result.backward()

        expected_grad = 3 * z.detach()  # dP_2/dz = 3z
        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        z = torch.empty(1, 5, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)

        # Should broadcast to (3, 5) shape
        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n and z."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)  # (1, 3)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.shape == (3, 3)

        # Verify some values
        # P_0(z) = 1 for all z
        torch.testing.assert_close(
            result[0, :],
            torch.ones(3, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        # P_1(z) = z
        torch.testing.assert_close(
            result[1, :], z.squeeze(), rtol=1e-10, atol=1e-10
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        z = torch.tensor([0.5], dtype=torch.float32)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.dtype == torch.float32

        expected = torch.tensor([-0.125], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        z = torch.tensor([0.5], dtype=dtype)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        z_f32 = z.to(torch.float32)
        expected = torchscience.special_functions.legendre_polynomial_p(
            n_f32, z_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_non_integer_degree(self):
        """Test non-integer degree (generalized Legendre function)."""
        n = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)

        # Compare to scipy
        expected = torch.tensor(
            [scipy_legendre_p(0.5, zi) for zi in z.tolist()],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_large_degree(self):
        """Test with larger polynomial degrees.

        Note: The hypergeometric implementation has limited accuracy for
        large degrees (n > 20). For high-precision needs with large n,
        consider using scipy.special.eval_legendre.
        """
        z = torch.tensor([0.5], dtype=torch.float64)

        # Test up to n=20 where the implementation is accurate
        for n_val in [10, 15, 20]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.legendre_polynomial_p(n, z)
            expected = torch.tensor(
                [scipy_legendre_p(n_val, 0.5)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Large degree n={n_val} failed",
            )

    def test_z_outside_standard_domain(self):
        """Test z outside the standard domain [-1, 1]."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)

        # P_2(z) = (3z^2 - 1)/2, valid for all z
        expected = (3 * z**2 - 1) / 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_zeros_at_known_locations(self):
        """Test that P_n has n zeros in (-1, 1).

        The zeros of P_n are the Gauss-Legendre quadrature nodes.
        """
        # P_2 has zeros at +-1/sqrt(3) ~ +-0.577
        n = torch.tensor([2.0], dtype=torch.float64)
        z_zeros = torch.tensor(
            [-1 / math.sqrt(3), 1 / math.sqrt(3)], dtype=torch.float64
        )
        result = torchscience.special_functions.legendre_polynomial_p(
            n, z_zeros
        )
        torch.testing.assert_close(
            result, torch.zeros(2, dtype=torch.float64), rtol=1e-10, atol=1e-10
        )

        # P_3 has zeros at 0 and +-sqrt(3/5) ~ +-0.775
        n = torch.tensor([3.0], dtype=torch.float64)
        z_zeros = torch.tensor(
            [0.0, -math.sqrt(3 / 5), math.sqrt(3 / 5)], dtype=torch.float64
        )
        result = torchscience.special_functions.legendre_polynomial_p(
            n, z_zeros
        )
        torch.testing.assert_close(
            result, torch.zeros(3, dtype=torch.float64), rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.legendre_polynomial_p(
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

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        result.sum().backward()

        # Compare gradient to CPU
        z_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.legendre_polynomial_p(
            n.cpu(), z_cpu
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            z.grad.cpu(), z_cpu.grad, rtol=1e-10, atol=1e-10
        )

    def test_orthogonality_relation(self):
        """Test orthogonality: integral of P_m * P_n over [-1,1] = 2/(2n+1) * delta_{mn}.

        We use a simple numerical integration to verify.
        """
        # Use a large number of uniform points for trapezoidal integration
        x = torch.linspace(-0.9999, 0.9999, 1000, dtype=torch.float64)
        dx = x[1] - x[0]

        for m_val in [0, 1, 2, 3]:
            for n_val in [0, 1, 2, 3]:
                m = torch.tensor([float(m_val)], dtype=torch.float64)
                n = torch.tensor([float(n_val)], dtype=torch.float64)

                P_m = torchscience.special_functions.legendre_polynomial_p(
                    m, x
                )
                P_n = torchscience.special_functions.legendre_polynomial_p(
                    n, x
                )

                integral = torch.sum(P_m * P_n) * dx

                if m_val == n_val:
                    expected = 2.0 / (2 * n_val + 1)
                else:
                    expected = 0.0

                # Use relaxed tolerance due to numerical integration error
                torch.testing.assert_close(
                    integral.item(),
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                    msg=f"Orthogonality failed for m={m_val}, n={n_val}",
                )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness with respect to z."""
        z = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.legendre_polynomial_p(n, z)

        # Second-order gradients
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_legendre(n, z):
            return torchscience.special_functions.legendre_polynomial_p(n, z)

        n = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = compiled_legendre(n, z)
        expected = torch.tensor([-0.125], dtype=torch.float64)
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

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.dtype == expected_dtype

        # Verify against analytical formula P_2(z) = (3z^2 - 1)/2
        expected = (3 * z.to(expected_dtype) ** 2 - 1) / 2
        rtol = 1e-5 if expected_dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(result, expected, rtol=rtol, atol=rtol)

    def test_complex_n_and_z(self):
        """Test with both complex n and z."""
        n = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex128)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        assert result.dtype == torch.complex128

        # Verify against analytical formula P_2(z) = (3z^2 - 1)/2
        expected = (3 * z**2 - 1) / 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        z_real = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.legendre_polynomial_p(
            n, z_real
        )
        result_complex = torchscience.special_functions.legendre_polynomial_p(
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

        For negative integer n, P_n(z) can be defined via:
        P_{-n-1}(z) = P_n(z) for integer n >= 0
        """
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        # P_{-1}(z) = P_0(z) = 1
        n = torch.tensor([-1.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

        # P_{-2}(z) = P_1(z) = z
        n = torch.tensor([-2.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        torch.testing.assert_close(result, z, rtol=1e-8, atol=1e-8)

        # P_{-3}(z) = P_2(z) = (3z^2 - 1)/2
        n = torch.tensor([-3.0], dtype=torch.float64)
        result = torchscience.special_functions.legendre_polynomial_p(n, z)
        expected = (3 * z**2 - 1) / 2
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_negative_n_non_integer(self):
        """Test with negative non-integer n values."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        n = torch.tensor([-0.5], dtype=torch.float64)

        result = torchscience.special_functions.legendre_polynomial_p(n, z)

        # Compare to scipy
        expected = torch.tensor(
            [scipy_legendre_p(-0.5, zi) for zi in z.tolist()],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_negative_n_symmetry(self):
        """Test the symmetry relation P_{-n-1}(z) = P_n(z)."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_pos = torch.tensor([float(n_val)], dtype=torch.float64)
            n_neg = torch.tensor([float(-n_val - 1)], dtype=torch.float64)

            result_pos = torchscience.special_functions.legendre_polynomial_p(
                n_pos, z
            )
            result_neg = torchscience.special_functions.legendre_polynomial_p(
                n_neg, z
            )

            torch.testing.assert_close(
                result_neg,
                result_pos,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Symmetry P_{-n_val - 1}(z) = P_{n_val}(z) failed",
            )
