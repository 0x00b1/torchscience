import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


def scipy_laguerre_l(n: float, alpha: float, z: float) -> float:
    """Reference implementation using scipy."""
    return scipy.special.eval_genlaguerre(n, alpha, z)


class TestLaguerrePolynomialL:
    """Tests for generalized Laguerre polynomial L_n^alpha(z)."""

    def test_forward_l0_equals_1(self):
        """Test that L_0^alpha(z) = 1 for all alpha and z."""
        n = torch.tensor([0.0], dtype=torch.float64)
        alpha = torch.tensor([0.0, 1.0, 2.5, -0.5], dtype=torch.float64)
        z = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64)

        for a in alpha:
            for zi in z:
                result = torchscience.special_functions.laguerre_polynomial_l(
                    n, a.unsqueeze(0), zi.unsqueeze(0)
                )
                expected = torch.tensor([1.0], dtype=torch.float64)
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_forward_l1_formula(self):
        """Test L_1^alpha(z) = 1 + alpha - z."""
        n = torch.tensor([1.0], dtype=torch.float64)
        alpha_vals = [0.0, 1.0, 2.0, 0.5]
        z_vals = [0.0, 1.0, 2.0, 0.5]

        for alpha_val in alpha_vals:
            for z_val in z_vals:
                alpha = torch.tensor([alpha_val], dtype=torch.float64)
                z = torch.tensor([z_val], dtype=torch.float64)
                result = torchscience.special_functions.laguerre_polynomial_l(
                    n, alpha, z
                )
                expected = torch.tensor(
                    [1.0 + alpha_val - z_val], dtype=torch.float64
                )
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-10,
                    atol=1e-10,
                    msg=f"L_1^{alpha_val}({z_val}) mismatch",
                )

    def test_forward_l2_formula(self):
        """Test L_2^alpha(z) = [(1+alpha)(2+alpha) - 2(2+alpha)z + z^2]/2."""
        n = torch.tensor([2.0], dtype=torch.float64)

        for alpha_val in [0.0, 1.0, 2.0]:
            for z_val in [0.0, 1.0, 2.0]:
                alpha = torch.tensor([alpha_val], dtype=torch.float64)
                z = torch.tensor([z_val], dtype=torch.float64)
                result = torchscience.special_functions.laguerre_polynomial_l(
                    n, alpha, z
                )
                # L_2^alpha(z) = [(1+alpha)(2+alpha) - 2(2+alpha)z + z^2]/2
                expected_val = (
                    (1 + alpha_val) * (2 + alpha_val)
                    - 2 * (2 + alpha_val) * z_val
                    + z_val**2
                ) / 2
                expected = torch.tensor([expected_val], dtype=torch.float64)
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    msg=f"L_2^{alpha_val}({z_val}) mismatch",
                )

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.eval_genlaguerre."""
        z_vals = torch.linspace(0.1, 5.0, 10, dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            for alpha_val in [0.0, 0.5, 1.0, 2.0]:
                n = torch.tensor([float(n_val)], dtype=torch.float64)
                alpha = torch.tensor([alpha_val], dtype=torch.float64)
                result = torchscience.special_functions.laguerre_polynomial_l(
                    n, alpha, z_vals
                )

                expected_list = [
                    scipy_laguerre_l(n_val, alpha_val, z.item())
                    for z in z_vals
                ]
                expected = torch.tensor(expected_list, dtype=torch.float64)

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    msg=f"Mismatch for n={n_val}, alpha={alpha_val}",
                )

    def test_ordinary_laguerre_alpha_0(self):
        """Test that L_n^0(z) = L_n(z) (ordinary Laguerre polynomial)."""
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        alpha = torch.tensor([0.0], dtype=torch.float64)

        # Compare to scipy for ordinary Laguerre
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

            # scipy.special.eval_laguerre for ordinary Laguerre
            expected_list = [
                scipy.special.eval_laguerre(n_val, zi.item()) for zi in z
            ]
            expected = torch.tensor(expected_list, dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Ordinary Laguerre L_{n_val} mismatch",
            )

    def test_recurrence_relation(self):
        """Test (n+1) L_{n+1}^alpha = (2n + alpha + 1 - z) L_n^alpha - (n + alpha) L_{n-1}^alpha."""
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)

        for n_val in range(1, 8):
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n_val - 1)], dtype=torch.float64)
            n_plus_1 = torch.tensor([float(n_val + 1)], dtype=torch.float64)

            L_n = torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )
            L_n_minus_1 = torchscience.special_functions.laguerre_polynomial_l(
                n_minus_1, alpha, z
            )
            L_n_plus_1 = torchscience.special_functions.laguerre_polynomial_l(
                n_plus_1, alpha, z
            )

            # Recurrence: (n+1) L_{n+1} = (2n + alpha + 1 - z) L_n - (n + alpha) L_{n-1}
            left = (n_val + 1) * L_n_plus_1
            right = (2 * n_val + 1.0 + 1 - z) * L_n - (
                n_val + 1.0
            ) * L_n_minus_1

            torch.testing.assert_close(
                left,
                right,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Recurrence failed at n={n_val}",
            )

    def test_value_at_z_0(self):
        """Test L_n^alpha(0) = C(n+alpha, n) = Gamma(n+alpha+1)/(Gamma(alpha+1)*n!)."""
        z = torch.tensor([0.0], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            for alpha_val in [0.0, 0.5, 1.0, 2.0]:
                n = torch.tensor([float(n_val)], dtype=torch.float64)
                alpha = torch.tensor([alpha_val], dtype=torch.float64)
                result = torchscience.special_functions.laguerre_polynomial_l(
                    n, alpha, z
                )

                # L_n^alpha(0) = binomial(n+alpha, n)
                expected_val = scipy_laguerre_l(n_val, alpha_val, 0.0)
                expected = torch.tensor([expected_val], dtype=torch.float64)

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    msg=f"L_{n_val}^{alpha_val}(0) mismatch",
                )

    def test_gradcheck_z(self):
        """Test gradient correctness with respect to z."""
        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )

        def func(z):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_alpha(self):
        """Test gradient correctness with respect to alpha."""
        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0], dtype=torch.float64)

        def func(alpha):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        assert torch.autograd.gradcheck(
            func, (alpha,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradcheck_n(self):
        """Test gradient correctness with respect to n."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        def func(n):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradcheck_all(self):
        """Test gradient correctness with respect to all parameters."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(n, alpha, z):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        assert torch.autograd.gradcheck(
            func, (n, alpha, z), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        alpha = torch.empty(1, device="meta", dtype=torch.float64)
        z = torch.empty(1, 5, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )

        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n, alpha, and z."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        alpha = torch.tensor([0.0], dtype=torch.float64)  # (1,)
        z = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)  # (1, 3)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        assert result.shape == (3, 3)

        # Verify L_0^0(z) = 1 for all z
        torch.testing.assert_close(
            result[0, :],
            torch.ones(3, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # Verify L_1^0(z) = 1 - z
        expected_L1 = 1.0 - z.squeeze()
        torch.testing.assert_close(
            result[1, :], expected_L1, rtol=1e-10, atol=1e-10
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        alpha = torch.tensor([0.0], dtype=torch.float32)
        z = torch.tensor([1.0], dtype=torch.float32)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        assert result.dtype == torch.float32

        # L_2^0(1) = (1 - 2*1 + 0.5*1^2) = -0.5
        expected = torch.tensor([-0.5], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        alpha = torch.tensor([0.0], dtype=dtype)
        z = torch.tensor([1.0], dtype=dtype)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        alpha_f32 = alpha.to(torch.float32)
        z_f32 = z.to(torch.float32)
        expected = torchscience.special_functions.laguerre_polynomial_l(
            n_f32, alpha_f32, z_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness with respect to z."""
        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(z):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        # Use larger eps and tolerances due to finite difference gradients
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-4, atol=1e-2, rtol=1e-2
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_laguerre(n, alpha, z):
            return torchscience.special_functions.laguerre_polynomial_l(
                n, alpha, z
            )

        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        result = compiled_laguerre(n, alpha, z)
        expected = torch.tensor([-0.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize(
        "n_dtype,alpha_dtype,z_dtype,expected_dtype",
        [
            (torch.float32, torch.float32, torch.complex64, torch.complex64),
            (torch.float64, torch.float64, torch.complex128, torch.complex128),
        ],
    )
    def test_complex_dtypes(
        self, n_dtype, alpha_dtype, z_dtype, expected_dtype
    ):
        """Test with complex dtypes."""
        n = torch.tensor([2.0], dtype=n_dtype)
        alpha = torch.tensor([1.0], dtype=alpha_dtype)
        z = torch.tensor([1.0 + 0.1j], dtype=z_dtype)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        assert result.dtype == expected_dtype

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        z_real = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z_real
        )
        result_complex = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z_complex
        )

        # Real part should match
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-8, atol=1e-8
        )
        # Imaginary part should be approximately zero
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_negative_alpha(self):
        """Test with negative alpha values (alpha > -1 for orthogonality)."""
        n = torch.tensor([2.0], dtype=torch.float64)
        alpha = torch.tensor([-0.5], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )

        # Compare to scipy
        expected = torch.tensor(
            [scipy_laguerre_l(2, -0.5, 1.0)], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        alpha = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.laguerre_polynomial_l(
            n.cpu(), alpha.cpu(), z.cpu()
        )
        torch.testing.assert_close(
            result.cpu(), result_cpu, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward(self):
        """Test backward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        alpha = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        z = torch.tensor(
            [0.5, 1.0, 2.0],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )

        result = torchscience.special_functions.laguerre_polynomial_l(
            n, alpha, z
        )
        result.sum().backward()

        # Compare gradient to CPU
        z_cpu = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.laguerre_polynomial_l(
            n.cpu(), alpha.cpu(), z_cpu
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            z.grad.cpu(), z_cpu.grad, rtol=1e-8, atol=1e-8
        )
