import pytest
import torch
import torch.testing

import torchscience.special_functions


def reference_charlier_c(n: int, x: float, a: float) -> float:
    """Reference implementation using the recurrence relation.

    C_0(x; a) = 1
    C_1(x; a) = x/a - 1
    a * C_{n+1}(x; a) = (x - n - a) * C_n(x; a) - n * C_{n-1}(x; a)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x / a - 1.0

    C_prev = 1.0  # C_0
    C_curr = x / a - 1.0  # C_1

    for k in range(1, n):
        C_next = ((x - k - a) * C_curr - k * C_prev) / a
        C_prev = C_curr
        C_curr = C_next

    return C_curr


class TestCharlierPolynomialC:
    """Tests for Charlier polynomial C_n(x; a)."""

    def test_forward_c0_equals_1(self):
        """Test that C_0(x; a) = 1 for all x and a."""
        n = torch.tensor([0.0], dtype=torch.float64)
        x_vals = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        a_vals = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)

        for a in a_vals:
            for xi in x_vals:
                result = torchscience.special_functions.charlier_polynomial_c(
                    n, xi.unsqueeze(0), a.unsqueeze(0)
                )
                expected = torch.tensor([1.0], dtype=torch.float64)
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_forward_c1_formula(self):
        """Test C_1(x; a) = x/a - 1."""
        n = torch.tensor([1.0], dtype=torch.float64)
        x_vals = [0.0, 1.0, 2.0, 3.0, 5.0]
        a_vals = [0.5, 1.0, 2.0, 3.0]

        for a_val in a_vals:
            for x_val in x_vals:
                a = torch.tensor([a_val], dtype=torch.float64)
                x = torch.tensor([x_val], dtype=torch.float64)
                result = torchscience.special_functions.charlier_polynomial_c(
                    n, x, a
                )
                expected = torch.tensor(
                    [x_val / a_val - 1.0], dtype=torch.float64
                )
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-10,
                    atol=1e-10,
                    msg=f"C_1({x_val}; {a_val}) mismatch",
                )

    def test_forward_c2_formula(self):
        """Test C_2(x; a) = (x^2 - (2a+1)x + a^2) / a^2."""
        n = torch.tensor([2.0], dtype=torch.float64)

        for a_val in [0.5, 1.0, 2.0, 3.0]:
            for x_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
                a = torch.tensor([a_val], dtype=torch.float64)
                x = torch.tensor([x_val], dtype=torch.float64)
                result = torchscience.special_functions.charlier_polynomial_c(
                    n, x, a
                )
                # C_2(x; a) = (x^2 - (2a+1)x + a^2) / a^2
                expected_val = (
                    x_val**2 - (2 * a_val + 1) * x_val + a_val**2
                ) / (a_val**2)
                expected = torch.tensor([expected_val], dtype=torch.float64)
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    msg=f"C_2({x_val}; {a_val}) mismatch",
                )

    def test_reference_agreement(self):
        """Test agreement with reference implementation."""
        x_vals = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )

        for n_val in [0, 1, 2, 3, 4, 5]:
            for a_val in [0.5, 1.0, 2.0, 3.0]:
                n = torch.tensor([float(n_val)], dtype=torch.float64)
                a = torch.tensor([a_val], dtype=torch.float64)
                result = torchscience.special_functions.charlier_polynomial_c(
                    n, x_vals, a
                )

                expected_list = [
                    reference_charlier_c(n_val, xi.item(), a_val)
                    for xi in x_vals
                ]
                expected = torch.tensor(expected_list, dtype=torch.float64)

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    msg=f"Mismatch for n={n_val}, a={a_val}",
                )

    def test_recurrence_relation(self):
        """Test a * C_{n+1} = (x - n - a) * C_n - n * C_{n-1}."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)

        for n_val in range(1, 8):
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            n_minus_1 = torch.tensor([float(n_val - 1)], dtype=torch.float64)
            n_plus_1 = torch.tensor([float(n_val + 1)], dtype=torch.float64)

            C_n = torchscience.special_functions.charlier_polynomial_c(n, x, a)
            C_n_minus_1 = torchscience.special_functions.charlier_polynomial_c(
                n_minus_1, x, a
            )
            C_n_plus_1 = torchscience.special_functions.charlier_polynomial_c(
                n_plus_1, x, a
            )

            # Recurrence: a * C_{n+1} = (x - n - a) * C_n - n * C_{n-1}
            left = 2.0 * C_n_plus_1
            right = (x - float(n_val) - 2.0) * C_n - float(n_val) * C_n_minus_1

            torch.testing.assert_close(
                left,
                right,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Recurrence failed at n={n_val}",
            )

    def test_value_at_x_0(self):
        """Test C_n(0; a) values."""
        x = torch.tensor([0.0], dtype=torch.float64)

        for n_val in [0, 1, 2, 3, 4, 5]:
            for a_val in [0.5, 1.0, 2.0, 3.0]:
                n = torch.tensor([float(n_val)], dtype=torch.float64)
                a = torch.tensor([a_val], dtype=torch.float64)
                result = torchscience.special_functions.charlier_polynomial_c(
                    n, x, a
                )

                expected_val = reference_charlier_c(n_val, 0.0, a_val)
                expected = torch.tensor([expected_val], dtype=torch.float64)

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    msg=f"C_{n_val}(0; {a_val}) mismatch",
                )

    def test_gradcheck_x(self):
        """Test gradient correctness with respect to x."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([1.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_a(self):
        """Test gradient correctness with respect to a."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)

        def func(a):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        assert torch.autograd.gradcheck(
            func, (a,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradcheck_n(self):
        """Test gradient correctness with respect to n."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)

        def func(n):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradcheck_all(self):
        """Test gradient correctness with respect to all parameters."""
        n = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)

        def func(n, x, a):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        assert torch.autograd.gradcheck(
            func, (n, x, a), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        x = torch.empty(1, 5, device="meta", dtype=torch.float64)
        a = torch.empty(1, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)

        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n, x, and a."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        x = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)  # (1, 3)
        a = torch.tensor([1.0], dtype=torch.float64)  # (1,)

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        assert result.shape == (3, 3)

        # Verify C_0(x; a) = 1 for all x
        torch.testing.assert_close(
            result[0, :],
            torch.ones(3, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # Verify C_1(x; a) = x/a - 1 = x - 1 for a = 1
        expected_C1 = x.squeeze() - 1.0
        torch.testing.assert_close(
            result[1, :], expected_C1, rtol=1e-10, atol=1e-10
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        x = torch.tensor([2.0], dtype=torch.float32)
        a = torch.tensor([1.0], dtype=torch.float32)

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        assert result.dtype == torch.float32

        # C_2(2; 1) = (4 - 3*2 + 1) / 1 = -1
        expected = torch.tensor([-1.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        x = torch.tensor([2.0], dtype=dtype)
        a = torch.tensor([1.0], dtype=dtype)

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        x_f32 = x.to(torch.float32)
        a_f32 = a.to(torch.float32)
        expected = torchscience.special_functions.charlier_polynomial_c(
            n_f32, x_f32, a_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_gradgradcheck_x(self):
        """Test second-order gradient correctness with respect to x."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        # Use larger eps and tolerances due to finite difference gradients
        assert torch.autograd.gradgradcheck(
            func, (x,), eps=1e-4, atol=1e-2, rtol=1e-2
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_charlier(n, x, a):
            return torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

        n = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([2.0], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)

        result = compiled_charlier(n, x, a)
        expected = torch.tensor([-1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize(
        "n_dtype,x_dtype,a_dtype,expected_dtype",
        [
            (torch.float32, torch.complex64, torch.float32, torch.complex64),
            (torch.float64, torch.complex128, torch.float64, torch.complex128),
        ],
    )
    def test_complex_dtypes(self, n_dtype, x_dtype, a_dtype, expected_dtype):
        """Test with complex dtypes."""
        n = torch.tensor([2.0], dtype=n_dtype)
        x = torch.tensor([1.0 + 0.1j], dtype=x_dtype)
        a = torch.tensor([1.0], dtype=a_dtype)

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        assert result.dtype == expected_dtype

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        x_real = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        a = torch.tensor([1.0], dtype=torch.float64)

        result_real = torchscience.special_functions.charlier_polynomial_c(
            n, x_real, a
        )
        result_complex = torchscience.special_functions.charlier_polynomial_c(
            n, x_complex, a
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

    def test_large_n(self):
        """Test with larger polynomial degrees."""
        a = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([3.0], dtype=torch.float64)

        for n_val in [10, 15, 20]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.charlier_polynomial_c(
                n, x, a
            )

            expected = torch.tensor(
                [reference_charlier_c(n_val, 3.0, 2.0)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"C_{n_val}(3; 2) mismatch",
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device="cuda")
        a = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.charlier_polynomial_c(
            n.cpu(), x.cpu(), a.cpu()
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
        x = torch.tensor(
            [0.5, 1.0, 2.0],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        a = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.charlier_polynomial_c(n, x, a)
        result.sum().backward()

        # Compare gradient to CPU
        x_cpu = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.charlier_polynomial_c(
            n.cpu(), x_cpu, a.cpu()
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            x.grad.cpu(), x_cpu.grad, rtol=1e-8, atol=1e-8
        )
