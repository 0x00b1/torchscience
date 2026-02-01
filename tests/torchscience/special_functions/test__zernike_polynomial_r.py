import pytest
import torch
import torch.testing

import torchscience.special_functions


def reference_zernike_r(n: int, m: int, rho: float) -> float:
    """Reference implementation using explicit sum formula."""
    n = int(n)
    m = int(abs(m))

    # Check constraints
    if n < 0 or m < 0 or n < m or (n - m) % 2 != 0:
        return 0.0

    # Use explicit sum formula
    k = (n - m) // 2
    result = 0.0
    for s in range(k + 1):
        sign = (-1) ** s
        # Compute (n-s)! / (s! * ((n+m)/2-s)! * ((n-m)/2-s)!)
        from math import factorial

        numerator = factorial(n - s)
        denominator = (
            factorial(s)
            * factorial((n + m) // 2 - s)
            * factorial((n - m) // 2 - s)
        )
        coeff = sign * numerator / denominator
        result += coeff * (rho ** (n - 2 * s))

    return result


class TestZernikePolynomialR:
    """Tests for radial Zernike polynomial R_n^m(rho)."""

    def test_forward_r00_equals_1(self):
        """Test that R_0^0(rho) = 1 for all rho."""
        n = torch.tensor([0.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r11_equals_rho(self):
        """Test that R_1^1(rho) = rho."""
        n = torch.tensor([1.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        torch.testing.assert_close(result, rho, rtol=1e-10, atol=1e-10)

    def test_forward_r20_formula(self):
        """Test R_2^0(rho) = 2*rho^2 - 1."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = 2 * rho**2 - 1
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r22_formula(self):
        """Test R_2^2(rho) = rho^2."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = rho**2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r31_formula(self):
        """Test R_3^1(rho) = 3*rho^3 - 2*rho."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = 3 * rho**3 - 2 * rho
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r33_formula(self):
        """Test R_3^3(rho) = rho^3."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([3.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = rho**3
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r40_formula(self):
        """Test R_4^0(rho) = 6*rho^4 - 6*rho^2 + 1."""
        n = torch.tensor([4.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = 6 * rho**4 - 6 * rho**2 + 1
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r42_formula(self):
        """Test R_4^2(rho) = 4*rho^4 - 3*rho^2."""
        n = torch.tensor([4.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = 4 * rho**4 - 3 * rho**2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_r44_formula(self):
        """Test R_4^4(rho) = rho^4."""
        n = torch.tensor([4.0], dtype=torch.float64)
        m = torch.tensor([4.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = rho**4
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_reference_agreement(self):
        """Test agreement with explicit sum reference implementation."""
        rho_vals = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)

        # Test various (n, m) pairs
        test_cases = [
            (0, 0),
            (1, 1),
            (2, 0),
            (2, 2),
            (3, 1),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, 4),
            (5, 1),
            (5, 3),
            (5, 5),
            (6, 0),
            (6, 2),
            (6, 4),
            (6, 6),
        ]

        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)
            result = torchscience.special_functions.zernike_polynomial_r(
                n, m, rho_vals
            )

            expected_list = [
                reference_zernike_r(n_val, m_val, r.item()) for r in rho_vals
            ]
            expected = torch.tensor(expected_list, dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                msg=f"Mismatch for n={n_val}, m={m_val}",
            )

    def test_value_at_rho_1(self):
        """Test R_n^m(1) = 1 for all valid n, m."""
        rho = torch.tensor([1.0], dtype=torch.float64)

        # Test various (n, m) pairs
        test_cases = [
            (0, 0),
            (1, 1),
            (2, 0),
            (2, 2),
            (3, 1),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, 4),
        ]

        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)
            result = torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )
            expected = torch.tensor([1.0], dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"R_{n_val}^{m_val}(1) should be 1",
            )

    def test_value_at_rho_0_m_positive(self):
        """Test R_n^m(0) = 0 for m > 0."""
        rho = torch.tensor([0.0], dtype=torch.float64)

        # Test various (n, m) pairs with m > 0
        test_cases = [(1, 1), (2, 2), (3, 1), (3, 3), (4, 2), (4, 4)]

        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)
            result = torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )
            expected = torch.tensor([0.0], dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"R_{n_val}^{m_val}(0) should be 0",
            )

    def test_value_at_rho_0_m_zero(self):
        """Test R_n^0(0) = (-1)^(n/2) for n even."""
        rho = torch.tensor([0.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)

        # Test n = 0, 2, 4, 6
        for n_val in [0, 2, 4, 6]:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            result = torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )
            expected_val = (-1) ** (n_val // 2)
            expected = torch.tensor([float(expected_val)], dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"R_{n_val}^0(0) should be {expected_val}",
            )

    def test_invalid_m_greater_than_n(self):
        """Test that R_n^m = 0 when m > n."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([3.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_invalid_n_minus_m_odd(self):
        """Test that R_n^m = 0 when (n-m) is odd."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)  # n-m = 3 is odd
        rho = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_invalid_negative_n(self):
        """Test that R_n^m = 0 when n < 0."""
        n = torch.tensor([-1.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_m_treated_as_abs(self):
        """Test that negative m is treated as |m|."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m_pos = torch.tensor([2.0], dtype=torch.float64)
        m_neg = torch.tensor([-2.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64)

        result_pos = torchscience.special_functions.zernike_polynomial_r(
            n, m_pos, rho
        )
        result_neg = torchscience.special_functions.zernike_polynomial_r(
            n, m_neg, rho
        )

        torch.testing.assert_close(
            result_pos, result_neg, rtol=1e-10, atol=1e-10
        )

    def test_gradcheck_rho(self):
        """Test gradient correctness with respect to rho."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def func(rho):
            return torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )

        assert torch.autograd.gradcheck(
            func, (rho,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_rho_r31(self):
        """Test gradient correctness for R_3^1."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def func(rho):
            return torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )

        assert torch.autograd.gradcheck(
            func, (rho,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        m = torch.empty(1, device="meta", dtype=torch.float64)
        rho = torch.empty(1, 5, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)

        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n, m, and rho."""
        n = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float64)  # (3, 1)
        m = torch.tensor([0.0], dtype=torch.float64)  # (1,)
        rho = torch.tensor([[0.5, 1.0]], dtype=torch.float64)  # (1, 2)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        assert result.shape == (3, 2)

        # Verify R_0^0 = 1
        torch.testing.assert_close(
            result[0, :],
            torch.ones(2, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # Verify R_2^0(0.5) = 2*0.25 - 1 = -0.5, R_2^0(1) = 2 - 1 = 1
        expected_r2 = torch.tensor([-0.5, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            result[1, :], expected_r2, rtol=1e-10, atol=1e-10
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        m = torch.tensor([0.0], dtype=torch.float32)
        rho = torch.tensor([0.5], dtype=torch.float32)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        assert result.dtype == torch.float32

        # R_2^0(0.5) = 2*0.25 - 1 = -0.5
        expected = torch.tensor([-0.5], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        m = torch.tensor([0.0], dtype=dtype)
        rho = torch.tensor([0.5], dtype=dtype)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        m_f32 = m.to(torch.float32)
        rho_f32 = rho.to(torch.float32)
        expected = torchscience.special_functions.zernike_polynomial_r(
            n_f32, m_f32, rho_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_gradgradcheck_rho(self):
        """Test second-order gradient correctness with respect to rho."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(rho):
            return torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )

        # Use larger eps and tolerances due to finite difference gradients
        assert torch.autograd.gradgradcheck(
            func, (rho,), eps=1e-4, atol=1e-2, rtol=1e-2
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_zernike(n, m, rho):
            return torchscience.special_functions.zernike_polynomial_r(
                n, m, rho
            )

        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64)

        result = compiled_zernike(n, m, rho)
        expected = torch.tensor([-0.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize(
        "n_dtype,m_dtype,rho_dtype,expected_dtype",
        [
            (torch.float32, torch.float32, torch.complex64, torch.complex64),
            (torch.float64, torch.float64, torch.complex128, torch.complex128),
        ],
    )
    def test_complex_dtypes(self, n_dtype, m_dtype, rho_dtype, expected_dtype):
        """Test with complex dtypes."""
        n = torch.tensor([2.0], dtype=n_dtype)
        m = torch.tensor([0.0], dtype=m_dtype)
        rho = torch.tensor([0.5 + 0.0j], dtype=rho_dtype)

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        assert result.dtype == expected_dtype

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho_real = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        rho_complex = rho_real.to(torch.complex128)

        result_real = torchscience.special_functions.zernike_polynomial_r(
            n, m, rho_real
        )
        result_complex = torchscience.special_functions.zernike_polynomial_r(
            n, m, rho_complex
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        n = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        m = torch.tensor([0.0], dtype=torch.float64, device="cuda")
        rho = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, device="cuda"
        )

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.zernike_polynomial_r(
            n.cpu(), m.cpu(), rho.cpu()
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
        m = torch.tensor([0.0], dtype=torch.float64, device="cuda")
        rho = torch.tensor(
            [0.25, 0.5, 0.75],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )

        result = torchscience.special_functions.zernike_polynomial_r(n, m, rho)
        result.sum().backward()

        # Compare gradient to CPU
        rho_cpu = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.zernike_polynomial_r(
            n.cpu(), m.cpu(), rho_cpu
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            rho.grad.cpu(), rho_cpu.grad, rtol=1e-8, atol=1e-8
        )
