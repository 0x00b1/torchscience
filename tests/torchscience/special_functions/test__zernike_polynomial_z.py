import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


def reference_zernike_z(n: int, m: int, rho: float, theta: float) -> float:
    """Reference implementation using explicit radial formula and angular terms."""
    from math import cos, factorial, sin

    n = int(n)
    m_val = int(m)
    abs_m = abs(m_val)

    # Check constraints
    if n < 0 or n < abs_m or (n - abs_m) % 2 != 0:
        return 0.0

    # Compute radial part R_n^|m|(rho) using explicit sum formula
    k = (n - abs_m) // 2
    radial = 0.0
    for s in range(k + 1):
        sign = (-1) ** s
        numerator = factorial(n - s)
        denominator = (
            factorial(s)
            * factorial((n + abs_m) // 2 - s)
            * factorial((n - abs_m) // 2 - s)
        )
        coeff = sign * numerator / denominator
        radial += coeff * (rho ** (n - 2 * s))

    # Combine with angular part
    if m_val >= 0:
        return radial * cos(m_val * theta)
    else:
        return radial * sin(abs_m * theta)


class TestZernikePolynomialZ:
    """Tests for full Zernike polynomial Z_n^m(rho, theta)."""

    def test_forward_z00_piston(self):
        """Test that Z_0^0(rho, theta) = 1 for all rho, theta."""
        n = torch.tensor([0.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        theta = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_z11_x_tilt(self):
        """Test that Z_1^1(rho, theta) = rho * cos(theta)."""
        n = torch.tensor([1.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        # Test at theta = 0
        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        expected = rho.unsqueeze(1) * torch.cos(m * theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z1_minus1_y_tilt(self):
        """Test that Z_1^{-1}(rho, theta) = rho * sin(theta)."""
        n = torch.tensor([1.0], dtype=torch.float64)
        m = torch.tensor([-1.0], dtype=torch.float64)
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        # For m < 0, Z = R * sin(|m| * theta)
        abs_m = torch.abs(m)
        expected = rho.unsqueeze(1) * torch.sin(abs_m * theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z20_defocus(self):
        """Test that Z_2^0(rho, theta) = 2*rho^2 - 1 (independent of theta)."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4, math.pi], dtype=torch.float64)

        # Z_2^0 should be the same for all theta
        for t_val in theta:
            t = t_val.unsqueeze(0)
            result = torchscience.special_functions.zernike_polynomial_z(
                n, m, rho, t
            )
            expected = 2 * rho**2 - 1
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z22_astigmatism_0(self):
        """Test that Z_2^2(rho, theta) = rho^2 * cos(2*theta)."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        # R_2^2(rho) = rho^2
        expected = (rho**2).unsqueeze(1) * torch.cos(2 * theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z2_minus2_astigmatism_45(self):
        """Test that Z_2^{-2}(rho, theta) = rho^2 * sin(2*theta)."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([-2.0], dtype=torch.float64)
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        # R_2^2(rho) = rho^2, for m < 0 use sin
        expected = (rho**2).unsqueeze(1) * torch.sin(2 * theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z31_x_coma(self):
        """Test that Z_3^1(rho, theta) = (3*rho^3 - 2*rho) * cos(theta)."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 3], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        # R_3^1(rho) = 3*rho^3 - 2*rho
        radial = 3 * rho**3 - 2 * rho
        expected = radial.unsqueeze(1) * torch.cos(theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_z3_minus1_y_coma(self):
        """Test that Z_3^{-1}(rho, theta) = (3*rho^3 - 2*rho) * sin(theta)."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([-1.0], dtype=torch.float64)
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([math.pi / 6, math.pi / 2], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho.unsqueeze(1), theta.unsqueeze(0)
        )

        # R_3^1(rho) = 3*rho^3 - 2*rho
        radial = 3 * rho**3 - 2 * rho
        expected = radial.unsqueeze(1) * torch.sin(theta.unsqueeze(0))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_reference_agreement(self):
        """Test agreement with explicit reference implementation."""
        rho_vals = torch.linspace(0.1, 1.0, 5, dtype=torch.float64)
        theta_vals = torch.linspace(0.0, 2 * math.pi, 5, dtype=torch.float64)

        # Test various (n, m) pairs including negative m
        test_cases = [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 2),
            (2, -2),
            (3, 1),
            (3, -1),
            (3, 3),
            (3, -3),
            (4, 0),
            (4, 2),
            (4, -2),
        ]

        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)

            for rho_t in rho_vals:
                for theta_t in theta_vals:
                    rho = rho_t.unsqueeze(0)
                    theta = theta_t.unsqueeze(0)

                    result = (
                        torchscience.special_functions.zernike_polynomial_z(
                            n, m, rho, theta
                        )
                    )

                    expected_val = reference_zernike_z(
                        n_val, m_val, rho_t.item(), theta_t.item()
                    )
                    expected = torch.tensor(
                        [expected_val], dtype=torch.float64
                    )

                    torch.testing.assert_close(
                        result,
                        expected,
                        rtol=1e-7,
                        atol=1e-7,
                        msg=f"Mismatch for n={n_val}, m={m_val}, "
                        f"rho={rho_t.item():.3f}, theta={theta_t.item():.3f}",
                    )

    def test_value_at_rho_0_m_nonzero(self):
        """Test Z_n^m(0, theta) = 0 for |m| > 0."""
        rho = torch.tensor([0.0], dtype=torch.float64)
        theta = torch.tensor([math.pi / 4], dtype=torch.float64)

        # Test various (n, m) pairs with m != 0
        test_cases = [(1, 1), (1, -1), (2, 2), (2, -2), (3, 1), (3, -1)]

        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)
            result = torchscience.special_functions.zernike_polynomial_z(
                n, m, rho, theta
            )
            expected = torch.tensor([0.0], dtype=torch.float64)

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Z_{n_val}^{m_val}(0, theta) should be 0",
            )

    def test_invalid_m_greater_than_n(self):
        """Test that Z_n^m = 0 when |m| > n."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([3.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64)
        theta = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_invalid_n_minus_m_odd(self):
        """Test that Z_n^m = 0 when (n-|m|) is odd."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)  # n-|m| = 3 is odd
        rho = torch.tensor([0.5], dtype=torch.float64)
        theta = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_gradcheck_rho_theta(self):
        """Test gradient correctness with respect to rho and theta."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(rho, theta):
            return torchscience.special_functions.zernike_polynomial_z(
                n, m, rho, theta
            )

        assert torch.autograd.gradcheck(
            func, (rho, theta), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z31(self):
        """Test gradient correctness for Z_3^1 (x-coma)."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(rho, theta):
            return torchscience.special_functions.zernike_polynomial_z(
                n, m, rho, theta
            )

        assert torch.autograd.gradcheck(
            func, (rho, theta), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        n = torch.empty(3, 1, device="meta", dtype=torch.float64)
        m = torch.empty(1, device="meta", dtype=torch.float64)
        rho = torch.empty(1, 5, device="meta", dtype=torch.float64)
        theta = torch.empty(1, device="meta", dtype=torch.float64)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )

        assert result.device.type == "meta"
        assert result.shape == (3, 5)

    def test_broadcasting(self):
        """Test broadcasting between n, m, rho, and theta."""
        n = torch.tensor([[0.0], [2.0]], dtype=torch.float64)  # (2, 1)
        m = torch.tensor([0.0], dtype=torch.float64)  # (1,)
        rho = torch.tensor([0.5], dtype=torch.float64)  # (1,)
        theta = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)  # (2,)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        assert result.shape == (2, 2)

        # Verify Z_0^0 = 1
        torch.testing.assert_close(
            result[0, :],
            torch.ones(2, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # Verify Z_2^0 = 2*rho^2 - 1 = -0.5 (independent of theta)
        expected_z20 = 2 * 0.5**2 - 1
        torch.testing.assert_close(
            result[1, :],
            torch.tensor([expected_z20, expected_z20], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([2.0], dtype=torch.float32)
        m = torch.tensor([2.0], dtype=torch.float32)
        rho = torch.tensor([0.5], dtype=torch.float32)
        theta = torch.tensor([0.0], dtype=torch.float32)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        assert result.dtype == torch.float32

        # Z_2^2(rho, theta) = rho^2 * cos(2*theta) = 0.25 * 1 = 0.25
        expected = torch.tensor([0.25], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtypes(self, dtype):
        """Test with low-precision dtypes."""
        n = torch.tensor([2.0], dtype=dtype)
        m = torch.tensor([0.0], dtype=dtype)
        rho = torch.tensor([0.5], dtype=dtype)
        theta = torch.tensor([0.0], dtype=dtype)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        assert result.dtype == dtype

        # Compare to float32 reference
        n_f32 = n.to(torch.float32)
        m_f32 = m.to(torch.float32)
        rho_f32 = rho.to(torch.float32)
        theta_f32 = theta.to(torch.float32)
        expected = torchscience.special_functions.zernike_polynomial_z(
            n_f32, m_f32, rho_f32, theta_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=rtol
        )

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""

        @torch.compile
        def compiled_zernike(n, m, rho, theta):
            return torchscience.special_functions.zernike_polynomial_z(
                n, m, rho, theta
            )

        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        theta = torch.tensor([0.0], dtype=torch.float64)

        result = compiled_zernike(n, m, rho, theta)
        # Z_2^2(1, 0) = 1 * cos(0) = 1
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize(
        "rho_dtype,theta_dtype,expected_dtype",
        [
            (torch.complex64, torch.float32, torch.complex64),
            (torch.complex128, torch.float64, torch.complex128),
            (torch.complex128, torch.complex128, torch.complex128),
        ],
    )
    def test_complex_dtypes(self, rho_dtype, theta_dtype, expected_dtype):
        """Test with complex dtypes."""
        n = torch.tensor([2.0], dtype=torch.float32)
        m = torch.tensor([0.0], dtype=torch.float32)
        # Create real tensors first then convert to complex if needed
        if rho_dtype in (torch.complex64, torch.complex128):
            rho = torch.tensor(
                [0.5],
                dtype=torch.float32
                if rho_dtype == torch.complex64
                else torch.float64,
            ).to(rho_dtype)
        else:
            rho = torch.tensor([0.5], dtype=rho_dtype)
        if theta_dtype in (torch.complex64, torch.complex128):
            theta = torch.tensor(
                [0.0],
                dtype=torch.float32
                if theta_dtype == torch.complex64
                else torch.float64,
            ).to(theta_dtype)
        else:
            theta = torch.tensor([0.0], dtype=theta_dtype)

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        assert result.dtype == expected_dtype

    def test_complex_real_axis(self):
        """Test complex dtype with purely real values matches real dtype."""
        n = torch.tensor([2.0], dtype=torch.float64)
        m = torch.tensor([2.0], dtype=torch.float64)
        rho_real = torch.tensor([0.5, 0.75], dtype=torch.float64)
        theta_real = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)
        rho_complex = rho_real.to(torch.complex128)
        theta_complex = theta_real.to(torch.complex128)

        result_real = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho_real.unsqueeze(1), theta_real.unsqueeze(0)
        )
        result_complex = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho_complex.unsqueeze(1), theta_complex.unsqueeze(0)
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
        m = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        rho = torch.tensor([0.5, 0.75], dtype=torch.float64, device="cuda")
        theta = torch.tensor(
            [0.0, math.pi / 4], dtype=torch.float64, device="cuda"
        )

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        assert result.device.type == "cuda"

        # Compare to CPU
        result_cpu = torchscience.special_functions.zernike_polynomial_z(
            n.cpu(), m.cpu(), rho.cpu(), theta.cpu()
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
        m = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        rho = torch.tensor(
            [0.5, 0.75],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        theta = torch.tensor(
            [0.0, 0.5],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )

        result = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        result.sum().backward()

        # Compare gradient to CPU
        rho_cpu = torch.tensor(
            [0.5, 0.75], dtype=torch.float64, requires_grad=True
        )
        theta_cpu = torch.tensor(
            [0.0, 0.5], dtype=torch.float64, requires_grad=True
        )
        result_cpu = torchscience.special_functions.zernike_polynomial_z(
            n.cpu(), m.cpu(), rho_cpu, theta_cpu
        )
        result_cpu.sum().backward()

        torch.testing.assert_close(
            rho.grad.cpu(), rho_cpu.grad, rtol=1e-8, atol=1e-8
        )
        torch.testing.assert_close(
            theta.grad.cpu(), theta_cpu.grad, rtol=1e-8, atol=1e-8
        )

    def test_angular_periodicity(self):
        """Test that Z_n^m is 2*pi periodic in theta."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([0.75], dtype=torch.float64)
        theta = torch.tensor([0.5], dtype=torch.float64)
        theta_plus_2pi = theta + 2 * math.pi

        result1 = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta
        )
        result2 = torchscience.special_functions.zernike_polynomial_z(
            n, m, rho, theta_plus_2pi
        )

        torch.testing.assert_close(result1, result2, rtol=1e-10, atol=1e-10)
