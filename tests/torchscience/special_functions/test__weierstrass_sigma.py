import pytest
import torch
import torch.testing

try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

import torchscience.special_functions


def mpmath_weierstrass_sigma(z: complex, g2: float, g3: float) -> complex:
    """Reference implementation using mpmath.ellipfun('S', ...)."""
    with mpmath.workdps(30):
        mp_z = (
            mpmath.mpc(z.real, z.imag)
            if isinstance(z, complex)
            else mpmath.mpf(z)
        )
        # mpmath uses 'S' for sigma function
        result = mpmath.ellipfun("S", mp_z, g2=g2, g3=g3)
        return complex(result)


class TestWeierstrassSigma:
    """Tests for Weierstrass sigma function sigma(z; g2, g3)."""

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_agreement(self):
        """Test agreement with mpmath reference."""
        # Test various z values with different g2, g3 combinations
        test_cases = [
            # (z_real, z_imag, g2, g3)
            (0.5, 0.0, 1.0, 0.0),
            (0.3, 0.0, 1.0, 0.5),
            (0.4, 0.2, 2.0, 1.0),
            (0.6, 0.3, 0.5, 0.25),
            (0.2, 0.5, 1.5, 0.75),
        ]
        for z_real, z_imag, g2_val, g3_val in test_cases:
            z_val = complex(z_real, z_imag) if z_imag != 0 else z_real
            z = torch.tensor([z_val], dtype=torch.complex128)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            result = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )
            expected = mpmath_weierstrass_sigma(z_val, g2_val, g3_val)

            torch.testing.assert_close(
                result.real,
                torch.tensor([expected.real], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
            )
            if abs(expected.imag) > 1e-10:
                torch.testing.assert_close(
                    result.imag,
                    torch.tensor([expected.imag], dtype=torch.float64),
                    rtol=1e-6,
                    atol=1e-6,
                )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_agreement_real_input(self):
        """Test agreement with mpmath for real z values."""
        test_cases = [
            # (z, g2, g3)
            (0.5, 1.0, 0.0),
            (0.3, 2.0, 0.5),
            (0.7, 1.5, 0.25),
        ]
        for z_val, g2_val, g3_val in test_cases:
            z = torch.tensor([z_val], dtype=torch.float64)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            result = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )
            expected = mpmath_weierstrass_sigma(z_val, g2_val, g3_val)

            torch.testing.assert_close(
                result,
                torch.tensor([expected.real], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_sigma_zero(self):
        """Test that sigma(0) = 0."""
        z = torch.tensor([0.0], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

        torch.testing.assert_close(
            result,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sigma_zero_various_invariants(self):
        """Test that sigma(0) = 0 for various g2, g3."""
        test_cases = [
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.5),
            (0.5, 0.25),
        ]
        z = torch.tensor([0.0], dtype=torch.float64)

        for g2_val, g3_val in test_cases:
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            result = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )

            torch.testing.assert_close(
                result,
                torch.tensor([0.0], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_odd_function(self):
        """Test that sigma(-z) = -sigma(z) (sigma is odd)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        sigma_pos = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        sigma_neg = torchscience.special_functions.weierstrass_sigma(
            -z, g2, g3
        )

        torch.testing.assert_close(
            sigma_pos, -sigma_neg, rtol=1e-10, atol=1e-10
        )

    def test_odd_function_complex(self):
        """Test that sigma(-z) = -sigma(z) for complex z."""
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        sigma_pos = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        sigma_neg = torchscience.special_functions.weierstrass_sigma(
            -z, g2, g3
        )

        torch.testing.assert_close(sigma_pos, -sigma_neg, rtol=1e-8, atol=1e-8)

    def test_odd_function_multiple_points(self):
        """Test odd function property at multiple points."""
        z_vals = [0.2, 0.4, 0.6, 0.8]
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        for z_val in z_vals:
            z = torch.tensor([z_val], dtype=torch.float64)
            sigma_pos = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )
            sigma_neg = torchscience.special_functions.weierstrass_sigma(
                -z, g2, g3
            )
            torch.testing.assert_close(
                sigma_pos, -sigma_neg, rtol=1e-10, atol=1e-10
            )

    def test_gradcheck(self):
        """Test gradients with torch.autograd.gradcheck."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        g3 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.weierstrass_sigma,
            (z, g2, g3),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradcheck_z_only(self):
        """Test gradient w.r.t. z only."""
        z = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=False)
        g3 = torch.tensor([1.0], dtype=torch.float64, requires_grad=False)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.weierstrass_sigma,
            (z, g2, g3),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        g3 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.weierstrass_sigma,
            (z, g2, g3),
            eps=1e-4,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_complex_input(self):
        """Test with complex inputs."""
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.is_complex()

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_input_mpmath(self):
        """Test complex inputs against mpmath."""
        z_val = 0.3 + 0.4j
        g2_val = 1.0
        g3_val = 0.5

        z = torch.tensor([z_val], dtype=torch.complex128)
        g2 = torch.tensor([g2_val], dtype=torch.float64)
        g3 = torch.tensor([g3_val], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        expected = mpmath_weierstrass_sigma(z_val, g2_val, g3_val)

        torch.testing.assert_close(
            result.real,
            torch.tensor([expected.real], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([expected.imag], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        z = torch.tensor([[0.3], [0.4], [0.5]], dtype=torch.float64)
        g2 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.shape == (3, 2)

    def test_broadcasting_all_args(self):
        """Test broadcasting across all arguments."""
        z = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.shape == (3,)

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        z = torch.tensor([0.5], dtype=torch.float64, device="meta")
        g2 = torch.tensor([1.0], dtype=torch.float64, device="meta")
        g3 = torch.tensor([0.5], dtype=torch.float64, device="meta")

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_meta_tensor_shape_inference(self):
        """Test meta tensor shape inference with broadcasting."""
        z = torch.empty(5, 3, dtype=torch.float64, device="meta")
        g2 = torch.empty(1, dtype=torch.float64, device="meta")
        g3 = torch.empty(1, dtype=torch.float64, device="meta")

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.device.type == "meta"
        assert result.shape == (5, 3)

    def test_taylor_series_near_zero(self):
        """Test behavior near z=0 matches Taylor series.

        sigma(z) = z - (g2/240)*z^5 - (g3/840)*z^7 + O(z^9)
        """
        z_val = 0.1
        g2_val = 1.0
        g3_val = 0.5

        z = torch.tensor([z_val], dtype=torch.float64)
        g2 = torch.tensor([g2_val], dtype=torch.float64)
        g3 = torch.tensor([g3_val], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

        # Taylor approximation: sigma(z) ~ z - (g2/240)*z^5 - (g3/840)*z^7
        taylor_approx = (
            z_val - (g2_val / 240) * z_val**5 - (g3_val / 840) * z_val**7
        )

        # For small z, should be close
        torch.testing.assert_close(
            result,
            torch.tensor([taylor_approx], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_entire_function(self):
        """Test that sigma is entire (no poles) at various points."""
        # Unlike P which has poles, sigma should be finite everywhere
        z_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        for z_val in z_vals:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )
            assert torch.isfinite(result).all(), (
                f"Expected finite at z={z_val}, got {result}"
            )

    def test_float32(self):
        """Test with float32 precision."""
        z = torch.tensor([0.5], dtype=torch.float32)
        g2 = torch.tensor([1.0], dtype=torch.float32)
        g3 = torch.tensor([0.5], dtype=torch.float32)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        z = torch.tensor(
            [0.3, 0.4, 0.5, 0.6], dtype=torch.float64, requires_grad=True
        )
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        result.sum().backward()

        assert torch.isfinite(z.grad).all(), (
            f"Expected finite gradients, got {z.grad}"
        )

    def test_batch_computation(self):
        """Test batch computation with multiple z values."""
        z = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert result.shape == (5,)
        assert torch.isfinite(result).all()

    def test_negative_invariants(self):
        """Test with negative g2 and g3 values."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([-1.0], dtype=torch.float64)
        g3 = torch.tensor([-0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_equianharmonic_case(self):
        """Test the equianharmonic case where g2=0."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([0.0], dtype=torch.float64)
        g3 = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_lemniscatic_case(self):
        """Test the lemniscatic case where g3=0."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        assert torch.isfinite(result).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        g2 = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        g3 = torch.tensor([0.5], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

        # Compare with CPU result
        z_cpu = torch.tensor([0.5], dtype=torch.float64)
        g2_cpu = torch.tensor([1.0], dtype=torch.float64)
        g3_cpu = torch.tensor([0.5], dtype=torch.float64)
        expected = torchscience.special_functions.weierstrass_sigma(
            z_cpu, g2_cpu, g3_cpu
        )

        torch.testing.assert_close(
            result.cpu(), expected, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_gradient(self):
        """Test gradients on CUDA."""
        z = torch.tensor(
            [0.5], dtype=torch.float64, device="cuda", requires_grad=True
        )
        g2 = torch.tensor(
            [1.0], dtype=torch.float64, device="cuda", requires_grad=True
        )
        g3 = torch.tensor(
            [0.5], dtype=torch.float64, device="cuda", requires_grad=True
        )

        result = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        result.backward()

        assert torch.isfinite(z.grad).all()
        assert torch.isfinite(g2.grad).all()
        assert torch.isfinite(g3.grad).all()

    def test_homogeneity(self):
        """Test the homogeneity property.

        sigma(tz; t^-4 g2, t^-6 g3) = t * sigma(z; g2, g3)
        """
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)
        t = 2.0

        # sigma(z; g2, g3)
        sigma_original = torchscience.special_functions.weierstrass_sigma(
            z, g2, g3
        )

        # sigma(tz; t^-4 g2, t^-6 g3)
        z_scaled = t * z
        g2_scaled = (t ** (-4)) * g2
        g3_scaled = (t ** (-6)) * g3
        sigma_scaled = torchscience.special_functions.weierstrass_sigma(
            z_scaled, g2_scaled, g3_scaled
        )

        # Should equal t * sigma(z; g2, g3)
        expected = t * sigma_original

        torch.testing.assert_close(
            sigma_scaled, expected, rtol=1e-6, atol=1e-6
        )

    def test_autocast_cuda_float16(self):
        """Test autocast with CUDA float16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        z = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        g2 = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        g3 = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )

        # Result should be computed (autocast may or may not change dtype)
        assert result.device.type == "cuda"
