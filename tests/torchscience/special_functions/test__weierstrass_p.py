import pytest
import torch
import torch.testing

try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

import torchscience.special_functions


def mpmath_weierstrass_p(z: complex, g2: float, g3: float) -> complex:
    """Reference implementation using mpmath.ellipfun."""
    with mpmath.workdps(30):
        mp_z = (
            mpmath.mpc(z.real, z.imag)
            if isinstance(z, complex)
            else mpmath.mpf(z)
        )
        result = mpmath.ellipfun("P", mp_z, g2=g2, g3=g3)
        return complex(result)


def mpmath_weierstrass_p_prime(z: complex, g2: float, g3: float) -> complex:
    """Reference for P'(z) using mpmath derivative."""
    with mpmath.workdps(30):
        mp_z = (
            mpmath.mpc(z.real, z.imag)
            if isinstance(z, complex)
            else mpmath.mpf(z)
        )
        # P'(z) can be computed via the relation P'^2 = 4P^3 - g2*P - g3
        # or using mpmath.ellipfun with the derivative flag
        p_val = mpmath.ellipfun("P", mp_z, g2=g2, g3=g3)
        # P'^2 = 4P^3 - g2*P - g3, so P' = sqrt(4P^3 - g2*P - g3)
        # Sign depends on which half-period we're in
        discriminant = 4 * p_val**3 - g2 * p_val - g3
        return complex(mpmath.sqrt(discriminant))


class TestWeierstrassP:
    """Tests for Weierstrass elliptic function P(z; g2, g3)."""

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

            result = torchscience.special_functions.weierstrass_p(z, g2, g3)
            expected = mpmath_weierstrass_p(z_val, g2_val, g3_val)

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
        # Test with real z (non-complex tensor)
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

            result = torchscience.special_functions.weierstrass_p(z, g2, g3)
            expected = mpmath_weierstrass_p(z_val, g2_val, g3_val)

            torch.testing.assert_close(
                result,
                torch.tensor([expected.real], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_pole_at_zero(self):
        """Test that P(0) returns infinity (double pole at origin)."""
        z = torch.tensor([0.0], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        # P(z) has a double pole at z=0, so should return inf
        assert torch.isinf(result).all() or torch.isnan(result).all(), (
            f"Expected inf or nan at z=0, got {result}"
        )

    def test_pole_at_lattice_points(self):
        """Test behavior near lattice points (poles)."""
        # For a generic lattice, P(z) has poles at lattice points
        # Near z=0, P(z) ~ 1/z^2
        z_small = torch.tensor([1e-10], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.weierstrass_p(z_small, g2, g3)
        # Should be very large, approximately 1/z^2 = 1e20
        assert result.item() > 1e10, (
            f"Expected large value near pole, got {result}"
        )

    def test_differential_equation(self):
        """Test that P'^2 = 4P^3 - g2*P - g3.

        This is the fundamental differential equation satisfied by P.
        We verify it using autograd to compute P'.
        """
        # Use values away from poles
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        # Compute P(z)
        p_val = torchscience.special_functions.weierstrass_p(z, g2, g3)

        # Compute P'(z) via autograd
        (p_prime,) = torch.autograd.grad(p_val, z, create_graph=True)

        # Verify P'^2 = 4P^3 - g2*P - g3
        lhs = p_prime**2
        rhs = 4 * p_val**3 - g2 * p_val - g3

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_differential_equation_multiple_points(self):
        """Test differential equation at multiple points."""
        test_cases = [
            (0.3, 1.0, 0.0),
            (0.4, 2.0, 0.5),
            (0.6, 0.5, 0.25),
        ]
        for z_val, g2_val, g3_val in test_cases:
            z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            p_val = torchscience.special_functions.weierstrass_p(z, g2, g3)
            (p_prime,) = torch.autograd.grad(p_val, z, create_graph=True)

            lhs = p_prime**2
            rhs = 4 * p_val**3 - g2 * p_val - g3

            torch.testing.assert_close(
                lhs,
                rhs,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_even_function(self):
        """Test that P(-z) = P(z) (P is an even function)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        p_pos = torchscience.special_functions.weierstrass_p(z, g2, g3)
        p_neg = torchscience.special_functions.weierstrass_p(-z, g2, g3)

        torch.testing.assert_close(p_pos, p_neg, rtol=1e-10, atol=1e-10)

    def test_even_function_complex(self):
        """Test that P(-z) = P(z) for complex z."""
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        p_pos = torchscience.special_functions.weierstrass_p(z, g2, g3)
        p_neg = torchscience.special_functions.weierstrass_p(-z, g2, g3)

        torch.testing.assert_close(p_pos, p_neg, rtol=1e-8, atol=1e-8)

    def test_gradcheck(self):
        """Test gradients with torch.autograd.gradcheck."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        g3 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.weierstrass_p,
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
            torchscience.special_functions.weierstrass_p,
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
            torchscience.special_functions.weierstrass_p,
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

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
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

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        expected = mpmath_weierstrass_p(z_val, g2_val, g3_val)

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

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.shape == (3, 2)

    def test_broadcasting_all_args(self):
        """Test broadcasting across all arguments."""
        z = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.shape == (3,)

    def test_meta_tensor(self):
        """Test that the function works with meta tensors."""
        z = torch.tensor([0.5], dtype=torch.float64, device="meta")
        g2 = torch.tensor([1.0], dtype=torch.float64, device="meta")
        g3 = torch.tensor([0.5], dtype=torch.float64, device="meta")

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.device.type == "meta"
        assert result.shape == (1,)

    def test_meta_tensor_shape_inference(self):
        """Test meta tensor shape inference with broadcasting."""
        z = torch.empty(5, 3, dtype=torch.float64, device="meta")
        g2 = torch.empty(1, dtype=torch.float64, device="meta")
        g3 = torch.empty(1, dtype=torch.float64, device="meta")

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.device.type == "meta"
        assert result.shape == (5, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test that the function works on CUDA."""
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        g2 = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        g3 = torch.tensor([0.5], dtype=torch.float64, device="cuda")

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)

        # Compare with CPU result
        z_cpu = torch.tensor([0.5], dtype=torch.float64)
        g2_cpu = torch.tensor([1.0], dtype=torch.float64)
        g3_cpu = torch.tensor([0.5], dtype=torch.float64)
        expected = torchscience.special_functions.weierstrass_p(
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

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        result.backward()

        assert torch.isfinite(z.grad).all()
        assert torch.isfinite(g2.grad).all()
        assert torch.isfinite(g3.grad).all()

    def test_homogeneity(self):
        """Test the homogeneity property: P(tz; t^-4 g2, t^-6 g3) = t^-2 P(z; g2, g3)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)
        t = 2.0

        # P(z; g2, g3)
        p_original = torchscience.special_functions.weierstrass_p(z, g2, g3)

        # P(tz; t^-4 g2, t^-6 g3)
        z_scaled = t * z
        g2_scaled = (t ** (-4)) * g2
        g3_scaled = (t ** (-6)) * g3
        p_scaled = torchscience.special_functions.weierstrass_p(
            z_scaled, g2_scaled, g3_scaled
        )

        # Should equal t^-2 P(z; g2, g3)
        expected = (t ** (-2)) * p_original

        torch.testing.assert_close(p_scaled, expected, rtol=1e-6, atol=1e-6)

    def test_near_zero_behavior(self):
        """Test Laurent series behavior near z=0: P(z) ~ 1/z^2."""
        # For small z, P(z) ~ 1/z^2 + O(z^2)
        z_vals = [0.01, 0.001, 0.0001]
        g2 = torch.tensor([0.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        for z_val in z_vals:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.weierstrass_p(z, g2, g3)
            expected_approx = 1.0 / (z_val**2)
            # The relative error should decrease as z gets smaller
            rel_error = abs(
                (result.item() - expected_approx) / expected_approx
            )
            assert rel_error < 0.1, (
                f"At z={z_val}, expected ~{expected_approx}, got {result.item()}"
            )

    def test_equianharmonic_case(self):
        """Test the equianharmonic case where g2=0.

        When g2=0 and g3!=0, the lattice is equianharmonic (hexagonal).
        """
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([0.0], dtype=torch.float64)
        g3 = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_lemniscatic_case(self):
        """Test the lemniscatic case where g3=0.

        When g2!=0 and g3=0, the lattice is lemniscatic (square).
        """
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_float32(self):
        """Test with float32 precision."""
        z = torch.tensor([0.5], dtype=torch.float32)
        g2 = torch.tensor([1.0], dtype=torch.float32)
        g3 = torch.tensor([0.5], dtype=torch.float32)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_gradient_finite_away_from_poles(self):
        """Test that gradients are finite away from poles."""
        z = torch.tensor(
            [0.3, 0.4, 0.5, 0.6], dtype=torch.float64, requires_grad=True
        )
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        result.sum().backward()

        assert torch.isfinite(z.grad).all(), (
            f"Expected finite gradients, got {z.grad}"
        )

    def test_batch_computation(self):
        """Test batch computation with multiple z values."""
        z = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert result.shape == (5,)
        assert torch.isfinite(result).all()

    def test_negative_invariants(self):
        """Test with negative g2 and g3 values."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([-1.0], dtype=torch.float64)
        g3 = torch.tensor([-0.5], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_large_invariants(self):
        """Test with large g2 and g3 values."""
        z = torch.tensor([0.5], dtype=torch.float64)
        g2 = torch.tensor([100.0], dtype=torch.float64)
        g3 = torch.tensor([50.0], dtype=torch.float64)

        result = torchscience.special_functions.weierstrass_p(z, g2, g3)
        assert torch.isfinite(result).all()

    def test_autocast_cuda_float16(self):
        """Test autocast with CUDA float16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        z = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        g2 = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        g3 = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = torchscience.special_functions.weierstrass_p(z, g2, g3)

        # Result should be computed (autocast may or may not change dtype)
        assert result.device.type == "cuda"
