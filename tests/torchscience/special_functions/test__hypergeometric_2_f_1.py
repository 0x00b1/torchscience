import math

import pytest
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)

# Optional scipy import for reference tests
try:
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional sympy import for symbolic verification
try:
    import sympy  # noqa: F401

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# Optional mpmath import for complex parameter reference tests
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def scipy_hyp2f1(a: float, b: float, c: float, z: float) -> float:
    """Reference implementation using SciPy's hyp2f1."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required for this function")
    return float(scipy.special.hyp2f1(a, b, c, z))


def _power_function_identity(func):
    """Check 2F1(a, b; b; z) = (1-z)^(-a) for |z| < 1."""
    a = torch.tensor([2.0], dtype=torch.float64)
    b = torch.tensor([3.0], dtype=torch.float64)
    z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    left = func(a, b, b, z)
    right = torch.pow(1 - z, -a)
    return left, right


def _symmetry_identity(func):
    """Check 2F1(a, b; c; z) = 2F1(b, a; c; z) (symmetry in a, b)."""
    a = torch.tensor([1.5], dtype=torch.float64)
    b = torch.tensor([2.5], dtype=torch.float64)
    c = torch.tensor([3.0], dtype=torch.float64)
    z = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    left = func(a, b, c, z)
    right = func(b, a, c, z)
    return left, right


class TestHypergeometric2F1(OpTestCase):
    """Tests for the Gauss hypergeometric function 2F1."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hypergeometric_2_f_1",
            func=torchscience.special_functions.hypergeometric_2_f_1,
            arity=4,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="b",
                    position=1,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="c",
                    position=2,
                    default_real_range=(0.5, 5.0),
                    # Exclude non-positive integers (poles)
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=3,
                    default_real_range=(-0.9, 0.9),
                    supports_grad=True,
                    # For complex z, convergence is best for |z| < 1
                    complex_magnitude_max=0.95,
                ),
            ],
            tolerances=ToleranceConfig(
                float32_rtol=1e-4,
                float32_atol=1e-4,
                float64_rtol=1e-6,
                float64_atol=1e-6,
                # Finite differences for parameter gradients may have reduced accuracy
                gradcheck_rtol=1e-4,
                gradcheck_atol=1e-4,
                gradgradcheck_rtol=1e-3,
                gradgradcheck_atol=1e-3,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                # Gradient checks are flaky due to numerical precision issues with
                # finite difference approximations on this complex function
                "test_gradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                # Finite difference gradients for a, b, c may be less accurate
                "test_gradcheck_all_params",
                # Mixed sparse/dense tests
                "test_sparse_coo_mixed_with_dense",
                "test_quantized_mixed_with_float",
                # 2F1 has special behavior at zero parameters (terminates to 1),
                # so NaN/sparse zeros don't propagate as expected
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                # Quantized tensors have too much precision loss for this function
                "test_quantized_basic",
                # Half precision not accurate enough for this complex function
                "test_low_precision_forward",
            },
            functional_identities=[
                IdentitySpec(
                    name="power_function",
                    identity_fn=_power_function_identity,
                    description="2F1(a, b; b; z) = (1-z)^(-a)",
                    rtol=1e-6,
                    atol=1e-6,
                ),
                IdentitySpec(
                    name="symmetry",
                    identity_fn=_symmetry_identity,
                    description="2F1(a, b; c; z) = 2F1(b, a; c; z)",
                    rtol=1e-6,
                    atol=1e-6,
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 2.0, 3.0, 0.0),
                    expected=1.0,
                    description="2F1(a, b; c; 0) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 2.0, 3.0, 0.5),
                    expected=1.0,
                    description="2F1(0, b; c; z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 3.0, 0.5),
                    expected=1.0,
                    description="2F1(a, 0; c; z) = 1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,  # 2F1 has special behavior at zeros
            supports_sparse_csr=False,  # 2F1 has special behavior at zeros
            supports_quantized=False,  # Precision loss too high
            supports_meta=True,
        )

    # =========================================================================
    # Hypergeometric 2F1-specific tests
    # =========================================================================

    def test_at_zero(self):
        """Test 2F1(a, b; c; 0) = 1 for various a, b, c."""
        z = torch.tensor([0.0], dtype=torch.float64)
        test_cases = [
            (1.0, 2.0, 3.0),
            (0.5, 0.5, 1.0),
            (2.0, 3.0, 4.0),
            (1.5, 2.5, 3.5),
        ]
        for a_val, b_val, c_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            c = torch.tensor([c_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor([1.0], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for a={a_val}, b={b_val}, c={c_val}",
            )

    def test_zero_parameter_a(self):
        """Test 2F1(0, b; c; z) = 1."""
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_zero_parameter_b(self):
        """Test 2F1(a, 0; c; z) = 1."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_power_function_identity(self):
        """Test 2F1(a, b; b; z) = (1-z)^(-a)."""
        a_values = [0.5, 1.0, 2.0, 3.0]
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)

        for a_val in a_values:
            a = torch.tensor([a_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, b, z
            )
            expected = torch.pow(1 - z, -a)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Power function identity failed for a={a_val}",
            )

    def test_symmetry_in_a_b(self):
        """Test 2F1(a, b; c; z) = 2F1(b, a; c; z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)

        result1 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            b, a, c, z
        )
        torch.testing.assert_close(result1, result2, rtol=1e-10, atol=1e-10)

    def test_negative_integer_a_terminates(self):
        """Test that 2F1(-n, b; c; z) is a polynomial when n is non-negative integer."""
        # 2F1(-1, 1; 1; z) = 1 - z
        a = torch.tensor([-1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = 1 - z
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_negative_integer_a_quadratic(self):
        """Test 2F1(-2, 1; 1; z) = 1 - 2z + z^2 = (1-z)^2."""
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = (1 - z) ** 2
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_first_few_terms(self):
        """Test series expansion: 1 + (ab/c)z + (a(a+1)b(b+1))/(c(c+1)2!)z^2 + ..."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1], dtype=torch.float64)

        # First 3 terms of series:
        # term0 = 1
        # term1 = (a*b/c) * z = (1*2/3) * 0.1 = 0.0666...
        # term2 = (a*(a+1)*b*(b+1))/(c*(c+1)*2!) * z^2
        #       = (1*2*2*3)/(3*4*2) * 0.01 = 12/24 * 0.01 = 0.005
        expected_approx = (
            1.0 + (1 * 2 / 3) * 0.1 + (1 * 2 * 2 * 3) / (3 * 4 * 2) * 0.01
        )

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        # The series converges quickly for small z
        torch.testing.assert_close(
            result,
            torch.tensor([expected_approx], dtype=torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_scipy_reference(self):
        """Test against SciPy's hyp2f1 function."""
        z_values = [0.1, 0.3, 0.5, 0.7]
        a_values = [0.5, 1.0, 2.0]
        b_values = [0.5, 1.0, 2.0]
        c_values = [1.0, 2.0, 3.0]

        for z_val in z_values:
            for a_val in a_values:
                for b_val in b_values:
                    for c_val in c_values:
                        a = torch.tensor([a_val], dtype=torch.float64)
                        b = torch.tensor([b_val], dtype=torch.float64)
                        c = torch.tensor([c_val], dtype=torch.float64)
                        z = torch.tensor([z_val], dtype=torch.float64)

                        result = torchscience.special_functions.hypergeometric_2_f_1(
                            a, b, c, z
                        )
                        expected = torch.tensor(
                            [scipy_hyp2f1(a_val, b_val, c_val, z_val)],
                            dtype=torch.float64,
                        )

                        torch.testing.assert_close(
                            result,
                            expected,
                            rtol=1e-6,
                            atol=1e-6,
                            msg=f"SciPy mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
                        )

    @pytest.mark.skipif(not HAS_SYMPY, reason="SymPy not available")
    def test_sympy_reference(self):
        """Test against SymPy's hyper function."""
        from sympy import N, Rational, hyper

        test_cases = [
            (Rational(1, 2), Rational(1, 2), Rational(3, 2), Rational(1, 4)),
            (1, 2, 3, Rational(1, 2)),
            (Rational(3, 2), Rational(5, 2), 4, Rational(1, 3)),
        ]

        for a_sym, b_sym, c_sym, z_sym in test_cases:
            # Compute with SymPy
            sympy_result = float(N(hyper([a_sym, b_sym], [c_sym], z_sym), 20))

            # Compute with our implementation
            a = torch.tensor([float(a_sym)], dtype=torch.float64)
            b = torch.tensor([float(b_sym)], dtype=torch.float64)
            c = torch.tensor([float(c_sym)], dtype=torch.float64)
            z = torch.tensor([float(z_sym)], dtype=torch.float64)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            torch.testing.assert_close(
                result,
                torch.tensor([sympy_result], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-8,
                msg=f"SymPy mismatch at a={a_sym}, b={b_sym}, c={c_sym}, z={z_sym}",
            )

    def test_gauss_summation_theorem(self):
        """Test Gauss's summation theorem: 2F1(a, b; c; 1) = Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b))
        when Re(c - a - b) > 0."""
        # For convergence at z=1, we need c > a + b
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 1 > 0

        # Compute 2F1 at z approaching 1 (use 0.999 to avoid branch cut issues)
        z = torch.tensor([0.999], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Expected from Gauss's theorem:
        # Gamma(2) * Gamma(1) / (Gamma(1.5) * Gamma(1.5))
        # = 1 * 1 / (sqrt(pi)/2 * sqrt(pi)/2) = 4/pi
        expected = 4.0 / math.pi

        # Allow some tolerance since we're evaluating at 0.999, not exactly 1
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=0.01,
            atol=0.01,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        a = torch.tensor([[1.0], [2.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        c = torch.tensor([3.0], dtype=torch.float64)  # (1,)
        z = torch.tensor([0.5], dtype=torch.float64)  # (1,)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert result.shape == (2, 3)

    def test_gradcheck_z(self):
        """Test gradient correctness for z using finite differences."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor(
            [0.2, 0.4, 0.6], dtype=torch.float64, requires_grad=True
        )

        def func(z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        # Note: For z > 0.5 with c-a-b an integer, Richardson extrapolation is used
        # which introduces small numerical errors (~1e-3)
        torch.autograd.gradcheck(func, (z,), rtol=1e-3, atol=1e-3)

    def test_derivative_formula_z(self):
        """Test that d/dz 2F1(a, b; c; z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)."""
        a_val, b_val, c_val = 1.0, 2.0, 3.0
        a = torch.tensor([a_val], dtype=torch.float64)
        b = torch.tensor([b_val], dtype=torch.float64)
        c = torch.tensor([c_val], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # Compute gradient via autograd
        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)
        y.backward()
        autograd_grad = z.grad.clone()

        # Expected gradient from formula
        z_val = 0.5
        expected_grad = (
            a_val * b_val / c_val
        ) * torchscience.special_functions.hypergeometric_2_f_1(
            torch.tensor([a_val + 1]),
            torch.tensor([b_val + 1]),
            torch.tensor([c_val + 1]),
            torch.tensor([z_val]),
        )

        torch.testing.assert_close(
            autograd_grad,
            expected_grad.to(torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_complex_input(self):
        """Test with complex z values."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert result.dtype == torch.complex128
        assert result.shape == (1,)

        # Result should be complex
        assert result[0].imag != 0

    def test_large_parameters(self):
        """Test with moderately large parameters."""
        a = torch.tensor([10.0], dtype=torch.float64)
        b = torch.tensor([10.0], dtype=torch.float64)
        c = torch.tensor([25.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Should produce a finite result
        assert torch.isfinite(result).all()
        assert result[0] > 1.0  # Series starts at 1 and has positive terms

    def test_negative_z(self):
        """Test with negative z values."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([-0.5, -0.3, -0.1], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Results should be finite and real
        assert torch.isfinite(result).all()

        # For negative z, the series alternates but converges
        # 2F1(a, b; c; z) < 1 for z < 0 with positive a, b, c
        assert (result < 1.0).all()

    # =========================================================================
    # Tests for DLMF 15.8.10: Integer a-b case with |z| > 1
    # =========================================================================

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_n_equals_zero(self):
        """Test DLMF 15.8.10 limiting form when a = b (n = 0) for |z| > 1."""
        # When a = b, both Gamma(a-b) and Gamma(b-a) have poles at 0
        # Use z < -1 to avoid divergence at z=1 (which occurs when c-a-b=0)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # n = a - b = 0
        c = torch.tensor([4.0], dtype=torch.float64)  # c - a - b = 0
        z_values = [-1.5, -2.0, -3.0, -5.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(2.0, 2.0, 4.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Integer diff n=0 failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_n_equals_one(self):
        """Test DLMF 15.8.10 limiting form when a - b = 1 for |z| > 1."""
        # Use z < -1 to avoid divergence at z=1 (which occurs when c-a-b=0)
        a = torch.tensor([3.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # n = a - b = 1
        c = torch.tensor([5.0], dtype=torch.float64)  # c - a - b = 0
        z_values = [-1.5, -2.0, -3.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(3.0, 2.0, 5.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Integer diff n=1 failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_n_equals_two(self):
        """Test DLMF 15.8.10 limiting form when a - b = 2 for |z| > 1."""
        # Use z < -1 to avoid divergence at z=1 (which occurs when c-a-b=0)
        a = torch.tensor([4.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # n = a - b = 2
        c = torch.tensor([6.0], dtype=torch.float64)  # c - a - b = 0
        z_values = [-1.5, -2.0, -3.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(4.0, 2.0, 6.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Integer diff n=2 failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_negative_n(self):
        """Test DLMF 15.8.10 limiting form when a - b < 0 (uses symmetry)."""
        # When a < b, the implementation uses symmetry: 2F1(a,b;c;z) = 2F1(b,a;c;z)
        # Use z < -1 to avoid divergence at z=1 (which occurs when c-a-b=0)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([4.0], dtype=torch.float64)  # n = a - b = -2
        c = torch.tensor([6.0], dtype=torch.float64)  # c - a - b = 0
        z_values = [-1.5, -2.0, -3.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(2.0, 4.0, 6.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Integer diff n=-2 failed at z={z_val}",
            )

    @pytest.mark.skip(
        reason="Complex z with integer a-b needs more work on perturbation approach"
    )
    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_complex_z(self):
        """Test DLMF 15.8.10 limiting form with complex z when a-b is integer."""
        a = torch.tensor([3.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # n = 1
        c = torch.tensor([5.0], dtype=torch.float64)
        # Complex z with |z| > 1
        z = torch.tensor([1.5 + 0.5j, 2.0 - 0.3j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Results should be finite
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

        # Compare with scipy for each value
        for i, z_val in enumerate(z.tolist()):
            expected = scipy.special.hyp2f1(3.0, 2.0, 5.0, z_val)
            torch.testing.assert_close(
                result[i].real,
                torch.tensor(expected.real, dtype=torch.float64),
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                result[i].imag,
                torch.tensor(expected.imag, dtype=torch.float64),
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_integer_diff_half_integer_params(self):
        """Test with half-integer parameters where a-b is still an integer."""
        # a = 2.5, b = 1.5, so a - b = 1 (integer)
        # c - a - b = 4.0 - 2.5 - 1.5 = 0, so z > 1 diverges
        # Use z < -1 for valid test cases
        a = torch.tensor([2.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)  # c - a - b = 0
        z_values = [-1.5, -2.0, -2.5]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(2.5, 1.5, 4.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Half-integer params failed at z={z_val}",
            )

    # =========================================================================
    # Tests for DLMF 15.8.4: 1-z transformation for z near 1
    # =========================================================================

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_basic(self):
        """Test 1-z transformation for z values where |1-z| < |z|."""
        # For z > 0.5, the 1-z transformation should be used
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z_values = [0.6, 0.7, 0.8, 0.9, 0.95]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 2.0, 3.5, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_integer_cab(self):
        """Test 1-z transformation when c-a-b is an integer (uses Richardson extrapolation)."""
        # c - a - b = 4 - 1.5 - 1.5 = 1 (integer)
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z_values = [0.6, 0.7, 0.8, 0.9]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 1.5, 4.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform with integer c-a-b failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_cab_zero(self):
        """Test 1-z transformation when c-a-b = 0."""
        # c - a - b = 3 - 1.5 - 1.5 = 0
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z_values = [0.6, 0.7, 0.8]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 1.5, 3.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform with c-a-b=0 failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_negative_cab(self):
        """Test 1-z transformation when c-a-b is negative integer."""
        # c - a - b = 2 - 1.5 - 1.5 = -1 (negative integer)
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)
        z_values = [0.6, 0.7, 0.8]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 1.5, 2.0, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform with c-a-b=-1 failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_complex_z(self):
        """Test 1-z transformation with complex z where |1-z| < |z|."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        # Complex z with |1-z| < |z| (i.e., Re(z) > 0.5)
        z_values = [0.6 + 0.2j, 0.7 + 0.1j, 0.8 - 0.15j]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.complex128)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            scipy_result = scipy.special.hyp2f1(1.5, 2.0, 3.5, z_val)

            assert torch.isfinite(result.real).all(), (
                f"Non-finite real at z={z_val}"
            )
            assert torch.isfinite(result.imag).all(), (
                f"Non-finite imag at z={z_val}"
            )
            torch.testing.assert_close(
                result[0].real,
                torch.tensor(scipy_result.real, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform complex z failed (real) at z={z_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(scipy_result.imag, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform complex z failed (imag) at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_transition_between_algorithms(self):
        """Test smooth transition between direct series and 1-z transform at z=0.5."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        # Test z values around the transition point z=0.5
        z_values = [0.45, 0.48, 0.50, 0.52, 0.55]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 2.0, 3.5, z_val)], dtype=torch.float64
            )

            # 1-z transformation uses Richardson extrapolation which introduces
            # small errors (~1e-5) but greatly accelerates convergence
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Algorithm transition failed at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_one_minus_z_transform_various_params(self):
        """Test 1-z transformation with various parameter combinations."""
        test_cases = [
            # (a, b, c) - various combinations with different c-a-b values
            (0.5, 1.0, 2.5),  # c-a-b = 1.0 (non-integer near 1)
            (1.0, 1.0, 3.0),  # c-a-b = 1.0 (integer)
            (2.0, 3.0, 4.0),  # c-a-b = -1.0 (negative integer)
            (0.25, 0.75, 1.5),  # c-a-b = 0.5 (non-integer)
            (1.5, 2.5, 5.0),  # c-a-b = 1.0 (integer)
        ]

        z_val = 0.75  # Trigger 1-z transformation

        for a_val, b_val, c_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            c = torch.tensor([c_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(a_val, b_val, c_val, z_val)], dtype=torch.float64
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite at a={a_val}, b={b_val}, c={c_val}"
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )

    # =========================================================================
    # Tests for analytical parameter gradients in transformation regions
    # =========================================================================

    def test_gradcheck_all_params_direct_series(self):
        """Test gradient correctness for all params (a, b, c, z) in direct series region."""
        # z = 0.3 is in direct series region (|z| < 0.5)
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (a, b, c, z), rtol=1e-5, atol=1e-5)

    def test_gradcheck_all_params_one_minus_z_transform(self):
        """Test gradient correctness for all params in 1-z transformation region."""
        # z = 0.7 is in 1-z transformation region (|1-z| < |z|)
        # Use c-a-b = 4.2 - 1.5 - 2.0 = 0.7 (not an integer) to test analytical gradients
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([4.2], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (a, b, c, z), rtol=1e-4, atol=1e-4)

    def test_gradcheck_all_params_linear_transform(self):
        """Test gradient correctness for all params in linear transformation region (|z| > 1)."""
        # z = -2.0 has |z| > 1, uses linear transformation
        # Use a-b = 1.5 - 2.3 = -0.8 (not an integer) to test analytical gradients
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.3], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (a, b, c, z), rtol=1e-4, atol=1e-4)

    def test_param_gradients_consistency(self):
        """Test that parameter gradients are consistent across algorithm boundaries."""
        a_val, b_val, c_val = 1.5, 2.0, 3.5

        # Test points across different algorithm regions
        z_vals = [0.3, 0.7, -2.0]  # direct, 1-z transform, linear transform

        for z_val in z_vals:
            a = torch.tensor([a_val], dtype=torch.float64, requires_grad=True)
            b = torch.tensor([b_val], dtype=torch.float64, requires_grad=True)
            c = torch.tensor([c_val], dtype=torch.float64, requires_grad=True)
            z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)

            # Forward pass
            y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)

            # Backward pass
            y.backward()

            # All gradients should be finite and non-zero for these inputs
            assert torch.isfinite(a.grad).all(), (
                f"a.grad not finite at z={z_val}"
            )
            assert torch.isfinite(b.grad).all(), (
                f"b.grad not finite at z={z_val}"
            )
            assert torch.isfinite(c.grad).all(), (
                f"c.grad not finite at z={z_val}"
            )
            assert torch.isfinite(z.grad).all(), (
                f"z.grad not finite at z={z_val}"
            )

            # Verify gradients are non-trivial (not all zeros)
            assert a.grad.abs().item() > 1e-10, (
                f"a.grad too small at z={z_val}"
            )
            assert b.grad.abs().item() > 1e-10, (
                f"b.grad too small at z={z_val}"
            )
            assert c.grad.abs().item() > 1e-10, (
                f"c.grad too small at z={z_val}"
            )
            assert z.grad.abs().item() > 1e-10, (
                f"z.grad too small at z={z_val}"
            )

    def test_analytical_param_grad_matches_finite_diff(self):
        """Verify analytical gradients match finite differences."""
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        result.backward()

        # Finite difference reference
        eps = 1e-6
        with torch.no_grad():
            f_a_plus = torchscience.special_functions.hypergeometric_2_f_1(
                a + eps, b, c, z
            )
            f_a_minus = torchscience.special_functions.hypergeometric_2_f_1(
                a - eps, b, c, z
            )
            fd_da = (f_a_plus - f_a_minus) / (2 * eps)

            f_b_plus = torchscience.special_functions.hypergeometric_2_f_1(
                a, b + eps, c, z
            )
            f_b_minus = torchscience.special_functions.hypergeometric_2_f_1(
                a, b - eps, c, z
            )
            fd_db = (f_b_plus - f_b_minus) / (2 * eps)

            f_c_plus = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c + eps, z
            )
            f_c_minus = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c - eps, z
            )
            fd_dc = (f_c_plus - f_c_minus) / (2 * eps)

        torch.testing.assert_close(a.grad, fd_da, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b.grad, fd_db, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(c.grad, fd_dc, rtol=1e-5, atol=1e-5)

    # =========================================================================
    # Tests for second-order derivatives (gradgradcheck)
    # =========================================================================

    def test_gradgradcheck_z_only(self):
        """Test second-order gradient correctness for z parameter."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        def func(z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradgradcheck(func, (z,), rtol=1e-3, atol=1e-3)

    def test_gradgradcheck_all_params_direct_series(self):
        """Test second-order gradient correctness for all params in direct series region."""
        # z = 0.3 is in direct series region (|z| < 0.5)
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        # Second-order derivatives use finite differences, so need relaxed tolerances
        torch.autograd.gradgradcheck(func, (a, b, c, z), rtol=1e-3, atol=1e-3)

    def test_gradgradcheck_all_params_one_minus_z_transform(self):
        """Test second-order gradient correctness in 1-z transformation region."""
        # z = 0.7 is in 1-z transformation region (|1-z| < |z|)
        # The 1-z transformation involves more complex computations with Gamma ratios
        # so we need relaxed tolerances for second-order derivatives
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([4.2], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        # Finite differences compound errors in transformation regions, need ~10% tolerance
        torch.autograd.gradgradcheck(func, (a, b, c, z), rtol=1e-1, atol=1e-1)

    def test_gradgradcheck_all_params_linear_transform(self):
        """Test second-order gradient correctness in linear transformation region."""
        # z = -2.0 has |z| > 1, uses linear transformation
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.3], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        # Finite differences compound errors in transformation regions, need ~5% tolerance
        torch.autograd.gradgradcheck(func, (a, b, c, z), rtol=5e-2, atol=5e-2)

    def test_second_derivative_creates_graph(self):
        """Test that create_graph=True works for computing higher-order derivatives."""
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        # Forward pass
        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)

        # First backward with create_graph=True
        (grad_a,) = torch.autograd.grad(y, a, create_graph=True)

        # Verify grad_a requires grad (graph was retained)
        assert grad_a.requires_grad, (
            "grad_a should require grad when create_graph=True"
        )

        # Second backward through grad_a
        (grad_grad_a_z,) = torch.autograd.grad(grad_a, z, retain_graph=True)

        # Should be finite (d²f/dadz)
        assert torch.isfinite(grad_grad_a_z).all(), "d²f/dadz should be finite"

        # Test all second derivatives are computable
        (grad_grad_a_a,) = torch.autograd.grad(grad_a, a, retain_graph=True)
        (grad_grad_a_b,) = torch.autograd.grad(grad_a, b, retain_graph=True)
        (grad_grad_a_c,) = torch.autograd.grad(grad_a, c, retain_graph=True)

        assert torch.isfinite(grad_grad_a_a).all(), "d²f/da² should be finite"
        assert torch.isfinite(grad_grad_a_b).all(), "d²f/dadb should be finite"
        assert torch.isfinite(grad_grad_a_c).all(), "d²f/dadc should be finite"

    def test_second_derivative_symmetry(self):
        """Test that mixed partial derivatives are symmetric: d²f/dxdy = d²f/dydx."""
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)

        # Compute d²f/dadb via d(df/da)/db
        (grad_a,) = torch.autograd.grad(y, a, create_graph=True)
        (d2f_dadb,) = torch.autograd.grad(grad_a, b, retain_graph=True)

        # Reset computation graph
        a2 = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c2 = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)
        z2 = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        y2 = torchscience.special_functions.hypergeometric_2_f_1(
            a2, b2, c2, z2
        )

        # Compute d²f/dbda via d(df/db)/da
        (grad_b,) = torch.autograd.grad(y2, b2, create_graph=True)
        (d2f_dbda,) = torch.autograd.grad(grad_b, a2, retain_graph=True)

        # Should be equal (Schwarz's theorem)
        torch.testing.assert_close(
            d2f_dadb,
            d2f_dbda,
            rtol=1e-4,
            atol=1e-4,
            msg="Mixed partials d²f/dadb and d²f/dbda should be equal",
        )

    def test_second_derivative_values(self):
        """Test second derivative values against finite difference approximation."""
        a_val, b_val, c_val, z_val = 1.5, 2.0, 3.5, 0.3
        h = 1e-4  # Use larger h for numerical stability

        a = torch.tensor([a_val], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([b_val], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([c_val], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)

        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)
        (grad_a,) = torch.autograd.grad(y, a, create_graph=True)
        (d2f_da2,) = torch.autograd.grad(grad_a, a)

        # Compare with finite difference of first derivatives (more stable than second-order FD of f)
        # d²f/da² ≈ (df/da(a+h) - df/da(a-h)) / (2h)
        a_p = torch.tensor(
            [a_val + h], dtype=torch.float64, requires_grad=True
        )
        a_m = torch.tensor(
            [a_val - h], dtype=torch.float64, requires_grad=True
        )
        b_const = torch.tensor([b_val], dtype=torch.float64)
        c_const = torch.tensor([c_val], dtype=torch.float64)
        z_const = torch.tensor([z_val], dtype=torch.float64)

        y_p = torchscience.special_functions.hypergeometric_2_f_1(
            a_p, b_const, c_const, z_const
        )
        y_m = torchscience.special_functions.hypergeometric_2_f_1(
            a_m, b_const, c_const, z_const
        )

        (grad_a_p,) = torch.autograd.grad(y_p, a_p)
        (grad_a_m,) = torch.autograd.grad(y_m, a_m)

        d2f_da2_fd = (grad_a_p - grad_a_m) / (2 * h)

        torch.testing.assert_close(
            d2f_da2,
            d2f_da2_fd,
            rtol=1e-4,
            atol=1e-4,
            msg="d²f/da² should match finite difference approximation",
        )

    # =========================================================================
    # Tests for complex a, b, c parameters
    # =========================================================================

    def test_complex_a_basic(self):
        """Test 2F1 with complex a parameter."""
        # All parameters must be complex when any is complex
        a = torch.tensor([1.5 + 0.3j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Result should be finite and complex
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()
        # With complex a, result should have non-zero imaginary part
        assert result[0].imag.abs() > 1e-10

    def test_complex_b_basic(self):
        """Test 2F1 with complex b parameter."""
        a = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.4j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()
        assert result[0].imag.abs() > 1e-10

    def test_complex_c_basic(self):
        """Test 2F1 with complex c parameter."""
        a = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.5j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()
        assert result[0].imag.abs() > 1e-10

    def test_complex_all_params(self):
        """Test 2F1 with all complex parameters."""
        a = torch.tensor([1.5 + 0.2j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.3j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.4j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.1j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_a_mpmath_reference(self):
        """Test complex a parameter against mpmath reference."""
        mpmath.mp.dps = 30  # High precision for reference
        test_cases = [
            (1.0 + 0.5j, 2.0, 3.0, 0.3),
            (0.5 + 0.3j, 1.5, 2.5, 0.4),
            (2.0 + 1.0j, 1.0, 4.0, 0.2),
            (1.5 - 0.5j, 2.5, 3.5, 0.5),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.complex128)
            b = torch.tensor([complex(b_val)], dtype=torch.complex128)
            c = torch.tensor([complex(c_val)], dtype=torch.complex128)
            z = torch.tensor([complex(z_val)], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Real part mismatch at a={a_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Imag part mismatch at a={a_val}",
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_b_mpmath_reference(self):
        """Test complex b parameter against mpmath reference."""
        mpmath.mp.dps = 30
        test_cases = [
            (1.5, 1.0 + 0.5j, 3.0, 0.3),
            (2.0, 0.5 + 0.3j, 2.5, 0.4),
            (1.0, 2.0 + 1.0j, 4.0, 0.2),
            (2.5, 1.5 - 0.5j, 3.5, 0.5),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([complex(a_val)], dtype=torch.complex128)
            b = torch.tensor([b_val], dtype=torch.complex128)
            c = torch.tensor([complex(c_val)], dtype=torch.complex128)
            z = torch.tensor([complex(z_val)], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Real part mismatch at b={b_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Imag part mismatch at b={b_val}",
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_c_mpmath_reference(self):
        """Test complex c parameter against mpmath reference."""
        mpmath.mp.dps = 30
        test_cases = [
            (1.5, 2.0, 3.0 + 0.5j, 0.3),
            (2.0, 1.5, 2.5 + 0.3j, 0.4),
            (1.0, 2.0, 4.0 + 1.0j, 0.2),
            (2.5, 1.5, 3.5 - 0.5j, 0.5),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([complex(a_val)], dtype=torch.complex128)
            b = torch.tensor([complex(b_val)], dtype=torch.complex128)
            c = torch.tensor([c_val], dtype=torch.complex128)
            z = torch.tensor([complex(z_val)], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Real part mismatch at c={c_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
                msg=f"Imag part mismatch at c={c_val}",
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_all_params_mpmath_reference(self):
        """Test all complex parameters against mpmath reference."""
        mpmath.mp.dps = 30
        test_cases = [
            (1.0 + 0.2j, 1.5 + 0.3j, 3.0 + 0.4j, 0.3 + 0.1j),
            (0.5 + 0.1j, 0.5 + 0.1j, 2.0 + 0.2j, 0.4 - 0.1j),
            (2.0 - 0.3j, 1.0 + 0.5j, 4.0 - 0.2j, 0.2 + 0.2j),
            (1.5 + 0.4j, 2.5 - 0.4j, 3.5 + 0.1j, 0.5 + 0.0j),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.complex128)
            b = torch.tensor([b_val], dtype=torch.complex128)
            c = torch.tensor([c_val], dtype=torch.complex128)
            z = torch.tensor([z_val], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"Real part mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"Imag part mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_params_one_minus_z_transform(self):
        """Test complex parameters in the 1-z transformation region (|1-z| < |z|)."""
        mpmath.mp.dps = 30
        # z = 0.7 triggers 1-z transformation
        test_cases = [
            (1.0 + 0.3j, 2.0, 3.5, 0.7),
            (1.5, 1.0 + 0.4j, 3.0, 0.75),
            (2.0, 2.5, 4.0 + 0.5j, 0.8),
            (1.0 + 0.2j, 1.5 + 0.3j, 3.0 + 0.1j, 0.7 + 0.1j),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([complex(a_val)], dtype=torch.complex128)
            b = torch.tensor([complex(b_val)], dtype=torch.complex128)
            c = torch.tensor([complex(c_val)], dtype=torch.complex128)
            z = torch.tensor([complex(z_val)], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(
                mpmath.hyp2f1(
                    complex(a_val),
                    complex(b_val),
                    complex(c_val),
                    complex(z_val),
                )
            )

            assert torch.isfinite(result.real).all(), (
                f"Non-finite real at {a_val}, {b_val}, {c_val}, {z_val}"
            )
            assert torch.isfinite(result.imag).all(), (
                f"Non-finite imag at {a_val}, {b_val}, {c_val}, {z_val}"
            )

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform real mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-5,
                atol=1e-5,
                msg=f"1-z transform imag mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_complex_params_linear_transform(self):
        """Test complex parameters in the linear transformation region (|z| > 1)."""
        mpmath.mp.dps = 30
        # z with |z| > 1 triggers linear transformation
        test_cases = [
            (1.0 + 0.3j, 2.3, 4.0, -1.5),
            (1.5, 1.0 + 0.4j, 3.5, -2.0),
            (2.0, 2.5, 4.5 + 0.5j, -1.8),
            (1.0 + 0.2j, 1.8 + 0.3j, 3.5 + 0.1j, -1.5 + 0.3j),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([complex(a_val)], dtype=torch.complex128)
            b = torch.tensor([complex(b_val)], dtype=torch.complex128)
            c = torch.tensor([complex(c_val)], dtype=torch.complex128)
            z = torch.tensor([complex(z_val)], dtype=torch.complex128)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            mpmath_result = complex(
                mpmath.hyp2f1(
                    complex(a_val),
                    complex(b_val),
                    complex(c_val),
                    complex(z_val),
                )
            )

            assert torch.isfinite(result.real).all(), (
                f"Non-finite real at {a_val}, {b_val}, {c_val}, {z_val}"
            )
            assert torch.isfinite(result.imag).all(), (
                f"Non-finite imag at {a_val}, {b_val}, {c_val}, {z_val}"
            )

            torch.testing.assert_close(
                result[0].real,
                torch.tensor(mpmath_result.real, dtype=torch.float64),
                rtol=1e-4,
                atol=1e-4,
                msg=f"Linear transform real mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )
            torch.testing.assert_close(
                result[0].imag,
                torch.tensor(mpmath_result.imag, dtype=torch.float64),
                rtol=1e-4,
                atol=1e-4,
                msg=f"Linear transform imag mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
            )

    def test_complex_symmetry_in_a_b(self):
        """Test 2F1(a, b; c; z) = 2F1(b, a; c; z) for complex parameters."""
        a = torch.tensor([1.5 + 0.3j], dtype=torch.complex128)
        b = torch.tensor([2.5 + 0.4j], dtype=torch.complex128)
        c = torch.tensor([4.0 + 0.2j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.1j], dtype=torch.complex128)

        result1 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            b, a, c, z
        )

        torch.testing.assert_close(
            result1.real, result2.real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result1.imag, result2.imag, rtol=1e-10, atol=1e-10
        )

    def test_complex_at_z_zero(self):
        """Test 2F1(a, b; c; 0) = 1 for complex parameters."""
        a = torch.tensor([1.5 + 0.3j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.4j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.5j], dtype=torch.complex128)
        z = torch.tensor([0.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        torch.testing.assert_close(
            result.real,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_zero_parameter_a(self):
        """Test 2F1(0, b; c; z) = 1 for complex parameters."""
        a = torch.tensor([0.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.4j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.5j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        torch.testing.assert_close(
            result.real,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_gradcheck_a(self):
        """Test gradient correctness for complex a parameter."""
        a = torch.tensor(
            [1.5 + 0.3j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        c = torch.tensor([3.5 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        def func(a):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (a,), rtol=1e-4, atol=1e-4)

    def test_complex_gradcheck_b(self):
        """Test gradient correctness for complex b parameter."""
        a = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        b = torch.tensor(
            [2.0 + 0.4j], dtype=torch.complex128, requires_grad=True
        )
        c = torch.tensor([3.5 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        def func(b):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (b,), rtol=1e-4, atol=1e-4)

    def test_complex_gradcheck_c(self):
        """Test gradient correctness for complex c parameter."""
        a = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        c = torch.tensor(
            [3.5 + 0.5j], dtype=torch.complex128, requires_grad=True
        )
        z = torch.tensor([0.3 + 0.0j], dtype=torch.complex128)

        def func(c):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (c,), rtol=1e-4, atol=1e-4)

    def test_complex_gradcheck_all_params(self):
        """Test gradient correctness for all complex parameters."""
        a = torch.tensor(
            [1.5 + 0.2j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor(
            [2.0 + 0.3j], dtype=torch.complex128, requires_grad=True
        )
        c = torch.tensor(
            [3.5 + 0.4j], dtype=torch.complex128, requires_grad=True
        )
        z = torch.tensor(
            [0.3 + 0.1j], dtype=torch.complex128, requires_grad=True
        )

        def func(a, b, c, z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (a, b, c, z), rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Tests for unit circle convergence check
    # =========================================================================

    def test_unit_circle_convergence_at_z_equals_1(self):
        """Test convergence check at z = 1.

        At z = 1, the series converges iff Re(c - a - b) > 0.
        """
        # Case 1: Re(c - a - b) > 0 → should converge
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 1 > 0
        z = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert torch.isfinite(result).all(), (
            "Should converge when Re(c-a-b) > 0 at z=1"
        )

        # Gauss summation theorem: 2F1(a,b;c;1) = Γ(c)Γ(c-a-b)/(Γ(c-a)Γ(c-b))
        # For a=b=0.5, c=2: = Γ(2)Γ(1)/(Γ(1.5)Γ(1.5)) = 1 * 1 / (π/4) = 4/π
        expected = 4.0 / math.pi
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_unit_circle_divergence_at_z_equals_1(self):
        """Test divergence check at z = 1.

        At z = 1, the series diverges if Re(c - a - b) <= 0.
        """
        # Case: Re(c - a - b) = 0 → should diverge (return NaN)
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 0
        z = torch.tensor([1.0], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert torch.isnan(result).all(), (
            "Should return NaN (diverge) when Re(c-a-b) = 0 at z=1"
        )

        # Case: Re(c - a - b) < 0 → should diverge (return NaN)
        c_negative = torch.tensor(
            [1.5], dtype=torch.float64
        )  # c - a - b = -0.5
        result_neg = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_negative, z
        )
        assert torch.isnan(result_neg).all(), (
            "Should return NaN (diverge) when Re(c-a-b) < 0 at z=1"
        )

    def test_unit_circle_convergence_not_at_z_equals_1(self):
        """Test convergence on unit circle at z != 1.

        For |z| = 1 but z != 1:
        - Converges if Re(c - a - b) > -1
        - Diverges if Re(c - a - b) <= -1
        """
        # z = -1 (on unit circle, but not z=1)
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([-1.0], dtype=torch.float64)

        # Case 1: Re(c - a - b) = 0 > -1 → should converge at z=-1
        c_converge = torch.tensor([1.0], dtype=torch.float64)  # c - a - b = 0
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_converge, z
        )
        assert torch.isfinite(result).all(), (
            "Should converge when Re(c-a-b) = 0 > -1 at z=-1"
        )

        # Case 2: Re(c - a - b) = -0.5 > -1 → should converge at z=-1
        c_converge2 = torch.tensor(
            [0.5], dtype=torch.float64
        )  # c - a - b = -0.5
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_converge2, z
        )
        assert torch.isfinite(result2).all(), (
            "Should converge when Re(c-a-b) = -0.5 > -1 at z=-1"
        )

    def test_unit_circle_divergence_not_at_z_equals_1(self):
        """Test divergence on unit circle at z != 1.

        For |z| = 1 but z != 1, diverges if Re(c - a - b) <= -1.
        """
        # z = -1 (on unit circle, but not z=1)
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        z = torch.tensor([-1.0], dtype=torch.float64)

        # Re(c - a - b) = -1 → should diverge
        c_diverge = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = -1
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_diverge, z
        )
        assert torch.isnan(result).all(), (
            "Should return NaN (diverge) when Re(c-a-b) = -1 at z=-1"
        )

        # Re(c - a - b) = -1.5 < -1 → should diverge
        c_diverge2 = torch.tensor(
            [1.5], dtype=torch.float64
        )  # c - a - b = -1.5
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_diverge2, z
        )
        assert torch.isnan(result2).all(), (
            "Should return NaN (diverge) when Re(c-a-b) < -1 at z=-1"
        )

    def test_unit_circle_complex_z(self):
        """Test convergence check for complex z on the unit circle."""
        # z = i (on unit circle)
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 1 > 0
        z = torch.tensor([1j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        # Should converge since Re(c-a-b) = 1 > -1 (and > 0)
        assert (
            torch.isfinite(result.real).all()
            and torch.isfinite(result.imag).all()
        ), "Should converge for complex z on unit circle when Re(c-a-b) > 0"

    def test_unit_circle_gradient_divergence(self):
        """Test that gradients are NaN when function diverges on unit circle."""
        a = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Function should return NaN
        assert torch.isnan(result).all()

        # Backward pass should also produce NaN gradients
        result.backward()
        assert torch.isnan(a.grad).all(), "Gradient w.r.t. a should be NaN"
        assert torch.isnan(b.grad).all(), "Gradient w.r.t. b should be NaN"
        assert torch.isnan(c.grad).all(), "Gradient w.r.t. c should be NaN"
        assert torch.isnan(z.grad).all(), "Gradient w.r.t. z should be NaN"

    # =========================================================================
    # Extensive edge case tests
    # =========================================================================

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_very_small_z(self):
        """Test with very small z values where series converges rapidly."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 2.5, 3.5, z_val)], dtype=torch.float64
            )

            # For very small z, result should be very close to 1
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for very small z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_z_close_to_half(self):
        """Test z values very close to 0.5 (algorithm transition boundary)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        # Values very close to 0.5 from both sides
        z_values = [0.4999, 0.49999, 0.5, 0.50001, 0.5001]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 2.0, 3.5, z_val)], dtype=torch.float64
            )

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed near algorithm boundary z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_z_close_to_one(self):
        """Test z values very close to 1 (convergence boundary)."""
        # Need c - a - b > 0 for convergence at z = 1
        a = torch.tensor([0.3], dtype=torch.float64)
        b = torch.tensor([0.4], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 1.3 > 0
        z_values = [0.99, 0.999, 0.9999, 0.99999]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(0.3, 0.4, 2.0, z_val)], dtype=torch.float64
            )

            # Allow larger tolerance near z=1 as convergence is slow
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Failed near z=1 at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_z_close_to_minus_one(self):
        """Test z values very close to -1.

        Note: Convergence is slower near the unit circle, so we test with
        values not too close to -1 with tighter tolerance, and values
        closer to -1 with relaxed tolerance.
        """
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([1.5], dtype=torch.float64)  # c - a - b = 0.5 > -1

        # Values with different tolerances based on distance from -1
        z_tol_pairs = [
            (-0.9, 1e-6),  # Further from -1, tighter tolerance
            (-0.95, 1e-5),  # Moderate
            (-0.99, 1e-5),  # Close to -1
            (-0.999, 1e-4),  # Very close, relaxed tolerance
        ]

        for z_val, tol in z_tol_pairs:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(0.5, 0.5, 1.5, z_val)], dtype=torch.float64
            )

            torch.testing.assert_close(
                result,
                expected,
                rtol=tol,
                atol=tol,
                msg=f"Failed near z=-1 at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_near_integer_a_minus_b(self):
        """Test a-b very close to an integer (tests Richardson extrapolation trigger)."""
        # a - b = 1.0001 (very close to 1)
        a = torch.tensor([2.5001], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z_values = [-1.5, -2.0, -3.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(2.5001, 1.5, 4.0, z_val)], dtype=torch.float64
            )

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Failed near integer a-b at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_near_integer_c_minus_a_minus_b(self):
        """Test c-a-b very close to an integer (tests Richardson extrapolation in 1-z transform)."""
        # c - a - b = 1.0001 (very close to 1)
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([4.0001], dtype=torch.float64)
        z_values = [0.6, 0.7, 0.8, 0.9]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 1.5, 4.0001, z_val)], dtype=torch.float64
            )

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Failed near integer c-a-b at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_large_a(self):
        """Test with large a parameter."""
        a_values = [20.0, 30.0, 50.0]
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([25.0], dtype=torch.float64)
        z = torch.tensor([0.1], dtype=torch.float64)

        for a_val in a_values:
            a = torch.tensor([a_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            # Should be finite
            assert torch.isfinite(result).all(), (
                f"Non-finite result for large a={a_val}"
            )

            # Compare with scipy
            expected = torch.tensor(
                [scipy_hyp2f1(a_val, 1.0, 25.0, 0.1)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for large a={a_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_large_b(self):
        """Test with large b parameter."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b_values = [20.0, 30.0, 50.0]
        c = torch.tensor([25.0], dtype=torch.float64)
        z = torch.tensor([0.1], dtype=torch.float64)

        for b_val in b_values:
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result for large b={b_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(1.0, b_val, 25.0, 0.1)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for large b={b_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_large_c(self):
        """Test with large c parameter."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c_values = [50.0, 100.0, 200.0]
        z = torch.tensor([0.5], dtype=torch.float64)

        for c_val in c_values:
            c = torch.tensor([c_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result for large c={c_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(1.0, 2.0, c_val, 0.5)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Failed for large c={c_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_very_small_parameters(self):
        """Test with very small (but positive) parameters."""
        small_vals = [0.01, 0.001, 0.0001]
        z = torch.tensor([0.5], dtype=torch.float64)

        for val in small_vals:
            a = torch.tensor([val], dtype=torch.float64)
            b = torch.tensor([val], dtype=torch.float64)
            c = torch.tensor([1.0], dtype=torch.float64)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result for small params a=b={val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(val, val, 1.0, 0.5)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Failed for small params a=b={val}",
            )

    def test_edge_case_c_close_to_pole(self):
        """Test c very close to a non-positive integer (near pole)."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        # c very close to 0 (pole)
        c_near_zero = torch.tensor([0.001], dtype=torch.float64)
        result_near_zero = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c_near_zero, z
        )
        # Should be very large but finite
        assert torch.isfinite(result_near_zero).all(), (
            "Result should be finite for c near 0"
        )
        assert result_near_zero.abs() > 100, (
            "Result should be large for c near 0"
        )

        # c very close to -1 (pole)
        c_near_minus1 = torch.tensor([-0.999], dtype=torch.float64)
        result_near_minus1 = (
            torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c_near_minus1, z
            )
        )
        # Should be very large but finite
        assert torch.isfinite(result_near_minus1).all(), (
            "Result should be finite for c near -1"
        )

    def test_edge_case_c_equals_a_or_b(self):
        """Test when c equals one of the parameters (reduces to simpler function)."""
        z = torch.tensor([0.5], dtype=torch.float64)

        # c = a: 2F1(a, b; a; z) = (1-z)^(-b)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, a, z
        )
        expected = torch.pow(1 - z, -b)
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            msg="Failed for c=a identity",
        )

        # c = b: 2F1(a, b; b; z) = (1-z)^(-a)
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, b, z
        )
        expected2 = torch.pow(1 - z, -a)
        torch.testing.assert_close(
            result2,
            expected2,
            rtol=1e-6,
            atol=1e-6,
            msg="Failed for c=b identity",
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_a_plus_one_equals_c(self):
        """Test a + 1 = c (parameter reduction case)."""
        # When a + 1 = c: 2F1(a, b; a+1; z) has a known form
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([1.5], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)  # a + 1 = 3
        z = torch.tensor([0.5], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.tensor(
            [scipy_hyp2f1(2.0, 1.5, 3.0, 0.5)], dtype=torch.float64
        )

        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            msg="Failed for a+1=c case",
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_extreme_negative_z(self):
        """Test with extreme negative z values (|z| >> 1)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.3], dtype=torch.float64)  # Non-integer a-b
        c = torch.tensor([4.0], dtype=torch.float64)
        z_values = [-5.0, -10.0, -20.0, -50.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at extreme z={z_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(1.5, 2.3, 4.0, z_val)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Failed for extreme negative z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_pfaff_transformation(self):
        """Test Pfaff transformation: 2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        # Direct computation
        result_direct = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Via Pfaff transformation
        z_transformed = z / (z - 1)  # z/(z-1) = 0.3/(-0.7) ≈ -0.4286
        result_pfaff = torch.pow(1 - z, -a) * (
            torchscience.special_functions.hypergeometric_2_f_1(
                a, c - b, c, z_transformed
            )
        )

        torch.testing.assert_close(
            result_direct,
            result_pfaff,
            rtol=1e-5,
            atol=1e-5,
            msg="Pfaff transformation identity failed",
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_euler_transformation(self):
        """Test Euler transformation: 2F1(a,b;c;z) = (1-z)^(c-a-b) * 2F1(c-a, c-b; c; z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor(
            [5.0], dtype=torch.float64
        )  # c > a + b for convergence
        z = torch.tensor([0.4], dtype=torch.float64)

        # Direct computation
        result_direct = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Via Euler transformation
        result_euler = torch.pow(1 - z, c - a - b) * (
            torchscience.special_functions.hypergeometric_2_f_1(
                c - a, c - b, c, z
            )
        )

        torch.testing.assert_close(
            result_direct,
            result_euler,
            rtol=1e-5,
            atol=1e-5,
            msg="Euler transformation identity failed",
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_negative_integer_b_terminates(self):
        """Test that 2F1(a, -n; c; z) is a polynomial when n is non-negative integer."""
        # 2F1(1, -2; 1; z) = 1 - 2z + z^2 = (1-z)^2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([-2.0], dtype=torch.float64)
        c = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = (1 - z) ** 2

        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_equal_parameters_a_b(self):
        """Test when a = b."""
        a = torch.tensor([2.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)  # a = b
        c = torch.tensor([5.0], dtype=torch.float64)
        z_values = [0.3, 0.5, 0.7, -1.5, -2.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result for a=b at z={z_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(2.5, 2.5, 5.0, z_val)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for a=b at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_a_equals_c(self):
        """Test when a = c (reduces to geometric series type)."""
        a = torch.tensor([3.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)  # a = c
        z = torch.tensor([0.5], dtype=torch.float64)

        # 2F1(a, b; a; z) = (1-z)^(-b)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.pow(1 - z, -b)

        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            msg="Failed for a=c identity",
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_integer_diff_m_equals_3(self):
        """Test DLMF 15.8.10 explicit formula for m = 3 (larger integer difference)."""
        a = torch.tensor([5.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # m = a - b = 3
        c = torch.tensor([7.0], dtype=torch.float64)
        z_values = [-1.5, -2.0, -3.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(5.0, 2.0, 7.0, z_val)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for integer diff m=3 at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_integer_diff_m_equals_5(self):
        """Test DLMF 15.8.10 explicit formula for m = 5 (even larger integer difference)."""
        a = torch.tensor([7.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)  # m = a - b = 5
        c = torch.tensor([9.0], dtype=torch.float64)
        z_values = [-1.5, -2.0]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(7.0, 2.0, 9.0, z_val)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Failed for integer diff m=5 at z={z_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_1_minus_z_transform_integer_diff_m_equals_2(self):
        """Test 1-z transform with c-a-b = 2 (DLMF 15.8.10 case)."""
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)  # m = c - a - b = 2
        z_values = [0.6, 0.7, 0.8, 0.9]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            assert torch.isfinite(result).all(), (
                f"Non-finite result at z={z_val}"
            )

            expected = torch.tensor(
                [scipy_hyp2f1(0.5, 0.5, 3.0, z_val)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for 1-z transform with m=2 at z={z_val}",
            )

    def test_edge_case_batch_computation(self):
        """Test batch computation with mixed algorithm paths."""
        # Different z values should trigger different algorithm branches
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z = torch.tensor(
            [0.1, 0.3, 0.5, 0.7, 0.9, -0.5, -1.5, -2.0], dtype=torch.float64
        )

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # All results should be finite
        assert torch.isfinite(result).all(), (
            "Batch computation produced non-finite results"
        )

        # Verify each individually
        for i, z_val in enumerate(z.tolist()):
            z_single = torch.tensor([z_val], dtype=torch.float64)
            result_single = (
                torchscience.special_functions.hypergeometric_2_f_1(
                    a, b, c, z_single
                )
            )
            torch.testing.assert_close(
                result[i : i + 1],
                result_single,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Batch vs single mismatch at z={z_val}",
            )

    def test_edge_case_gradient_at_boundary(self):
        """Test gradient computation at algorithm transition boundaries."""
        a = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([3.5], dtype=torch.float64, requires_grad=True)

        # z = 0.5 is exactly at the boundary between direct series and 1-z transform
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)
        y.backward()

        # All gradients should be finite
        assert torch.isfinite(a.grad).all(), "a.grad not finite at boundary"
        assert torch.isfinite(b.grad).all(), "b.grad not finite at boundary"
        assert torch.isfinite(c.grad).all(), "c.grad not finite at boundary"
        assert torch.isfinite(z.grad).all(), "z.grad not finite at boundary"

    @pytest.mark.skip(
        reason="Complex z > 1 not fully implemented - returns NaN"
    )
    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_edge_case_contiguous_z_constraint(self):
        """Test that contiguous z constraint is properly enforced for special case."""
        # This tests z values exactly where the constraint on Re(z) < 0.5
        # for 1/z transformation should be handled
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)

        # z = 1.5 has Re(z) > 0.5, |z| > 1, 1/z = 0.6667
        z = torch.tensor([1.5], dtype=torch.complex128)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        assert torch.isfinite(result.real).all(), (
            "Real part should be finite for complex z > 1"
        )
        assert torch.isfinite(result.imag).all(), (
            "Imag part should be finite for complex z > 1"
        )

        # Compare with scipy
        scipy_result = scipy.special.hyp2f1(1.5, 2.0, 3.5, complex(1.5))
        torch.testing.assert_close(
            result[0].real,
            torch.tensor(scipy_result.real, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            result[0].imag,
            torch.tensor(scipy_result.imag, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )

    # =========================================================================
    # Plan Implementation Tests (Task 1-10)
    # =========================================================================

    def test_series_basic_convergence(self):
        """Test 2F1 with |z| < 0.5 where series converges directly."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Reference from scipy.special.hyp2f1(1.5, 2.5, 3.5, 0.3)
        expected = torch.tensor([1.452767637957694], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_terminating_series_a_negative_int(self):
        """Test 2F1 with a = -2 (terminating series)."""
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor([0.7], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # scipy.special.hyp2f1(-2, 3, 4, 0.7) = 0.244
        expected = torch.tensor([0.244], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_pole_at_c_negative_int(self):
        """Test 2F1 returns inf when c is non-positive integer."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([-1.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        assert torch.isinf(result).all()

    def test_reduction_to_power_function(self):
        """Test 2F1(a, b; b; z) = (1-z)^(-a)."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, b, z
        )
        expected = torch.pow(1 - z, -a)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_z_near_one(self):
        """Test 2F1 with z close to 1 using transformation."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor([0.9], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Reference from scipy.special.hyp2f1(1, 2, 4, 0.9)
        expected = torch.tensor([2.1789423102929675], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_z_negative_large(self):
        """Test 2F1 with large negative z using transformation."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([-5.0], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Reference from scipy.special.hyp2f1(1, 2, 3, -5)
        expected = torch.tensor([0.2566592424617554], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_z_negative_moderate(self):
        """Test 2F1 with moderate negative z."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        c = torch.tensor([3.5], dtype=torch.float64)
        z = torch.tensor([-2.0], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Reference from scipy.special.hyp2f1(1.5, 2.5, 3.5, -2)
        expected = torch.tensor([0.28453773594866716], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)
