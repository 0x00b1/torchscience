import math

import pytest
import sympy
import torch
import torch.testing
from sympy import I, N, symbols

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SpecialValue,
    SymbolicDerivativeVerifier,
    ToleranceConfig,
)


def sympy_chebyshev_t(
    v: float | complex, z: float | complex
) -> float | complex:
    """Wrapper for SymPy Chebyshev T function via cos(v * acos(z))."""
    if isinstance(v, complex):
        sympy_v = sympy.Float(v.real) + I * sympy.Float(v.imag)
    else:
        sympy_v = sympy.Float(v)

    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)

    result = N(sympy.cos(sympy_v * sympy.acos(sympy_z)), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_chebyshev_t_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for Chebyshev polynomial T."""
    v, z = symbols("v z")
    # T_v(z) = cos(v * acos(z))
    expr = sympy.cos(v * sympy.acos(z))
    return SymbolicDerivativeVerifier(expr, [v, z])


def reference_chebyshev_t(v, z):
    """Reference implementation using torch.cos(v * torch.acos(z))."""
    if torch.is_complex(v) or torch.is_complex(z):
        if not torch.is_complex(z):
            z = z.to(
                torch.complex128
                if z.dtype == torch.float64
                else torch.complex64
            )
        if not torch.is_complex(v):
            v = v.to(z.dtype)
    return torch.cos(v * torch.acos(z))


def _check_recurrence(func) -> bool:
    """Check T_n(z) = 2z*T_{n-1}(z) - T_{n-2}(z)."""
    z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    for n in range(2, 10):
        v_n = torch.tensor([float(n)], dtype=torch.float64)
        v_n_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
        v_n_2 = torch.tensor([float(n - 2)], dtype=torch.float64)

        left = func(v_n, z)
        right = 2 * z * func(v_n_1, z) - func(v_n_2, z)
        if not torch.allclose(left, right, rtol=1e-10, atol=1e-10):
            return False
    return True


class TestChebyshevPolynomialT(OpTestCase):
    """Tests for Chebyshev polynomial of the first kind."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="chebyshev_polynomial_t",
            func=torchscience.special_functions.chebyshev_polynomial_t,
            arity=2,
            input_specs=[
                InputSpec(
                    name="v",
                    position=0,
                    default_real_range=(0.0, 10.0),
                    can_be_integer=True,
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=1,
                    default_real_range=(-0.99, 0.99),
                    supports_grad=True,
                ),
            ],
            sympy_func=lambda v, z: sympy.cos(v * sympy.acos(z)),
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_complex_dtypes",  # No complex kernel implementation
                "test_dtype_preservation",  # Includes complex dtypes which are not implemented
                "test_gradcheck_complex",  # No complex kernel implementation
                "test_gradgradcheck_complex",  # No complex kernel implementation
                "test_gradgradcheck_real",  # backward_backward kernel needs hyperbolic domain support
                "test_sympy_reference_complex",  # No complex kernel implementation
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="three_term_recurrence",
                    check_fn=_check_recurrence,
                    description="T_n(z) = 2z*T_{n-1}(z) - T_{n-2}(z)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=1.0,
                    description="T_0(z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.5),
                    expected=0.5,
                    description="T_1(z) = z",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=-0.5,
                    description="T_2(0.5) = 2*0.25-1",
                ),
            ],
            singularities=[],
            # Sparse not supported: T_v(0) = cos(v*π/2) ≠ 0 in general
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            # Quantized has too much precision loss for this function
            supports_quantized=False,
            supports_meta=True,
        )

    # =========================================================================
    # Chebyshev-specific tests (not covered by framework)
    # =========================================================================

    def test_integer_degree_polynomial_values(self):
        """Test exact polynomial values for integer degrees."""
        z = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float64)

        # T_0(x) = 1
        v = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected)

        # T_1(x) = x
        v = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        torch.testing.assert_close(result, z)

        # T_2(x) = 2x^2 - 1
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = 2 * z**2 - 1
        torch.testing.assert_close(result, expected)

        # T_3(x) = 4x^3 - 3x
        v = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = 4 * z**3 - 3 * z
        torch.testing.assert_close(result, expected)

    def test_negative_degree_symmetry(self):
        """Test T_{-n}(z) = T_n(z)."""
        z = torch.tensor([0.5, 0.3, -0.7], dtype=torch.float64)
        for n in range(6):
            v_pos = torch.tensor([float(n)], dtype=torch.float64)
            v_neg = torch.tensor([float(-n)], dtype=torch.float64)
            result_pos = torchscience.special_functions.chebyshev_polynomial_t(
                v_pos, z
            )
            result_neg = torchscience.special_functions.chebyshev_polynomial_t(
                v_neg, z
            )
            torch.testing.assert_close(
                result_pos, result_neg, rtol=1e-10, atol=1e-10
            )

    def test_non_integer_degree(self):
        """Test non-integer degree uses analytic continuation."""
        z = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skip(reason="No complex kernel implementation")
    def test_complex_z_real_v(self):
        """Test complex z with real v."""
        z = torch.tensor(
            [1.0 + 0.1j, 0.5 + 0.5j, -0.5 - 0.5j], dtype=torch.complex128
        )
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skip(reason="No complex kernel implementation")
    def test_complex_z_complex_v(self):
        """Test complex z with complex v."""
        z = torch.tensor([0.5 + 0.2j], dtype=torch.complex128)
        v = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_continuation_z_greater_than_1(self):
        """Test T_v(z) = cosh(v * acosh(z)) for z > 1."""
        z = torch.tensor([1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.cosh(v * torch.acosh(z))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_continuation_z_less_than_minus_1(self):
        """Test T_v(z) = cos(v*pi) * cosh(v * acosh(-z)) for z < -1."""
        z = torch.tensor([-1.5, -2.0, -3.0, -5.0], dtype=torch.float64)
        v = torch.tensor([0.25], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.cos(v * torch.tensor(math.pi)) * torch.cosh(
            v * torch.acosh(-z)
        )
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_z_outside_domain_integer_v(self):
        """Test that real z outside [-1, 1] with integer v uses recurrence."""
        z = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        # T_2(z) = 2z^2 - 1
        expected = torch.tensor([3.5, 7.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_continuity_at_z_equals_1(self):
        """Test continuity at z = 1."""
        v = torch.tensor([2.5], dtype=torch.float64)
        z_inside = torch.tensor([1.0 - 1e-8], dtype=torch.float64)
        z_outside = torch.tensor([1.0 + 1e-8], dtype=torch.float64)

        result_inside = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_inside
        )
        result_outside = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_outside
        )

        # Both should be close to T_v(1) = 1
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            result_inside, expected, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            result_outside, expected, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.skip(reason="Complex dtype not implemented")
    def test_branch_cut_near_plus_one(self):
        """Test behavior near z = 1 with small imaginary parts."""
        z_above = torch.tensor([1.0 + 0.01j], dtype=torch.complex128)
        z_below = torch.tensor([1.0 - 0.01j], dtype=torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)

        result_above = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_above
        )
        result_below = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_below
        )

        # Results should be conjugates
        torch.testing.assert_close(
            result_above, result_below.conj(), rtol=1e-6, atol=1e-6
        )

    def test_gradcheck_z_greater_than_1(self):
        """Test gradient correctness for z > 1."""
        z = torch.tensor(
            [1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z_less_than_minus_1(self):
        """Test gradient correctness for z < -1."""
        z = torch.tensor(
            [-1.5, -2.0, -3.0], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_v_noninteger(self):
        """Test gradient correctness for non-integer v."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (v,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_boundary_gradient_at_z_equals_1(self):
        """Test gradient at z = 1 boundary."""
        z = torch.tensor(
            [1.0 - 1e-10], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        y = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        y.backward()

        # At z = 1: dT_v/dz = v^2
        expected_grad = v * v
        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-3, atol=1e-3)

    def test_unrolling_boundary(self):
        """Test degrees at the unrolling boundary (n=7,8,9) work correctly."""
        z = torch.linspace(-1, 1, 50, dtype=torch.float64)
        for n in [6, 7, 8, 9, 10, 15, 20]:
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype):
        """Test forward pass with low-precision dtypes."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype

        # Compare against float32 reference
        z_f32 = z.to(torch.float32)
        v_f32 = v.to(torch.float32)
        expected = torchscience.special_functions.chebyshev_polynomial_t(
            v_f32, z_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    def test_integer_degree_matches_reference(self):
        """Test that integer degree recurrence matches cos(n*arccos(x))."""
        z = torch.linspace(-1, 1, 100, dtype=torch.float64)
        for n in range(10):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_very_large_integer_degrees(self):
        """Test correctness for very large integer degrees (n > 1000).

        For large n, the recurrence can accumulate numerical errors.
        We verify against the analytic formula cos(n * arccos(z)).
        """
        z = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        large_degrees = [100, 500, 1000, 2000, 5000]

        for n in large_degrees:
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)
            # Relax tolerance for very large degrees due to accumulated error
            rtol = 1e-8 if n <= 1000 else 1e-6
            torch.testing.assert_close(result, expected, rtol=rtol, atol=1e-10)

    def test_numerical_stability_extreme_inputs(self):
        """Test numerical stability with extreme input values.

        Tests:
        - z very close to boundaries (±1)
        - Very small z values
        - Large z values (hyperbolic domain)
        - Combinations of extreme v and z
        """
        # z very close to +1 and -1 (within machine epsilon)
        eps = torch.finfo(torch.float64).eps
        z_near_boundaries = torch.tensor(
            [1.0 - eps, 1.0 - 10 * eps, -1.0 + eps, -1.0 + 10 * eps],
            dtype=torch.float64,
        )
        v = torch.tensor([2.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_near_boundaries
        )
        assert torch.all(torch.isfinite(result)), "NaN/Inf near boundaries"

        # Very small z values
        z_small = torch.tensor([1e-15, 1e-10, 1e-5], dtype=torch.float64)
        for n in [1, 2, 10, 100]:
            v_int = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v_int, z_small
            )
            expected = reference_chebyshev_t(v_int, z_small)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

        # Large z in hyperbolic domain
        z_large = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float64)
        v_half = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(
            v_half, z_large
        )
        expected = torch.cosh(v_half * torch.acosh(z_large))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

        # Large integer degree with z outside [-1, 1]
        z_outside = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v_large = torch.tensor([100.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(
            v_large, z_outside
        )
        # T_100(1.5) and T_100(2.0) should be finite and match polynomial
        assert torch.all(torch.isfinite(result)), (
            "NaN/Inf for large n outside domain"
        )

        # Verify gradient stability near boundaries
        z_grad = torch.tensor(
            [1.0 - 1e-8, -1.0 + 1e-8], dtype=torch.float64, requires_grad=True
        )
        v_grad = torch.tensor([2.5], dtype=torch.float64)
        y = torchscience.special_functions.chebyshev_polynomial_t(
            v_grad, z_grad
        )
        y.sum().backward()
        assert torch.all(torch.isfinite(z_grad.grad)), (
            "NaN/Inf gradient near boundaries"
        )

    def test_unrolling_boundary_transition(self):
        """Test the exact boundary where unrolling switches to general recurrence.

        The implementation unrolls degrees 0-7 explicitly and uses general
        recurrence for n >= 8. This tests the transition is seamless.
        """
        z = torch.linspace(-1, 1, 200, dtype=torch.float64)

        # Test degrees around the unrolling boundary
        boundary_degrees = [6, 7, 8, 9]

        for n in boundary_degrees:
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)

            # Should match exactly (within float64 precision)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-12,
                atol=1e-12,
                msg=f"Mismatch at degree {n} (unrolling boundary)",
            )

        # Test that T_7 and T_8 satisfy the recurrence T_8 = 2z*T_7 - T_6
        v6 = torch.tensor([6.0], dtype=torch.float64)
        v7 = torch.tensor([7.0], dtype=torch.float64)
        v8 = torch.tensor([8.0], dtype=torch.float64)

        t6 = torchscience.special_functions.chebyshev_polynomial_t(v6, z)
        t7 = torchscience.special_functions.chebyshev_polynomial_t(v7, z)
        t8 = torchscience.special_functions.chebyshev_polynomial_t(v8, z)

        recurrence_result = 2 * z * t7 - t6
        torch.testing.assert_close(
            t8,
            recurrence_result,
            rtol=1e-12,
            atol=1e-12,
            msg="Recurrence relation fails at unrolling boundary (n=8)",
        )

        # Also verify gradients work correctly at the boundary
        z_grad = torch.linspace(
            -0.99, 0.99, 50, dtype=torch.float64, requires_grad=True
        )
        for n in [7, 8]:
            v_n = torch.tensor([float(n)], dtype=torch.float64)

            def func(z):
                return torchscience.special_functions.chebyshev_polynomial_t(
                    v_n, z
                )

            assert torch.autograd.gradcheck(
                func, (z_grad.clone().requires_grad_(True),), eps=1e-6
            ), f"gradcheck failed at unrolling boundary n={n}"

    # =========================================================================
    # CUDA-specific tests
    # =========================================================================

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward_matches_cpu(self):
        """Test that CUDA forward pass matches CPU results."""
        z_cpu = torch.linspace(-0.99, 0.99, 100, dtype=torch.float64)
        v_cpu = torch.tensor(
            [0.0, 1.0, 2.0, 2.5, 5.0, 10.0], dtype=torch.float64
        )

        z_cuda = z_cpu.cuda()
        v_cuda = v_cpu.cuda()

        for i in range(len(v_cpu)):
            v_i_cpu = v_cpu[i : i + 1]
            v_i_cuda = v_cuda[i : i + 1]

            result_cpu = torchscience.special_functions.chebyshev_polynomial_t(
                v_i_cpu, z_cpu
            )
            result_cuda = (
                torchscience.special_functions.chebyshev_polynomial_t(
                    v_i_cuda, z_cuda
                )
            )

            torch.testing.assert_close(
                result_cuda.cpu(),
                result_cpu,
                rtol=1e-12,
                atol=1e-12,
                msg=f"CUDA/CPU mismatch for v={v_cpu[i].item()}",
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward_matches_cpu(self):
        """Test that CUDA backward pass matches CPU results."""
        z_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        v_cpu = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        z_cuda = z_cpu.detach().clone().cuda().requires_grad_(True)
        v_cuda = v_cpu.detach().clone().cuda().requires_grad_(True)

        # Forward
        result_cpu = torchscience.special_functions.chebyshev_polynomial_t(
            v_cpu, z_cpu
        )
        result_cuda = torchscience.special_functions.chebyshev_polynomial_t(
            v_cuda, z_cuda
        )

        # Backward
        result_cpu.sum().backward()
        result_cuda.sum().backward()

        torch.testing.assert_close(
            z_cuda.grad.cpu(), z_cpu.grad, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            v_cuda.grad.cpu(), v_cpu.grad, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_hyperbolic_domain(self):
        """Test CUDA correctness in hyperbolic domain (|z| > 1)."""
        z_cpu = torch.tensor([1.5, 2.0, -1.5, -2.0, -3.0], dtype=torch.float64)
        v_cpu = torch.tensor([0.5], dtype=torch.float64)

        z_cuda = z_cpu.cuda()
        v_cuda = v_cpu.cuda()

        result_cpu = torchscience.special_functions.chebyshev_polynomial_t(
            v_cpu, z_cpu
        )
        result_cuda = torchscience.special_functions.chebyshev_polynomial_t(
            v_cuda, z_cuda
        )

        torch.testing.assert_close(
            result_cuda.cpu(), result_cpu, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_complex_dtype(self):
        """Test CUDA with complex dtypes."""
        z = torch.tensor(
            [0.5 + 0.1j, -0.3 + 0.2j, 0.0 - 0.5j], dtype=torch.complex128
        )
        v = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)

        result_cpu = torchscience.special_functions.chebyshev_polynomial_t(
            v, z
        )
        result_cuda = torchscience.special_functions.chebyshev_polynomial_t(
            v.cuda(), z.cuda()
        )

        torch.testing.assert_close(
            result_cuda.cpu(), result_cpu, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cuda_low_precision(self, dtype):
        """Test CUDA with low-precision dtypes."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=dtype, device="cuda")
        v = torch.tensor([2.0], dtype=dtype, device="cuda")

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype
        assert result.device.type == "cuda"

        # Compare against float32
        expected = torchscience.special_functions.chebyshev_polynomial_t(
            v.float(), z.float()
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.float(), expected, rtol=rtol, atol=rtol
        )

    # =========================================================================
    # Integer v gradient tests
    # =========================================================================

    def test_integer_v_gradient_computed_via_analytic(self):
        """Test that gradient w.r.t. integer v is computed using analytic formula.

        Even when the forward pass uses recurrence for integer v, the backward
        pass should compute dT/dv = -sin(v * arccos(z)) * arccos(z).
        """
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        result.backward()

        # Expected gradient: dT/dv = -sin(v * arccos(z)) * arccos(z)
        theta = torch.acos(z)
        expected_grad = -torch.sin(v.detach() * theta) * theta

        torch.testing.assert_close(
            v.grad, expected_grad, rtol=1e-10, atol=1e-10
        )

    def test_integer_v_gradient_at_v_zero(self):
        """Test gradient at v=0 (T_0(z) = 1, dT/dv = 0 at v=0)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        result.backward()

        # At v=0: dT/dv = -sin(0) * arccos(z) = 0
        expected_grad = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(
            v.grad, expected_grad, rtol=1e-10, atol=1e-10
        )

    def test_integer_v_gradcheck(self):
        """Test gradient correctness for integer v values via gradcheck."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        for n in [1.0, 2.0, 3.0, 5.0, 10.0]:
            v = torch.tensor([n], dtype=torch.float64, requires_grad=True)

            def func(v):
                return torchscience.special_functions.chebyshev_polynomial_t(
                    v, z
                )

            assert torch.autograd.gradcheck(
                func, (v,), eps=1e-6, atol=1e-5, rtol=1e-5
            ), f"gradcheck failed for integer v={n}"

    # =========================================================================
    # Batched and broadcasting stress tests
    # =========================================================================

    def test_large_batch_dimensions(self):
        """Test with large batch dimensions."""
        batch_sizes = [(1000,), (100, 100), (10, 10, 10, 10)]

        for shape in batch_sizes:
            z = (
                torch.rand(shape, dtype=torch.float64) * 1.98 - 0.99
            )  # [-0.99, 0.99]
            v = torch.tensor([2.5], dtype=torch.float64)

            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)

            assert result.shape == shape
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_complex_broadcasting_patterns(self):
        """Test complex broadcasting patterns between v and z."""
        # v: (3, 1), z: (1, 5) -> output: (3, 5)
        v = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
        z = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]], dtype=torch.float64)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.shape == (3, 5)

        # Verify each element
        for i in range(3):
            for j in range(5):
                expected = reference_chebyshev_t(
                    v[i, 0:1], z[0, j : j + 1]
                ).item()
                torch.testing.assert_close(
                    result[i, j],
                    torch.tensor(expected, dtype=torch.float64),
                    rtol=1e-10,
                    atol=1e-10,
                )

        # v: (2, 3, 1), z: (4,) -> output: (2, 3, 4)
        v = torch.rand(2, 3, 1, dtype=torch.float64) * 5
        z = torch.rand(4, dtype=torch.float64) * 1.98 - 0.99

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.shape == (2, 3, 4)

    def test_broadcasting_with_gradients(self):
        """Test that gradients are correctly reduced in broadcasting."""
        v = torch.tensor(
            [[2.5]], dtype=torch.float64, requires_grad=True
        )  # (1, 1)
        # Create z as a leaf tensor with requires_grad
        z = torch.empty(3, 4, dtype=torch.float64).uniform_(-0.99, 0.99)
        z.requires_grad_(True)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.shape == (3, 4)

        loss = result.sum()
        loss.backward()

        # v.grad should be reduced to shape (1, 1)
        assert v.grad.shape == (1, 1)
        # z.grad should have same shape as z
        assert z.grad.shape == (3, 4)

    def test_scalar_like_tensors(self):
        """Test with 0-dimensional and 1-element tensors."""
        # 0-d tensors
        v_0d = torch.tensor(2.0, dtype=torch.float64)
        z_0d = torch.tensor(0.5, dtype=torch.float64)

        result = torchscience.special_functions.chebyshev_polynomial_t(
            v_0d, z_0d
        )
        expected = reference_chebyshev_t(
            v_0d.unsqueeze(0), z_0d.unsqueeze(0)
        ).squeeze()

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Missing edge case tests
    # =========================================================================

    def test_z_equals_zero_various_v(self):
        """Test T_v(0) for various v values.

        T_v(0) = cos(v * pi/2):
        - T_0(0) = cos(0) = 1
        - T_1(0) = cos(pi/2) = 0
        - T_2(0) = cos(pi) = -1
        - T_3(0) = cos(3*pi/2) = 0
        - T_4(0) = cos(2*pi) = 1
        - T_0.5(0) = cos(pi/4) = sqrt(2)/2
        - T_1.5(0) = cos(3*pi/4) = -sqrt(2)/2
        - T_2.5(0) = cos(5*pi/4) = -sqrt(2)/2
        """
        z = torch.tensor([0.0], dtype=torch.float64)

        test_cases = [
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, -1.0),
            (3.0, 0.0),
            (4.0, 1.0),
            (0.5, math.sqrt(2) / 2),  # cos(pi/4)
            (1.5, -math.sqrt(2) / 2),  # cos(3*pi/4)
            (2.5, -math.sqrt(2) / 2),  # cos(5*pi/4)
        ]

        for v_val, expected_val in test_cases:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"T_{v_val}(0) failed",
            )

    def test_half_integer_v_at_boundaries(self):
        """Test half-integer v at z = ±1 boundaries.

        At z = 1: T_v(1) = 1 for all v
        At z = -1: T_v(-1) = cos(v * pi)
        """
        half_integers = [0.5, 1.5, 2.5, 3.5, 4.5]

        # z = 1: T_v(1) = 1
        z_plus_1 = torch.tensor([1.0], dtype=torch.float64)
        for v_val in half_integers:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z_plus_1
            )
            expected = torch.tensor([1.0], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"T_{v_val}(1) failed",
            )

        # z = -1: T_v(-1) = cos(v * pi)
        z_minus_1 = torch.tensor([-1.0], dtype=torch.float64)
        for v_val in half_integers:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z_minus_1
            )
            expected = torch.tensor(
                [math.cos(v_val * math.pi)], dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"T_{v_val}(-1) failed",
            )

    def test_half_integer_v_hyperbolic_negative_z(self):
        """Test half-integer v with z < -1 (hyperbolic domain).

        T_v(z) = cos(v*pi) * cosh(v * acosh(-z)) for z < -1
        """
        z_vals = [-1.5, -2.0, -3.0, -5.0, -10.0]
        v_vals = [0.5, 1.5, 2.5, 3.5]

        for v_val in v_vals:
            v = torch.tensor([v_val], dtype=torch.float64)
            z = torch.tensor(z_vals, dtype=torch.float64)

            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = torch.cos(v * math.pi) * torch.cosh(v * torch.acosh(-z))

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"T_{v_val}(z<-1) failed",
            )

    def test_gradcheck_half_integer_v_hyperbolic(self):
        """Test gradient correctness for half-integer v in hyperbolic domain."""
        z = torch.tensor(
            [-1.5, -2.0, -3.0], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)

        def func(v, z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (v, z), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_quarter_integer_v_various_domains(self):
        """Test quarter-integer v values across all domains."""
        v_vals = [0.25, 0.75, 1.25, 1.75, 2.25]

        # Standard domain |z| <= 1
        z_standard = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        for v_val in v_vals:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z_standard
            )
            expected = reference_chebyshev_t(v, z_standard)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

        # Hyperbolic domain z > 1
        z_pos_hyp = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        for v_val in v_vals:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z_pos_hyp
            )
            expected = torch.cosh(v * torch.acosh(z_pos_hyp))
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

        # Hyperbolic domain z < -1
        z_neg_hyp = torch.tensor([-1.5, -2.0, -3.0], dtype=torch.float64)
        for v_val in v_vals:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z_neg_hyp
            )
            expected = torch.cos(v * math.pi) * torch.cosh(
                v * torch.acosh(-z_neg_hyp)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    # =========================================================================
    # Complex gradgradcheck tests
    # =========================================================================

    @pytest.mark.skip(reason="Complex dtype not implemented")
    def test_gradgradcheck_complex_relaxed_tolerance(self):
        """Test complex second-order derivatives with relaxed tolerances."""
        z = torch.tensor(
            [0.3 + 0.1j, 0.5 + 0.2j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        v = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        # First-order should pass with tight tolerances
        assert torch.autograd.gradcheck(func, (z,), eps=1e-6)

        # Second-order also passes with tight tolerances
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.skip(reason="Complex dtype not implemented")
    def test_gradgradcheck_complex_away_from_branch_cuts(self):
        """Test complex gradgradcheck away from branch cuts (z=±1)."""
        # Points well away from branch cuts at z=±1
        z = torch.tensor(
            [0.0 + 0.5j, 0.0 - 0.5j, 0.3 + 0.3j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        v = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.skip(
        reason="backward_backward kernel needs update for degree gradient"
    )
    def test_second_order_derivative_real_analytic(self):
        """Test second-order derivatives with real inputs (analytic case).

        Verifies that d²T/dz² is computed correctly using the formula:
        d²T/dz² = -v² * cos(vθ) / (1-z²) + v * z * sin(vθ) / (1-z²)^(3/2)
        where θ = arccos(z)
        """
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        # Compute d²T/dz² numerically via gradgradcheck
        assert torch.autograd.gradgradcheck(func, (z,), eps=1e-6)

        # Also verify the actual value
        y = func(z)
        (grad_z,) = torch.autograd.grad(y, z, create_graph=True)
        (grad_grad_z,) = torch.autograd.grad(grad_z, z)

        # Analytical formula
        theta = torch.acos(z.detach())
        one_minus_z2 = 1 - z.detach() ** 2
        v_det = v.detach()
        expected = (
            -(v_det**2) * torch.cos(v_det * theta) / one_minus_z2
            + v_det * z.detach() * torch.sin(v_det * theta) / one_minus_z2**1.5
        )

        torch.testing.assert_close(grad_grad_z, expected, rtol=1e-8, atol=1e-8)

    # =========================================================================
    # Sparse tensor tests (additional coverage beyond mixin tests)
    # =========================================================================

    def test_sparse_coo_various_degrees(self):
        """Test sparse COO with various polynomial degrees."""
        pytest.skip("Sparse not supported: T_v(0) != 0 in general")

    # =========================================================================
    # Quantized tensor tests (additional coverage beyond mixin tests)
    # =========================================================================

    def test_quantized_integer_degrees(self):
        """Test quantized tensors with integer polynomial degrees."""
        pytest.skip("Quantized not supported: output range varies with degree")

    def test_quantized_special_values(self):
        """Test quantized tensors at special values."""
        pytest.skip("Quantized not supported: output range varies with degree")
