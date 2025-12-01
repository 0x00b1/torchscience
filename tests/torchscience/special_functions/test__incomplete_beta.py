import math

import pytest
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
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


def scipy_incomplete_beta(z: float, a: float, b: float) -> float:
    """Reference implementation using SciPy's betainc."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required for this function")
    return float(scipy.special.betainc(a, b, z))


class TestIncompleteBeta(OpTestCase):
    """Tests for the regularized incomplete beta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="incomplete_beta",
            func=torchscience.special_functions.incomplete_beta,
            arity=3,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.01, 0.99),
                    supports_grad=True,
                    # For complex z, the standard algorithm requires |z| < 1
                    complex_magnitude_max=1.0,
                ),
                InputSpec(
                    name="a",
                    position=1,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="b",
                    position=2,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
            ],
            tolerances=ToleranceConfig(
                float32_rtol=1e-5,
                float32_atol=1e-5,
                float64_rtol=1e-5,
                float64_atol=1e-5,
                # Adaptive Gauss-Kronrod quadrature enables tighter tolerances
                # (improved from 1e-4 to 1e-5 with adaptive quadrature)
                gradcheck_rtol=1e-5,
                gradcheck_atol=1e-5,
                # Second derivatives use adaptive quadrature for doubly
                # log-weighted integrals, enabling tighter tolerances
                gradgradcheck_rtol=1e-5,
                gradgradcheck_atol=1e-5,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                # Complex derivatives are numerically challenging due to Wirtinger
                # derivative conventions, branch cuts, and finite difference
                # approximation for parameter derivatives (a, b)
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                # Mixed sparse/dense and mixed quantized/float tests are skipped
                # because the ternary operator macros require all inputs to have
                # the same layout (all sparse or all quantized)
                "test_sparse_coo_mixed_with_dense",
                "test_quantized_mixed_with_float",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 2.0, 3.0),
                    expected=0.0,
                    description="I_0(a, b) = 0",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0, 3.0),
                    expected=1.0,
                    description="I_1(a, b) = 1",
                ),
                SpecialValue(
                    inputs=(0.5, 1.0, 1.0),
                    expected=0.5,
                    description="I_z(1, 1) = z (uniform CDF)",
                ),
            ],
            singularities=[],
            supports_sparse_coo=True,
            supports_sparse_csr=True,
            supports_quantized=True,
            supports_meta=True,
        )

    # =========================================================================
    # Incomplete beta-specific tests
    # =========================================================================

    def test_boundary_value_zero(self):
        """Test I_0(a, b) = 0 for various a, b."""
        z = torch.tensor([0.0], dtype=torch.float64)
        test_cases = [
            (1.0, 1.0),
            (2.0, 3.0),
            (0.5, 0.5),
            (5.0, 2.0),
            (0.1, 10.0),
        ]
        for a_val, b_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.tensor([0.0], dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_boundary_value_one(self):
        """Test I_1(a, b) = 1 for various a, b."""
        z = torch.tensor([1.0], dtype=torch.float64)
        test_cases = [
            (1.0, 1.0),
            (2.0, 3.0),
            (0.5, 0.5),
            (5.0, 2.0),
            (0.1, 10.0),
        ]
        for a_val, b_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.tensor([1.0], dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_uniform_distribution_cdf(self):
        """Test I_z(1, 1) = z (uniform distribution CDF)."""
        z = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(z, a, b)
        torch.testing.assert_close(result, z, rtol=1e-10, atol=1e-10)

    def test_special_case_a_equals_1(self):
        """Test I_z(1, b) = 1 - (1-z)^b."""
        z = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        b_values = [0.5, 1.0, 2.0, 3.0, 5.0]
        a = torch.tensor([1.0], dtype=torch.float64)

        for b_val in b_values:
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = 1.0 - torch.pow(1.0 - z, b)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Failed for b={b_val}",
            )

    def test_special_case_b_equals_1(self):
        """Test I_z(a, 1) = z^a."""
        z = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        a_values = [0.5, 1.0, 2.0, 3.0, 5.0]
        b = torch.tensor([1.0], dtype=torch.float64)

        for a_val in a_values:
            a = torch.tensor([a_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.pow(z, a)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Failed for a={a_val}",
            )

    def test_symmetry_relation(self):
        """Test I_z(a, b) + I_{1-z}(b, a) = 1."""
        z = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
        test_cases = [
            (1.0, 2.0),
            (2.0, 3.0),
            (0.5, 0.5),
            (3.0, 1.5),
            (2.5, 4.0),
        ]

        for a_val, b_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            I_z = torchscience.special_functions.incomplete_beta(z, a, b)
            I_1_minus_z = torchscience.special_functions.incomplete_beta(
                1.0 - z, b, a
            )
            result = I_z + I_1_minus_z
            expected = torch.ones_like(z)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Symmetry failed for a={a_val}, b={b_val}",
            )

    def test_symmetric_beta_at_half(self):
        """Test I_{0.5}(a, a) = 0.5 for symmetric beta distribution."""
        z = torch.tensor([0.5], dtype=torch.float64)
        a_values = [0.5, 1.0, 2.0, 3.0, 5.0]

        for a_val in a_values:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([a_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.tensor([0.5], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Symmetric test failed for a=b={a_val}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_scipy_reference(self):
        """Test against SciPy's betainc function."""
        z_values = [0.1, 0.25, 0.5, 0.75, 0.9]
        a_values = [0.5, 1.0, 2.0, 3.0]
        b_values = [0.5, 1.0, 2.0, 3.0]

        for z_val in z_values:
            for a_val in a_values:
                for b_val in b_values:
                    z = torch.tensor([z_val], dtype=torch.float64)
                    a = torch.tensor([a_val], dtype=torch.float64)
                    b = torch.tensor([b_val], dtype=torch.float64)

                    result = torchscience.special_functions.incomplete_beta(
                        z, a, b
                    )
                    expected = torch.tensor(
                        [scipy_incomplete_beta(z_val, a_val, b_val)],
                        dtype=torch.float64,
                    )

                    torch.testing.assert_close(
                        result,
                        expected,
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"SciPy mismatch at z={z_val}, a={a_val}, b={b_val}",
                    )

    def test_monotonicity_in_z(self):
        """Test that I_z(a, b) is monotonically increasing in z."""
        z = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
        test_cases = [
            (1.0, 1.0),
            (2.0, 3.0),
            (0.5, 0.5),
            (5.0, 2.0),
        ]

        for a_val, b_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)

            # Check monotonicity
            differences = result[1:] - result[:-1]
            assert torch.all(differences >= -1e-10), (
                f"Non-monotonic for a={a_val}, b={b_val}"
            )

    def test_gradcheck_z(self):
        """Test gradient correctness for z."""
        z = torch.tensor(
            [0.2, 0.4, 0.6, 0.8], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_a(self):
        """Test gradient correctness for a (analytical gradient)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        a = torch.tensor(
            [1.5, 2.0, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Analytical gradients allow tighter tolerances than finite differences
        assert torch.autograd.gradcheck(
            func, (a,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_b(self):
        """Test gradient correctness for b (analytical gradient)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor(
            [1.5, 2.0, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Analytical gradients allow tighter tolerances than finite differences
        assert torch.autograd.gradcheck(
            func, (b,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_analytical(self):
        """Test that dI/dz = z^(a-1) * (1-z)^(b-1) / B(a,b)."""
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        result.sum().backward()

        # Analytical gradient: z^(a-1) * (1-z)^(b-1) / B(a,b)
        log_beta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        log_grad = (
            (a - 1) * torch.log(z.detach())
            + (b - 1) * torch.log(1 - z.detach())
            - log_beta
        )
        expected_grad = torch.exp(log_grad)

        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-4, atol=1e-4)

    def test_broadcasting(self):
        """Test broadcasting between z, a, and b."""
        # z: (3,), a: (1,), b: (1,)
        z = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        assert result.shape == (3,)

        # z: (1,), a: (3,), b: (1,)
        z = torch.tensor([0.5], dtype=torch.float64)
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        assert result.shape == (3,)

        # z: (2, 1), a: (1, 3), b: (1,) -> (2, 3)
        z = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        a = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        assert result.shape == (2, 3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype):
        """Test forward pass with low-precision dtypes."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
        a = torch.tensor([2.0], dtype=dtype)
        b = torch.tensor([3.0], dtype=dtype)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        assert result.dtype == dtype

        # Compare against float32 reference
        z_f32 = z.to(torch.float32)
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
        expected = torchscience.special_functions.incomplete_beta(
            z_f32, a_f32, b_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    def test_extreme_parameters(self):
        """Test with extreme values of a and b."""
        z = torch.tensor([0.5], dtype=torch.float64)

        # Very small a and b (close to 0)
        a_small = torch.tensor([0.1], dtype=torch.float64)
        b_small = torch.tensor([0.1], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z, a_small, b_small
        )
        assert torch.isfinite(result).all()

        # Large a and b
        a_large = torch.tensor([50.0], dtype=torch.float64)
        b_large = torch.tensor([50.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z, a_large, b_large
        )
        assert torch.isfinite(result).all()
        # For a=b=50, I_0.5 should be close to 0.5
        torch.testing.assert_close(
            result,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )

        # Asymmetric large parameters
        a_large = torch.tensor([100.0], dtype=torch.float64)
        b_small = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z, a_large, b_small
        )
        assert torch.isfinite(result).all()

    def test_very_large_parameters(self):
        """Test with very large parameters (a, b > 1000).

        For very large parameters, the beta distribution becomes highly
        concentrated around its mean μ = a/(a+b). The continued fraction
        algorithm's dynamic iteration scaling should handle these cases.
        """
        z = torch.tensor([0.5], dtype=torch.float64)

        # Symmetric very large parameters: a = b = 1000
        # Mean = 0.5, distribution is highly concentrated around 0.5
        a_large = torch.tensor([1000.0], dtype=torch.float64)
        b_large = torch.tensor([1000.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z, a_large, b_large
        )
        assert torch.isfinite(result).all()
        # For a=b, I_0.5 should be exactly 0.5 (by symmetry)
        torch.testing.assert_close(
            result,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
            msg="Symmetric very large params: I_0.5(1000,1000) should be 0.5",
        )

        # Test at different z values for very large symmetric parameters
        z_values = torch.tensor(
            [0.4, 0.45, 0.5, 0.55, 0.6], dtype=torch.float64
        )
        result = torchscience.special_functions.incomplete_beta(
            z_values, a_large, b_large
        )
        assert torch.isfinite(result).all()
        # Results should be strictly monotonic
        assert torch.all(result[1:] > result[:-1])
        # Due to concentration, values away from 0.5 should be close to 0 or 1
        assert result[0] < 0.01  # z=0.4 should give very small value
        assert result[-1] > 0.99  # z=0.6 should give very large value

        # Asymmetric very large parameters: a = 2000, b = 500
        # Mean = 2000/2500 = 0.8
        a_asymm = torch.tensor([2000.0], dtype=torch.float64)
        b_asymm = torch.tensor([500.0], dtype=torch.float64)
        z_at_mean = torch.tensor([0.8], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z_at_mean, a_asymm, b_asymm
        )
        assert torch.isfinite(result).all()
        # At the mean, I_z should be close to 0.5
        torch.testing.assert_close(
            result,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=0.1,
            atol=0.1,
            msg="At mean: I_0.8(2000,500) should be near 0.5",
        )

        # Extreme case: a = b = 5000
        a_extreme = torch.tensor([5000.0], dtype=torch.float64)
        b_extreme = torch.tensor([5000.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(
            z, a_extreme, b_extreme
        )
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_very_large_parameters_scipy_reference(self):
        """Test very large parameters against SciPy reference.

        SciPy's betainc is used as the reference implementation to verify
        correctness for large parameter values.
        """
        test_cases = [
            # (z, a, b, description)
            (0.5, 1000.0, 1000.0, "symmetric large"),
            (0.5, 500.0, 1500.0, "asymmetric large"),
            (0.75, 1500.0, 500.0, "asymmetric large at mean"),
            (0.5, 2000.0, 2000.0, "very large symmetric"),
            (0.3, 3000.0, 7000.0, "extreme asymmetric at mean"),
        ]

        for z_val, a_val, b_val, desc in test_cases:
            z = torch.tensor([z_val], dtype=torch.float64)
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)

            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.tensor(
                [scipy_incomplete_beta(z_val, a_val, b_val)],
                dtype=torch.float64,
            )

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Large params ({desc}): z={z_val}, a={a_val}, b={b_val}",
            )

    def test_z_near_boundaries(self):
        """Test with z very close to 0 and 1."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        # z close to 0
        z_near_0 = torch.tensor([1e-10, 1e-8, 1e-6, 1e-4], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(z_near_0, a, b)
        assert torch.all(torch.isfinite(result))
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

        # z close to 1
        z_near_1 = torch.tensor(
            [1 - 1e-10, 1 - 1e-8, 1 - 1e-6, 1 - 1e-4], dtype=torch.float64
        )
        result = torchscience.special_functions.incomplete_beta(z_near_1, a, b)
        assert torch.all(torch.isfinite(result))
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_invalid_parameters(self):
        """Test behavior with invalid parameters."""
        z = torch.tensor([0.5], dtype=torch.float64)

        # Negative a
        a_neg = torch.tensor([-1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(z, a_neg, b)
        assert torch.isnan(result).all()

        # Negative b
        a = torch.tensor([2.0], dtype=torch.float64)
        b_neg = torch.tensor([-1.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(z, a, b_neg)
        assert torch.isnan(result).all()

        # Zero a
        a_zero = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_beta(z, a_zero, b)
        assert torch.isnan(result).all()

    # =========================================================================
    # Sparse tensor tests
    # =========================================================================

    def test_sparse_coo_basic(self):
        """Test sparse COO tensor support with all-sparse inputs.

        The sparse implementation requires all inputs to be sparse COO tensors
        with the same sparsity pattern. This is efficient for element-wise
        operations where all operands share the same non-zero structure.
        """
        # Create sparse tensors with the same indices (sparsity pattern)
        indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

        # z values in valid range (0, 1)
        z_values = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
        # a values (positive shape parameter)
        a_values = torch.tensor([2.0, 2.5, 3.0, 1.5], dtype=torch.float64)
        # b values (positive shape parameter)
        b_values = torch.tensor([3.0, 2.0, 2.5, 4.0], dtype=torch.float64)

        size = (4, 4)
        z_sparse = torch.sparse_coo_tensor(indices, z_values, size)
        a_sparse = torch.sparse_coo_tensor(indices, a_values, size)
        b_sparse = torch.sparse_coo_tensor(indices, b_values, size)

        result = torchscience.special_functions.incomplete_beta(
            z_sparse, a_sparse, b_sparse
        )
        assert result.is_sparse

        # Compare against dense computation on the values
        expected_values = torchscience.special_functions.incomplete_beta(
            z_values, a_values, b_values
        )
        torch.testing.assert_close(
            result._values(), expected_values, rtol=1e-5, atol=1e-5
        )

    def test_sparse_coo_coalesced(self):
        """Test that sparse COO output preserves coalesced status."""
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        z_values = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        a_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        b_values = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)

        size = (3, 3)
        z_sparse = torch.sparse_coo_tensor(indices, z_values, size).coalesce()
        a_sparse = torch.sparse_coo_tensor(indices, a_values, size).coalesce()
        b_sparse = torch.sparse_coo_tensor(indices, b_values, size).coalesce()

        result = torchscience.special_functions.incomplete_beta(
            z_sparse, a_sparse, b_sparse
        )
        assert result.is_coalesced()

    def test_sparse_csr_basic(self):
        """Test sparse CSR tensor support with all-sparse inputs.

        The sparse CSR implementation requires all inputs to be sparse CSR
        tensors with the same sparsity pattern.
        """
        # Create a simple sparse CSR matrix (3x3 with 3 non-zeros on diagonal)
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([0, 1, 2])

        # z values in valid range (0, 1)
        z_values = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        # a values (positive shape parameter)
        a_values = torch.tensor([2.0, 2.5, 3.0], dtype=torch.float64)
        # b values (positive shape parameter)
        b_values = torch.tensor([3.0, 2.5, 2.0], dtype=torch.float64)

        size = (3, 3)
        z_sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, z_values, size, dtype=torch.float64
        )
        a_sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, a_values, size, dtype=torch.float64
        )
        b_sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, b_values, size, dtype=torch.float64
        )

        result = torchscience.special_functions.incomplete_beta(
            z_sparse, a_sparse, b_sparse
        )
        assert result.layout == torch.sparse_csr

        # Compare against dense computation on the values
        expected_values = torchscience.special_functions.incomplete_beta(
            z_values, a_values, b_values
        )
        torch.testing.assert_close(
            result.values(), expected_values, rtol=1e-5, atol=1e-5
        )

    # =========================================================================
    # Quantized tensor tests
    # =========================================================================

    def test_quantized_basic(self):
        """Test quantized tensor support with all-quantized inputs.

        The quantized implementation requires all inputs to be quantized tensors.
        Inputs are dequantized, the operation is performed, and the result is
        re-quantized using the first input's quantization parameters.
        """
        # Quantization parameters
        scale = 0.01
        zero_point = 128

        # Create float tensors first
        z = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
        a = torch.tensor([2.0, 2.5, 3.0], dtype=torch.float32)
        b = torch.tensor([3.0, 2.5, 2.0], dtype=torch.float32)

        # Quantize all inputs
        # Note: a and b need appropriate scale/zero_point for their value ranges
        qz = torch.quantize_per_tensor(z, scale, zero_point, torch.quint8)
        qa = torch.quantize_per_tensor(a, 0.05, 0, torch.quint8)
        qb = torch.quantize_per_tensor(b, 0.05, 0, torch.quint8)

        result = torchscience.special_functions.incomplete_beta(qz, qa, qb)
        assert result.is_quantized

        # Compare with non-quantized computation
        # Use dequantized values to account for quantization error in inputs
        expected = torchscience.special_functions.incomplete_beta(
            qz.dequantize(), qa.dequantize(), qb.dequantize()
        )
        # Allow larger tolerance due to quantization error accumulation
        torch.testing.assert_close(
            result.dequantize(), expected, rtol=0.15, atol=0.15
        )

    # =========================================================================
    # CUDA tests
    # =========================================================================

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward_matches_cpu(self):
        """Test that CUDA forward pass matches CPU results."""
        z = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        result_cpu = torchscience.special_functions.incomplete_beta(z, a, b)
        result_cuda = torchscience.special_functions.incomplete_beta(
            z.cuda(), a.cuda(), b.cuda()
        )

        torch.testing.assert_close(
            result_cuda.cpu(), result_cpu, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward_matches_cpu(self):
        """Test that CUDA backward pass matches CPU results."""
        z_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a_cpu = torch.tensor([2.0], dtype=torch.float64)
        b_cpu = torch.tensor([3.0], dtype=torch.float64)

        z_cuda = z_cpu.detach().clone().cuda().requires_grad_(True)
        a_cuda = a_cpu.cuda()
        b_cuda = b_cpu.cuda()

        # Forward
        result_cpu = torchscience.special_functions.incomplete_beta(
            z_cpu, a_cpu, b_cpu
        )
        result_cuda = torchscience.special_functions.incomplete_beta(
            z_cuda, a_cuda, b_cuda
        )

        # Backward
        result_cpu.sum().backward()
        result_cuda.sum().backward()

        torch.testing.assert_close(
            z_cuda.grad.cpu(), z_cpu.grad, rtol=1e-8, atol=1e-8
        )

    # =========================================================================
    # Edge case tests for adaptive quadrature
    # =========================================================================

    def test_gradient_small_a(self):
        """Test gradients when a is moderately small.

        Uses a=0.5 which tests the gradient computation with parameters
        approaching the singularity regime. For a < 1, the integrand
        t^(a-1) creates singularities at t=0 that require adaptive quadrature.
        """
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(z, a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z, a), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_small_b(self):
        """Test gradients when b is moderately small.

        Uses b=0.5 which tests the gradient computation with parameters
        approaching the singularity regime. For b < 1, the integrand
        (1-t)^(b-1) creates singularities at t=1 that require adaptive quadrature.
        """
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(z, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z, b), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_near_zero(self):
        """Test gradients when z is very close to 0."""
        z = torch.tensor(
            [1e-6, 1e-4, 1e-2], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-7, atol=1e-4, rtol=1e-4
        )

    def test_gradient_z_near_one(self):
        """Test gradients when z is very close to 1."""
        z = torch.tensor(
            [1 - 1e-2, 1 - 1e-4, 1 - 1e-6],
            dtype=torch.float64,
            requires_grad=True,
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-7, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_scipy_reference_small_parameters(self):
        """Test against SciPy with small a and b parameters."""
        z_values = [0.2, 0.5, 0.8]
        small_params = [0.05, 0.1, 0.2]

        for z_val in z_values:
            for param in small_params:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([param], dtype=torch.float64)
                b = torch.tensor([2.0], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, param, 2.0)],
                    dtype=torch.float64,
                )

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=f"Small a={param} mismatch at z={z_val}",
                )

    def test_gradgradcheck_small_a(self):
        """Test second derivatives with moderately small a.

        Uses a=0.5 to test second-order derivative computation with
        parameters approaching the singularity regime.
        """
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(z, a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradgradcheck(
            func, (z, a), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    # =========================================================================
    # Very small parameter tests (a, b < 0.05)
    # =========================================================================

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_very_small_a_forward(self):
        """Test forward pass with very small a (a < 0.05).

        For a < 1, the integrand t^(a-1) has a singularity at t=0.
        Very small a (approaching 0) creates strong singularities that
        stress the quadrature algorithms.
        """
        z_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        very_small_a = [0.01, 0.02, 0.03, 0.04, 0.05]

        for z_val in z_values:
            for a_val in very_small_a:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([a_val], dtype=torch.float64)
                b = torch.tensor([2.0], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, a_val, 2.0)],
                    dtype=torch.float64,
                )

                # Allow slightly looser tolerance for extreme parameters
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=f"Very small a={a_val} mismatch at z={z_val}",
                )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_very_small_b_forward(self):
        """Test forward pass with very small b (b < 0.05).

        For b < 1, the integrand (1-t)^(b-1) has a singularity at t=1.
        Very small b creates strong singularities near t=1 that require
        careful handling, especially when z is close to 1.
        """
        z_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        very_small_b = [0.01, 0.02, 0.03, 0.04, 0.05]

        for z_val in z_values:
            for b_val in very_small_b:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([2.0], dtype=torch.float64)
                b = torch.tensor([b_val], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, 2.0, b_val)],
                    dtype=torch.float64,
                )

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=f"Very small b={b_val} mismatch at z={z_val}",
                )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_very_small_both_forward(self):
        """Test forward pass with both a and b very small.

        When both parameters are very small, singularities exist at both
        t=0 and t=1. This is the most challenging case for numerical
        integration.
        """
        z_values = [0.2, 0.5, 0.8]
        very_small_params = [0.01, 0.03, 0.05]

        for z_val in z_values:
            for param in very_small_params:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([param], dtype=torch.float64)
                b = torch.tensor([param], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, param, param)],
                    dtype=torch.float64,
                )

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"Very small a=b={param} mismatch at z={z_val}",
                )

    def test_small_a_gradient(self):
        """Test gradient correctness with small a.

        The log-weighted integrals J_a become more challenging to compute
        accurately when a is small due to the stronger ln(t) singularity.
        Uses a=0.5 which is a moderately small value.
        """
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(z, a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z, a), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_very_small_b_gradient(self):
        """Test gradient correctness with very small b.

        The log-weighted integrals J_b become more challenging when b is
        very small due to the stronger ln(1-t) singularity near t=1.
        """
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.03], dtype=torch.float64, requires_grad=True)

        def func(z, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z, b), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_very_small_b_z_near_one(self):
        """Test with very small b and z close to 1.

        This is the most challenging case for the t=1 singularity:
        when b < 1, (1-t)^(b-1) -> infinity as t -> 1, and when z is
        close to 1, the integration domain includes this singularity.
        """
        z = torch.tensor([0.9, 0.95, 0.99], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)
        very_small_b = [0.01, 0.02, 0.05]

        for b_val in very_small_b:
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_beta(z, a, b)

            # Result should be finite and in [0, 1]
            assert torch.all(torch.isfinite(result)), (
                f"Non-finite result for b={b_val}, z={z.tolist()}"
            )
            assert torch.all(result >= 0) and torch.all(result <= 1), (
                f"Result out of bounds for b={b_val}"
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_very_small_b_z_near_one_scipy(self):
        """Test against SciPy with very small b and z close to 1."""
        z_values = [0.9, 0.95, 0.99]
        very_small_b = [0.01, 0.02, 0.05]

        for z_val in z_values:
            for b_val in very_small_b:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([2.0], dtype=torch.float64)
                b = torch.tensor([b_val], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, 2.0, b_val)],
                    dtype=torch.float64,
                )

                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"Small b={b_val}, z={z_val} mismatch",
                )

    def test_small_b_gradgradcheck(self):
        """Test second derivatives with small b parameter.

        The doubly log-weighted integrals K_aa, K_ab, K_bb are most
        challenging when b is small due to the t=1 singularity. The
        dual-region integration helps with this case.
        """
        z = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(z, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Looser tolerances for second derivatives with small b
        assert torch.autograd.gradgradcheck(
            func, (z, b), eps=1e-5, atol=5e-3, rtol=5e-3
        )

    # =========================================================================
    # Quadrature convergence stress tests
    # =========================================================================

    def test_quadrature_stress_simultaneous_singularities(self):
        """Stress test: both a < 1 and b < 1 with z spanning full range.

        This creates singularities at both t=0 and t=1, requiring careful
        handling by the dual-region integration strategy.
        """
        z_values = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64
        )

        stress_cases = [
            (0.5, 0.5),  # Arcsine distribution
            (0.3, 0.3),  # Stronger singularities at both ends
            (0.1, 0.9),  # Asymmetric: strong t=0, moderate t=1
            (0.9, 0.1),  # Asymmetric: moderate t=0, strong t=1
            (0.2, 0.8),  # Asymmetric singularities
        ]

        for a_val, b_val in stress_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)

            result = torchscience.special_functions.incomplete_beta(
                z_values, a, b
            )

            # All results should be finite
            assert torch.isfinite(result).all(), (
                f"Non-finite result for a={a_val}, b={b_val}"
            )

            # Results should be in [0, 1]
            assert torch.all(result >= 0) and torch.all(result <= 1), (
                f"Result out of bounds for a={a_val}, b={b_val}"
            )

            # Results should be monotonically increasing
            assert torch.all(result[1:] >= result[:-1] - 1e-10), (
                f"Non-monotonic for a={a_val}, b={b_val}"
            )

    def test_quadrature_stress_extreme_z_with_singularities(self):
        """Stress test: z very close to boundaries with parameter singularities.

        When z is near 0 and a < 1, or z is near 1 and b < 1, the quadrature
        must handle integration up to a strong singularity.
        """
        # z near 0 with small a (strong t=0 singularity in domain)
        z_near_0 = torch.tensor([1e-4, 1e-3, 1e-2, 0.05], dtype=torch.float64)
        a_small = torch.tensor([0.3], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(
            z_near_0, a_small, b
        )
        assert torch.isfinite(result).all()
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # z near 1 with small b (t=1 singularity almost reached)
        z_near_1 = torch.tensor([0.95, 0.97, 0.99, 0.999], dtype=torch.float64)
        a = torch.tensor([2.0], dtype=torch.float64)
        b_small = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(
            z_near_1, a, b_small
        )
        assert torch.isfinite(result).all()
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_quadrature_stress_gradient_difficult_cases(self):
        """Stress test: gradients in cases that stress the quadrature.

        The log-weighted integrals J_a and J_b have ln(t) and ln(1-t) factors
        that create additional singularities. These cases test the adaptive
        quadrature's ability to achieve gradient accuracy.
        """
        # Case 1: Both singularities active
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)

        def func(z, a, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Should pass gradcheck despite difficult integrand
        assert torch.autograd.gradcheck(
            func, (z, a, b), eps=1e-5, atol=1e-3, rtol=1e-3
        )

        # Case 2: z near boundary with active singularity
        z2 = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
        a2 = torch.tensor([2.0], dtype=torch.float64)
        b2 = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)

        def func2(z, b):
            return torchscience.special_functions.incomplete_beta(z, a2, b)

        assert torch.autograd.gradcheck(
            func2, (z2, b2), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_quadrature_stress_accuracy_difficult_parameters(self):
        """Stress test: verify accuracy against SciPy for difficult cases.

        These cases were selected to stress the adaptive quadrature:
        - Very small parameters (strong singularities)
        - Parameter asymmetry
        - z values that span the full integration domain
        """
        difficult_cases = [
            # (z, a, b, description)
            (0.5, 0.05, 0.05, "extreme symmetric singularities"),
            (0.9, 2.0, 0.05, "z near 1 with strong t=1 singularity"),
            (0.1, 0.05, 2.0, "z near 0 with strong t=0 singularity"),
            (0.5, 0.1, 10.0, "highly asymmetric"),
            (0.5, 10.0, 0.1, "highly asymmetric reverse"),
            (0.99, 2.0, 0.1, "z very near 1 with b < 1"),
            (0.01, 0.1, 2.0, "z very near 0 with a < 1"),
        ]

        for z_val, a_val, b_val, desc in difficult_cases:
            z = torch.tensor([z_val], dtype=torch.float64)
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)

            result = torchscience.special_functions.incomplete_beta(z, a, b)
            expected = torch.tensor(
                [scipy_incomplete_beta(z_val, a_val, b_val)],
                dtype=torch.float64,
            )

            # Allow slightly looser tolerance for these difficult cases
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Stress test failed ({desc}): z={z_val}, a={a_val}, b={b_val}",
            )

    def test_quadrature_many_evaluations_consistency(self):
        """Stress test: many evaluations to check for numerical drift.

        Running many evaluations helps detect any systematic numerical issues
        like accumulating errors or inconsistent behavior.
        """
        # Generate a grid of test points
        n_points = 100
        z_values = torch.linspace(0.01, 0.99, n_points, dtype=torch.float64)

        # Test with various parameter combinations
        param_sets = [
            (2.0, 3.0),
            (0.5, 0.5),
            (5.0, 1.0),
            (1.0, 5.0),
        ]

        for a_val, b_val in param_sets:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)

            results = torchscience.special_functions.incomplete_beta(
                z_values, a, b
            )

            # All results should be finite and in [0, 1]
            assert torch.isfinite(results).all()
            assert torch.all(results >= 0) and torch.all(results <= 1)

            # Results should be monotonically increasing (with small tolerance)
            diffs = results[1:] - results[:-1]
            assert torch.all(diffs >= -1e-10), (
                f"Non-monotonic results for a={a_val}, b={b_val}"
            )

            # Boundary behavior: results should approach 0 at z->0 and 1 at z->1
            assert results[0] < 0.1, (
                f"I_0.01 too large for a={a_val}, b={b_val}"
            )
            assert results[-1] > 0.9, (
                f"I_0.99 too small for a={a_val}, b={b_val}"
            )

    # =========================================================================
    # Complex input tests
    # =========================================================================

    def test_complex_on_real_axis_matches_real(self):
        """Complex values on real axis should match real implementation."""
        z_real = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)
        a_real = torch.tensor([2.0], dtype=torch.float64)
        a_complex = a_real.to(torch.complex128)
        b_real = torch.tensor([3.0], dtype=torch.float64)
        b_complex = b_real.to(torch.complex128)

        result_real = torchscience.special_functions.incomplete_beta(
            z_real, a_real, b_real
        )
        result_complex = torchscience.special_functions.incomplete_beta(
            z_complex, a_complex, b_complex
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-8, atol=1e-8
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_symmetry_relation(self):
        """Test I_z(a,b) + I_{1-z}(b,a) = 1 for complex inputs."""
        z_real = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        z = z_real.to(torch.complex128)
        a = torch.tensor([2.0], dtype=torch.complex128)
        b = torch.tensor([3.0], dtype=torch.complex128)

        I_z = torchscience.special_functions.incomplete_beta(z, a, b)
        I_1_minus_z = torchscience.special_functions.incomplete_beta(
            1.0 - z, b, a
        )

        result = I_z + I_1_minus_z
        expected = torch.ones_like(result)

        torch.testing.assert_close(
            result.real, expected.real, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            result.imag, torch.zeros_like(result.imag), rtol=1e-4, atol=1e-4
        )

    def test_complex_boundary_z_zero(self):
        """Test I_0(a, b) = 0 for complex inputs."""
        z = torch.tensor([0.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        expected = torch.tensor([0.0 + 0.0j], dtype=torch.complex128)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_boundary_z_one(self):
        """Test I_1(a, b) = 1 for complex inputs."""
        z = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        expected = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_with_small_imaginary_part(self):
        """Test complex z with small imaginary component."""
        z = torch.tensor(
            [0.3 + 0.01j, 0.5 + 0.01j, 0.7 + 0.01j], dtype=torch.complex128
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Result should be close to real values but with small imaginary part
        z_real = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        a_real = torch.tensor([2.0], dtype=torch.float64)
        b_real = torch.tensor([3.0], dtype=torch.float64)

        result_real = torchscience.special_functions.incomplete_beta(
            z_real, a_real, b_real
        )

        # Real part should be close to purely real result
        torch.testing.assert_close(
            result.real, result_real, rtol=0.01, atol=0.01
        )

    def test_complex_gradcheck_z(self):
        """Test gradient correctness for complex z using Wirtinger derivatives."""
        z = torch.tensor(
            [0.3 + 0.05j, 0.5 + 0.05j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradcheck uses Wirtinger derivatives
        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_gradcheck_a(self):
        """Test gradient correctness for complex a using Wirtinger derivatives."""
        z = torch.tensor([0.5 + 0.05j], dtype=torch.complex128)
        a = torch.tensor(
            [1.5 + 0.0j, 2.5 + 0.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)

        def func(a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradcheck uses Wirtinger derivatives
        assert torch.autograd.gradcheck(
            func, (a,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_gradcheck_b(self):
        """Test gradient correctness for complex b using Wirtinger derivatives."""
        z = torch.tensor([0.5 + 0.05j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor(
            [1.5 + 0.0j, 2.5 + 0.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )

        def func(b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradcheck uses Wirtinger derivatives
        assert torch.autograd.gradcheck(
            func, (b,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_z(self):
        """Test second-order gradient for complex z using Wirtinger derivatives."""
        z = torch.tensor(
            [0.4 + 0.03j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradgradcheck uses Wirtinger derivatives
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_a(self):
        """Test second-order gradient for complex a using Wirtinger derivatives."""
        z = torch.tensor([0.5 + 0.03j], dtype=torch.complex128)
        a = torch.tensor(
            [2.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradgradcheck uses Wirtinger derivatives
        assert torch.autograd.gradgradcheck(
            func, (a,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_b(self):
        """Test second-order gradient for complex b using Wirtinger derivatives."""
        z = torch.tensor([0.5 + 0.03j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor(
            [3.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )

        def func(b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradgradcheck uses Wirtinger derivatives
        assert torch.autograd.gradgradcheck(
            func, (b,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_all_inputs(self):
        """Test second-order gradient for all complex inputs simultaneously."""
        z = torch.tensor(
            [0.4 + 0.02j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor(
            [2.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor(
            [3.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )

        def func(z, a, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradgradcheck uses Wirtinger derivatives
        assert torch.autograd.gradgradcheck(
            func, (z, a, b), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    def test_complex64_forward(self):
        """Test complex64 dtype forward pass."""
        z = torch.tensor([0.3 + 0.01j, 0.5 + 0.01j], dtype=torch.complex64)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex64)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex64)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result).all()

    def test_complex_invalid_parameters(self):
        """Test complex inputs with invalid parameters."""
        z = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)

        # Re(a) <= 0
        a_invalid = torch.tensor([-1.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.incomplete_beta(
            z, a_invalid, b
        )
        assert torch.isnan(result.real).all()

        # Re(b) <= 0
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b_invalid = torch.tensor([-1.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.incomplete_beta(
            z, a, b_invalid
        )
        assert torch.isnan(result.real).all()

    def test_complex_outside_unit_disk(self):
        """Test complex z outside unit disk uses analytic continuation."""
        z_outside = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(
            z_outside, a, b
        )
        # Analytic continuation should return finite values
        assert torch.isfinite(result).all()

    # =========================================================================
    # Analytic continuation tests for |z| >= 1
    # =========================================================================

    def test_analytic_continuation_region_b_symmetry(self):
        """Test Region B: |z| >= 1 but |1-z| < 1 uses symmetry relation.

        When |z| >= 1 but |1-z| < 1, the implementation uses:
        I_z(a,b) = 1 - I_{1-z}(b,a)
        """
        # z = 1.5 has |z| = 1.5 > 1, but |1-z| = |-0.5| = 0.5 < 1
        z = torch.tensor([1.5 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Verify via symmetry: I_z(a,b) = 1 - I_{1-z}(b,a)
        one_minus_z = 1.0 - z  # = -0.5, which has |...| < 1
        I_1_minus_z = torchscience.special_functions.incomplete_beta(
            one_minus_z, b, a
        )
        expected = 1.0 - I_1_minus_z

        torch.testing.assert_close(
            result.real, expected.real, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            result.imag, expected.imag, rtol=1e-5, atol=1e-5
        )

    def test_analytic_continuation_region_c_hypergeometric(self):
        """Test Region C: |z| > 1 and |1-z| >= 1 uses hypergeometric continuation.

        When both |z| > 1 and |1-z| >= 1, the implementation uses the
        hypergeometric 2F1 linear transformation formula.
        """
        # z = 2.0 has |z| = 2 > 1 and |1-z| = |-1| = 1 >= 1
        z = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Result should be finite (not NaN)
        assert torch.isfinite(result).all()

    def test_analytic_continuation_special_case_a_equals_1(self):
        """Test I_z(1, b) = 1 - (1-z)^b holds for |z| > 1.

        This closed-form solution should hold in the extended domain.
        """
        z = torch.tensor([1.5 + 0.0j, 2.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        expected = 1.0 - torch.pow(1.0 - z, b)

        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg="I_z(1, b) = 1 - (1-z)^b should hold for |z| > 1",
        )

    def test_analytic_continuation_special_case_b_equals_1(self):
        """Test I_z(a, 1) = z^a holds for |z| > 1.

        This closed-form solution should hold in the extended domain.
        """
        z = torch.tensor([1.5 + 0.0j, 2.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)
        expected = torch.pow(z, a)

        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg="I_z(a, 1) = z^a should hold for |z| > 1",
        )

    def test_analytic_continuation_symmetry_extended(self):
        """Test I_z(a,b) + I_{1-z}(b,a) = 1 holds for |z| > 1."""
        # Test z values with |z| > 1
        z_values = [1.5 + 0.0j, 2.0 + 0.0j, 1.0 + 0.5j]

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.complex128)
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            I_z = torchscience.special_functions.incomplete_beta(z, a, b)
            I_1_minus_z = torchscience.special_functions.incomplete_beta(
                1.0 - z, b, a
            )

            result = I_z + I_1_minus_z
            expected = torch.ones_like(result)

            torch.testing.assert_close(
                result.real,
                expected.real,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Symmetry failed for z={z_val}",
            )

    def test_analytic_continuation_continuity_at_boundary(self):
        """Test continuity as |z| crosses the unit circle.

        The function should be continuous at |z| = 1 (except at branch cuts).
        """
        # Approach |z| = 1 from inside and outside along the real axis
        z_inside = torch.tensor([0.99 + 0.0j], dtype=torch.complex128)
        z_outside = torch.tensor([1.01 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result_inside = torchscience.special_functions.incomplete_beta(
            z_inside, a, b
        )
        result_outside = torchscience.special_functions.incomplete_beta(
            z_outside, a, b
        )

        # Results should be approximately continuous (within some tolerance)
        # Note: there may be discontinuities at branch cuts
        torch.testing.assert_close(
            result_inside.real,
            result_outside.real,
            rtol=0.2,
            atol=0.2,
            msg="Discontinuity detected at |z| = 1",
        )

    def test_analytic_continuation_gradcheck_region_b(self):
        """Test gradient correctness in Region B (|z| > 1, |1-z| < 1)."""
        # z = 1.5 is in Region B
        z = torch.tensor(
            [1.5 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-2, rtol=1e-2
        )

    def test_analytic_continuation_gradcheck_region_c(self):
        """Test gradient correctness in Region C (|z| > 1, |1-z| >= 1)."""
        # z = 2.5 is in Region C (|z| = 2.5, |1-z| = 1.5)
        z = torch.tensor(
            [2.5 + 0.1j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Use looser tolerances for Region C (uses finite differences internally)
        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    def test_analytic_continuation_complex_z_various_angles(self):
        """Test analytic continuation for complex z at various angles."""
        # z values outside unit disk at different angles
        angles = [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]
        radius = 1.5

        for theta in angles:
            z_val = radius * complex(math.cos(theta), math.sin(theta))
            z = torch.tensor([z_val], dtype=torch.complex128)
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            result = torchscience.special_functions.incomplete_beta(z, a, b)

            # Result should be finite
            assert torch.isfinite(result).all(), (
                f"Non-finite result at angle={theta}"
            )

    def test_analytic_continuation_integer_a_minus_b(self):
        """Test analytic continuation when a-b is an integer (limiting case).

        The hypergeometric linear transformation has a pole when a-b is an
        integer, requiring special handling via limiting forms.
        """
        # a - b = 1 (integer)
        z = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        a = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Should return finite result (handled by limiting form)
        assert torch.isfinite(result).all()

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_analytic_continuation_gradgradcheck_region_b(self):
        """Test second-order gradients in Region B."""
        z = torch.tensor(
            [1.5 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Region B uses symmetry, so gradients should be well-behaved
        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=5e-2, rtol=5e-2
        )

    # =========================================================================
    # Complex gradient tests with constrained domain
    # =========================================================================
    # These tests use carefully constrained inputs that stay within the valid
    # domain |z| < 1 for the incomplete beta function. The framework tests are
    # skipped because random complex inputs may fall outside this domain.

    def test_complex_gradcheck_constrained_z(self):
        """Test gradient correctness for complex z within constrained domain.

        Uses z values with |z| < 0.8 to stay well within the unit disk and
        avoid numerical issues near the boundary.
        """
        # Create z values with |z| < 0.8 (well within unit disk)
        z = torch.tensor(
            [0.3 + 0.1j, 0.4 - 0.1j, 0.2 + 0.2j, 0.5 + 0.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Complex gradcheck uses Wirtinger derivatives
        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_gradcheck_constrained_a(self):
        """Test gradient correctness for complex a with constrained z."""
        z = torch.tensor([0.4 + 0.05j], dtype=torch.complex128)
        # Use real-valued a (imaginary part 0) for better numerical stability
        a = torch.tensor(
            [1.5 + 0.0j, 2.0 + 0.0j, 2.5 + 0.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        b = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)

        def func(a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (a,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_gradcheck_constrained_b(self):
        """Test gradient correctness for complex b with constrained z."""
        z = torch.tensor([0.4 + 0.05j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        # Use real-valued b (imaginary part 0) for better numerical stability
        b = torch.tensor(
            [1.5 + 0.0j, 2.0 + 0.0j, 2.5 + 0.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )

        def func(b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (b,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_gradcheck_all_constrained(self):
        """Test gradient correctness for all complex inputs simultaneously."""
        z = torch.tensor(
            [0.3 + 0.05j, 0.5 + 0.03j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        a = torch.tensor(
            [2.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor(
            [3.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )

        def func(z, a, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z, a, b), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_constrained_z(self):
        """Test second-order gradient for complex z with constrained domain."""
        # Use a single z value to make gradgradcheck faster
        z = torch.tensor(
            [0.4 + 0.05j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_constrained_a(self):
        """Test second-order gradient for complex a with constrained z."""
        z = torch.tensor([0.4 + 0.03j], dtype=torch.complex128)
        a = torch.tensor(
            [2.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradgradcheck(
            func, (a,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_constrained_b(self):
        """Test second-order gradient for complex b with constrained z."""
        z = torch.tensor([0.4 + 0.03j], dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor(
            [3.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )

        def func(b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradgradcheck(
            func, (b,), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula"
    )
    def test_complex_gradgradcheck_all_constrained(self):
        """Test second-order gradient for all complex inputs with constrained domain."""
        z = torch.tensor(
            [0.35 + 0.03j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor(
            [2.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )
        b = torch.tensor(
            [3.0 + 0.0j], dtype=torch.complex128, requires_grad=True
        )

        def func(z, a, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradgradcheck(
            func, (z, a, b), eps=1e-5, atol=1e-2, rtol=1e-2
        )

    def test_complex_gradcheck_varying_imaginary_part(self):
        """Test gradients for z with varying imaginary parts.

        Tests that the Wirtinger derivative implementation works correctly
        for different amounts of imaginary component in z.
        """
        imaginary_parts = [0.01, 0.05, 0.1, 0.2]

        for imag in imaginary_parts:
            z = torch.tensor(
                [0.4 + imag * 1j], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Failed for imaginary part = {imag}"

    def test_complex_gradcheck_near_real_axis(self):
        """Test gradients for z very close to the real axis.

        When Im(z) is very small, the function should behave smoothly and
        gradients should be close to the real-valued case.
        """
        z_complex = torch.tensor(
            [0.3 + 1e-8j, 0.5 + 1e-8j, 0.7 + 1e-8j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z_complex,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_complex_gradcheck_different_parameter_regimes(self):
        """Test gradients across different parameter regimes for complex inputs.

        Tests various combinations of (a, b) to ensure gradients work
        across different shapes of the beta distribution.
        """
        param_cases = [
            (0.5, 0.5),  # U-shaped (arcsine-like)
            (1.0, 1.0),  # Uniform
            (2.0, 2.0),  # Symmetric bell
            (2.0, 5.0),  # Skewed left
            (5.0, 2.0),  # Skewed right
            (0.5, 2.0),  # J-shaped
        ]

        z = torch.tensor(
            [0.4 + 0.05j], dtype=torch.complex128, requires_grad=True
        )

        for a_val, b_val in param_cases:
            a = torch.tensor([a_val + 0.0j], dtype=torch.complex128)
            b = torch.tensor([b_val + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Failed for a={a_val}, b={b_val}"

    # =========================================================================
    # Asymptotic expansion tests for extreme z values
    # =========================================================================
    # These tests verify that the asymptotic expansions provide accurate results
    # for z very close to 0 or 1, where standard numerical methods may struggle.

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_asymptotic_z_near_zero(self):
        """Test accuracy of asymptotic expansion for z very close to 0.

        The asymptotic expansion I_z(a,b) ≈ z^a / (a·B(a,b)) should be accurate
        for z < 1e-8.
        """
        z_values = [1e-10, 1e-12, 1e-14, 1e-15]
        param_cases = [
            (2.0, 3.0),
            (0.5, 2.0),
            (5.0, 1.0),
            (1.5, 1.5),
        ]

        for z_val in z_values:
            for a_val, b_val in param_cases:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([a_val], dtype=torch.float64)
                b = torch.tensor([b_val], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, a_val, b_val)],
                    dtype=torch.float64,
                )

                # For extreme z, use relative tolerance scaled by the magnitude
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-4,
                    atol=1e-20,
                    msg=f"Asymptotic z near 0: z={z_val}, a={a_val}, b={b_val}",
                )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_asymptotic_z_near_one(self):
        """Test accuracy of asymptotic expansion for z very close to 1.

        Uses the symmetry I_z(a,b) = 1 - I_{1-z}(b,a) combined with the
        asymptotic expansion near zero.
        """
        z_values = [1 - 1e-10, 1 - 1e-12, 1 - 1e-14, 1 - 1e-15]
        param_cases = [
            (2.0, 3.0),
            (0.5, 2.0),
            (5.0, 1.0),
            (1.5, 1.5),
        ]

        for z_val in z_values:
            for a_val, b_val in param_cases:
                z = torch.tensor([z_val], dtype=torch.float64)
                a = torch.tensor([a_val], dtype=torch.float64)
                b = torch.tensor([b_val], dtype=torch.float64)

                result = torchscience.special_functions.incomplete_beta(
                    z, a, b
                )
                expected = torch.tensor(
                    [scipy_incomplete_beta(z_val, a_val, b_val)],
                    dtype=torch.float64,
                )

                # Result should be very close to 1.0
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-4,
                    atol=1e-10,
                    msg=f"Asymptotic z near 1: z={z_val}, a={a_val}, b={b_val}",
                )

    def test_asymptotic_gradient_z_near_zero(self):
        """Test gradient accuracy at z values where asymptotic expansion is used."""
        z = torch.tensor(
            [1e-10, 1e-12], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Asymptotic gradients should still pass gradcheck
        # Use larger eps for numerical differentiation at extreme values
        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-12, atol=1e-3, rtol=1e-3
        )

    def test_asymptotic_gradient_z_near_one(self):
        """Test gradient accuracy at z values very close to 1."""
        z = torch.tensor(
            [1 - 1e-10, 1 - 1e-12], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Asymptotic gradients should still pass gradcheck
        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-12, atol=1e-3, rtol=1e-3
        )

    def test_asymptotic_continuity_near_threshold(self):
        """Test that there's no discontinuity at the asymptotic threshold.

        The transition between asymptotic and standard computation should
        be smooth without jumps in function value.
        """
        # Test points around the asymptotic threshold (1e-8)
        z_around_threshold = torch.tensor(
            [5e-9, 8e-9, 1e-8, 1.2e-8, 2e-8, 5e-8], dtype=torch.float64
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(
            z_around_threshold, a, b
        )

        # Results should be monotonically increasing
        diffs = result[1:] - result[:-1]
        assert torch.all(diffs > 0), (
            "Non-monotonic behavior near asymptotic threshold"
        )

        # Check that the function is smooth (no large jumps)
        relative_jumps = diffs[1:] / diffs[:-1]
        # Relative change between consecutive differences should be reasonable
        # (not more than 10x change indicating a discontinuity)
        assert torch.all(relative_jumps < 10) and torch.all(
            relative_jumps > 0.1
        ), "Large discontinuity detected near asymptotic threshold"

    def test_asymptotic_with_small_parameters(self):
        """Test asymptotic expansion with small shape parameters.

        When a < 1, the integrand has a singularity at t=0. The asymptotic
        expansion should still be accurate.
        """
        z = torch.tensor([1e-10, 1e-12], dtype=torch.float64)
        small_a = torch.tensor([0.3], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        result = torchscience.special_functions.incomplete_beta(z, small_a, b)

        # Result should be finite and positive
        assert torch.isfinite(result).all()
        assert torch.all(result > 0)
        assert torch.all(result < 1)

    def test_asymptotic_complex_z_near_zero(self):
        """Test asymptotic expansion for complex z near 0."""
        z = torch.tensor(
            [1e-10 + 1e-11j, 1e-12 + 1e-13j], dtype=torch.complex128
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Result should be finite and have small magnitude (close to 0)
        assert torch.isfinite(result).all()
        assert torch.all(torch.abs(result) < 1e-5)

    # =========================================================================
    # Targeted complex input tests for |z| < 1 (analytical gradient domain)
    # =========================================================================
    # These tests specifically exercise the analytical gradient computation
    # for complex inputs within the unit disk, where the continued fraction
    # and adaptive quadrature methods apply directly without analytic
    # continuation fallbacks.

    def test_complex_analytical_domain_grid(self):
        """Test complex z on a grid within the unit disk.

        Creates a grid of complex z values with |z| < 1 and verifies
        that forward computation produces finite, sensible results.
        """
        # Create grid of complex values within unit disk
        real_parts = [0.1, 0.3, 0.5, 0.7]
        imag_parts = [-0.3, -0.1, 0.0, 0.1, 0.3]

        z_values = []
        for r in real_parts:
            for i in imag_parts:
                z_complex = complex(r, i)
                if abs(z_complex) < 0.95:  # Stay well within unit disk
                    z_values.append(z_complex)

        z = torch.tensor(z_values, dtype=torch.complex128)
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # All results should be finite
        assert torch.isfinite(result).all(), (
            "Non-finite results in analytical domain"
        )

    def test_complex_analytical_domain_radial_sweep(self):
        """Test complex z at various angles with fixed |z| < 1.

        Sweeps through different angles in the complex plane while
        keeping |z| = 0.5, testing the analytical gradients at various
        orientations.
        """
        angles = [
            0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2,
            2 * math.pi / 3,
            math.pi,
            4 * math.pi / 3,
            3 * math.pi / 2,
            5 * math.pi / 3,
        ]
        radius = 0.5  # Well within unit disk

        for theta in angles:
            z_val = radius * complex(math.cos(theta), math.sin(theta))
            z = torch.tensor(
                [z_val], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed at angle={theta}"

    def test_complex_analytical_domain_varying_radius(self):
        """Test complex z at various radii within the unit disk.

        Tests gradients at different distances from the origin,
        verifying that analytical gradients work across the entire
        valid domain.
        """
        radii = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
        angle = math.pi / 4  # Fixed angle

        for r in radii:
            z_val = r * complex(math.cos(angle), math.sin(angle))
            z = torch.tensor(
                [z_val], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed at radius={r}"

    def test_complex_analytical_domain_small_imaginary(self):
        """Test complex z with progressively smaller imaginary parts.

        As Im(z) → 0, the complex result should approach the real result
        and gradients should remain stable.
        """
        imag_parts = [0.2, 0.1, 0.05, 0.01, 0.001]
        real_part = 0.5

        for imag in imag_parts:
            z = torch.tensor(
                [complex(real_part, imag)],
                dtype=torch.complex128,
                requires_grad=True,
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for Im(z)={imag}"

    def test_complex_analytical_domain_negative_imaginary(self):
        """Test complex z with negative imaginary parts.

        Verifies that the implementation handles the lower half-plane
        correctly within |z| < 1.
        """
        z_values = [0.3 - 0.1j, 0.5 - 0.2j, 0.4 - 0.3j, 0.6 - 0.1j]

        for z_val in z_values:
            z = torch.tensor(
                [z_val], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for z={z_val}"

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_analytical_domain_all_params_gradcheck(self):
        """Test gradcheck for all three complex parameters simultaneously.

        Verifies that the analytical gradients work correctly when
        computing derivatives with respect to z, a, and b together.
        """
        test_cases = [
            (0.3 + 0.1j, 2.0 + 0.0j, 3.0 + 0.0j),
            (0.5 + 0.05j, 1.5 + 0.0j, 2.5 + 0.0j),
            (0.4 - 0.1j, 3.0 + 0.0j, 2.0 + 0.0j),
            (0.6 + 0.15j, 0.8 + 0.0j, 1.2 + 0.0j),
        ]

        for z_val, a_val, b_val in test_cases:
            z = torch.tensor(
                [z_val], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor(
                [a_val], dtype=torch.complex128, requires_grad=True
            )
            b = torch.tensor(
                [b_val], dtype=torch.complex128, requires_grad=True
            )

            def func(z, a, b):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z, a, b), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for z={z_val}, a={a_val}, b={b_val}"

    @pytest.mark.xfail(
        reason="Complex second-order gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_analytical_domain_gradgradcheck_grid(self):
        """Test second-order gradients on a grid within the unit disk.

        Verifies that the analytical second derivatives (using trigamma
        functions and doubly log-weighted integrals) work correctly.
        """
        test_points = [
            0.3 + 0.05j,
            0.5 + 0.1j,
            0.4 - 0.08j,
            0.6 + 0.03j,
        ]

        for z_val in test_points:
            z = torch.tensor(
                [z_val], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradgradcheck(
                func, (z,), eps=1e-5, atol=1e-2, rtol=1e-2
            ), f"Gradgradcheck failed for z={z_val}"

    def test_complex_analytical_domain_small_params(self):
        """Test complex z with small shape parameters (a, b < 1).

        When a < 1 or b < 1, the integrand has singularities. Tests
        that the analytical gradients handle these cases correctly
        for complex inputs.
        """
        small_param_cases = [
            (0.5, 2.0),  # Small a
            (2.0, 0.5),  # Small b
            (0.5, 0.5),  # Both small (arcsine-like)
            (0.3, 0.7),  # Both small, asymmetric
        ]

        z = torch.tensor(
            [0.4 + 0.1j], dtype=torch.complex128, requires_grad=True
        )

        for a_val, b_val in small_param_cases:
            a = torch.tensor([a_val + 0.0j], dtype=torch.complex128)
            b = torch.tensor([b_val + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for a={a_val}, b={b_val}"

    @pytest.mark.xfail(
        reason="Complex parameter gradients use simplified formula without log-weighted integrals"
    )
    def test_complex_analytical_domain_large_params(self):
        """Test complex z with large shape parameters.

        For large a and b, the continued fraction requires more
        iterations. Tests that analytical gradients work with the
        adaptive iteration scaling.
        """
        large_param_cases = [
            (10.0, 10.0),
            (50.0, 50.0),
            (20.0, 80.0),
        ]

        z = torch.tensor(
            [0.5 + 0.05j], dtype=torch.complex128, requires_grad=True
        )

        for a_val, b_val in large_param_cases:
            a = torch.tensor([a_val + 0.0j], dtype=torch.complex128)
            b = torch.tensor([b_val + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for large params a={a_val}, b={b_val}"

    def test_complex_analytical_domain_wirtinger_consistency(self):
        """Verify Wirtinger derivative convention consistency.

        The backward pass uses grad * conj(df/dz) convention. This test
        verifies that the gradient direction is consistent with this
        convention by checking specific cases.
        """
        z = torch.tensor(
            [0.5 + 0.1j], dtype=torch.complex128, requires_grad=True
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # Backward with unit gradient
        result.backward(torch.tensor([1.0 + 0.0j], dtype=torch.complex128))

        # Gradient should be finite
        assert torch.isfinite(z.grad).all()

        # For comparison: compute with real grad_output
        z2 = torch.tensor(
            [0.5 + 0.1j], dtype=torch.complex128, requires_grad=True
        )
        result2 = torchscience.special_functions.incomplete_beta(z2, a, b)
        result2.backward(torch.tensor([1.0 + 0.0j], dtype=torch.complex128))

        # Gradients should match (same input, same grad_output)
        torch.testing.assert_close(z.grad, z2.grad, rtol=1e-10, atol=1e-10)

    def test_complex_analytical_domain_conjugate_symmetry(self):
        """Test conjugate symmetry: I_{z*}(a*, b*) = I_z(a, b)*.

        For a holomorphic function f, we have f(z*) = f(z)* when the
        function has real coefficients. This tests that property.
        """
        z_val = 0.4 + 0.15j
        a_val = 2.0 + 0.0j
        b_val = 3.0 + 0.0j

        # Compute I_z(a, b)
        z = torch.tensor([z_val], dtype=torch.complex128)
        a = torch.tensor([a_val], dtype=torch.complex128)
        b = torch.tensor([b_val], dtype=torch.complex128)
        result1 = torchscience.special_functions.incomplete_beta(z, a, b)

        # Compute I_{z*}(a*, b*)
        z_conj = torch.tensor([z_val.conjugate()], dtype=torch.complex128)
        a_conj = torch.tensor([a_val.conjugate()], dtype=torch.complex128)
        b_conj = torch.tensor([b_val.conjugate()], dtype=torch.complex128)
        result2 = torchscience.special_functions.incomplete_beta(
            z_conj, a_conj, b_conj
        )

        # I_{z*}(a*, b*) should equal I_z(a, b)*
        torch.testing.assert_close(
            result2,
            result1.conj(),
            rtol=1e-8,
            atol=1e-8,
            msg="Conjugate symmetry violated",
        )

    def test_complex_analytical_domain_batch_gradcheck(self):
        """Test gradcheck with batched complex inputs.

        Verifies that gradients work correctly for batched operations
        within the analytical domain.
        """
        # Batch of z values, all with |z| < 1
        z = torch.tensor(
            [0.2 + 0.05j, 0.3 + 0.1j, 0.5 + 0.02j, 0.6 - 0.1j, 0.4 + 0.2j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_complex_analytical_domain_near_boundary(self):
        """Test complex z near but inside the unit circle.

        Tests gradients for |z| close to 1 (but < 1), which is the
        boundary between the analytical domain and analytic continuation.
        """
        # z values with |z| between 0.9 and 0.99
        near_boundary_z = [
            0.9 + 0.0j,
            0.85 + 0.3j,
            0.7 + 0.6j,
            0.0 + 0.95j,
            -0.5 + 0.8j,
        ]

        for z_val in near_boundary_z:
            if abs(z_val) < 1.0:  # Ensure |z| < 1
                z = torch.tensor(
                    [z_val], dtype=torch.complex128, requires_grad=True
                )
                a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
                b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

                def func(z):
                    return torchscience.special_functions.incomplete_beta(
                        z, a, b
                    )

                assert torch.autograd.gradcheck(
                    func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
                ), (
                    f"Gradcheck failed for z near boundary: z={z_val}, |z|={abs(z_val)}"
                )

    def test_complex_analytical_domain_purely_imaginary_z(self):
        """Test complex z with zero real part (purely imaginary).

        Tests gradients for z on the imaginary axis within |z| < 1.
        """
        imag_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for imag in imag_values:
            z = torch.tensor(
                [complex(0, imag)], dtype=torch.complex128, requires_grad=True
            )
            a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
            b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

            def func(z):
                return torchscience.special_functions.incomplete_beta(z, a, b)

            assert torch.autograd.gradcheck(
                func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
            ), f"Gradcheck failed for purely imaginary z={imag}j"

    def test_complex_analytical_domain_complex_params(self):
        """Test with complex parameters a and b (not just real + 0j).

        Verifies gradients when a and b have small imaginary parts,
        which exercises more of the complex arithmetic paths.
        """
        z = torch.tensor(
            [0.4 + 0.1j], dtype=torch.complex128, requires_grad=True
        )

        # Parameters with small imaginary parts
        complex_param_cases = [
            (2.0 + 0.1j, 3.0 + 0.0j),
            (2.0 + 0.0j, 3.0 + 0.1j),
            (2.0 + 0.05j, 3.0 + 0.05j),
            (1.5 - 0.1j, 2.5 + 0.1j),
        ]

        for a_val, b_val in complex_param_cases:
            # Need positive real parts for valid domain
            if a_val.real > 0 and b_val.real > 0:
                a = torch.tensor([a_val], dtype=torch.complex128)
                b = torch.tensor([b_val], dtype=torch.complex128)

                def func(z):
                    return torchscience.special_functions.incomplete_beta(
                        z, a, b
                    )

                assert torch.autograd.gradcheck(
                    func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3
                ), f"Gradcheck failed for complex params a={a_val}, b={b_val}"

    def test_complex_analytical_domain_stress_many_points(self):
        """Stress test: many random points within the unit disk.

        Generates many random complex z values with |z| < 0.9 and
        verifies forward computation produces sensible results.
        """
        torch.manual_seed(42)

        # Generate random points in the unit disk
        n_points = 100
        angles = torch.rand(n_points) * 2 * math.pi
        radii = torch.rand(n_points) * 0.9  # |z| < 0.9

        z_real = radii * torch.cos(angles)
        z_imag = radii * torch.sin(angles)
        z = torch.complex(z_real, z_imag).to(torch.complex128)

        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        result = torchscience.special_functions.incomplete_beta(z, a, b)

        # All results should be finite
        assert torch.isfinite(result).all(), (
            f"Non-finite results: {torch.sum(~torch.isfinite(result))} out of {n_points}"
        )

    def test_complex_analytical_domain_gradient_continuity(self):
        """Test that gradients vary continuously with z.

        Computes gradients at nearby points and verifies they are
        close to each other (no discontinuities).
        """
        base_z = 0.5 + 0.1j
        perturbations = [1e-4, 1e-4j, -1e-4, -1e-4j]

        a = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        b = torch.tensor([3.0 + 0.0j], dtype=torch.complex128)

        # Compute gradient at base point
        z_base = torch.tensor(
            [base_z], dtype=torch.complex128, requires_grad=True
        )
        result_base = torchscience.special_functions.incomplete_beta(
            z_base, a, b
        )
        result_base.backward(
            torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
        )
        grad_base = z_base.grad.clone()

        # Compute gradients at perturbed points
        for pert in perturbations:
            z_pert = torch.tensor(
                [base_z + pert], dtype=torch.complex128, requires_grad=True
            )
            result_pert = torchscience.special_functions.incomplete_beta(
                z_pert, a, b
            )
            result_pert.backward(
                torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
            )
            grad_pert = z_pert.grad

            # Gradients should be close (within 1% for small perturbation)
            relative_diff = torch.abs(grad_pert - grad_base) / torch.abs(
                grad_base
            )
            assert relative_diff < 0.01, (
                f"Gradient discontinuity at z={base_z}+{pert}: rel_diff={relative_diff.item()}"
            )

    # =========================================================================
    # Small parameter gradient tests (Improvement 3)
    # =========================================================================
    # Note: Parameter derivatives dI/da and dI/db for very small parameters
    # (a, b < 0.05) involve near-cancellation of large terms:
    #   dI/da = J_a/B - I_z * [psi(a) - psi(a+b)]
    # When a is very small, both J_a/B and I_z * [psi(a) - psi(a+b)] are large
    # (~30-40) and nearly cancel to a small value (~0.2). This makes accurate
    # computation of parameter gradients challenging for extreme parameters.
    # The gradient w.r.t. z remains accurate even for small parameters.

    def test_small_a_gradcheck_z_only(self):
        """Test gradient correctness w.r.t. z with small a parameter.

        The gradient dI/dz = z^(a-1) * (1-z)^(b-1) / B(a,b) is analytically
        computed and remains accurate even for small a.
        """
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([0.1], dtype=torch.float64)  # Small but not extreme
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_small_b_gradcheck_z_only(self):
        """Test gradient correctness w.r.t. z with small b parameter."""
        z = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.1], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_moderately_small_params_gradcheck(self):
        """Test gradient correctness with moderately small parameters (0.2 < a,b < 0.5).

        This tests the improved dual-region integration and adaptive tolerances
        for parameters that are challenging but not extreme.
        """
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        def func(z, a, b):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Should pass with reasonable tolerances due to improved quadrature
        assert torch.autograd.gradcheck(
            func, (z, a, b), eps=1e-5, atol=5e-3, rtol=5e-3
        )

    def test_small_params_gradgradcheck(self):
        """Test second-order gradient correctness with moderately small parameters."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.float64)

        def func(z, a):
            return torchscience.special_functions.incomplete_beta(z, a, b)

        # Second-order gradients are more challenging; use relaxed tolerances
        assert torch.autograd.gradgradcheck(
            func, (z, a), eps=1e-5, atol=5e-2, rtol=5e-2
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_small_params_scipy_reference_z_gradient(self):
        """Verify z-gradients against numerical differences from SciPy.

        The dI/dz gradient is analytically computed and should be accurate
        even for small parameters.
        """
        test_cases = [
            # (z, a, b)
            (0.5, 0.1, 2.0),
            (0.5, 2.0, 0.1),
            (0.5, 0.2, 0.2),
            (0.3, 0.1, 3.0),
            (0.7, 3.0, 0.1),
        ]

        eps = 1e-7
        for z_val, a_val, b_val in test_cases:
            z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)

            result = torchscience.special_functions.incomplete_beta(z, a, b)
            result.backward()
            our_grad_z = z.grad.item()

            # Compute numerical gradient from SciPy
            scipy_plus = scipy_incomplete_beta(z_val + eps, a_val, b_val)
            scipy_minus = scipy_incomplete_beta(z_val - eps, a_val, b_val)
            scipy_grad_z = (scipy_plus - scipy_minus) / (2 * eps)

            rel_error = abs(our_grad_z - scipy_grad_z) / max(
                abs(scipy_grad_z), 1e-10
            )
            assert rel_error < 1e-4, (
                f"z-gradient mismatch at z={z_val}, a={a_val}, b={b_val}: "
                f"ours={our_grad_z}, scipy={scipy_grad_z}, rel_err={rel_error}"
            )
