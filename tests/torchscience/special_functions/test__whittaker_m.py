import pytest
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    ToleranceConfig,
)

# Optional mpmath import for reference tests
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def _transformation_identity(func):
    """Check M_kappa,mu(z) = e^(-z/2) * z^(mu+1/2) * M(a, b, z)."""
    kappa = torch.tensor([0.5], dtype=torch.float64)
    mu = torch.tensor([0.75], dtype=torch.float64)
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    # Whittaker M
    left = func(kappa, mu, z)

    # Transform parameters: a = mu - kappa + 1/2, b = 2*mu + 1
    a = mu - kappa + 0.5
    b = 2 * mu + 1

    # Right side: e^(-z/2) * z^(mu+1/2) * M(a, b, z)
    right = (
        torch.exp(-z / 2)
        * torch.pow(z, mu + 0.5)
        * torchscience.special_functions.confluent_hypergeometric_m(a, b, z)
    )
    return left, right


class TestWhittakerM(OpTestCase):
    """Tests for the Whittaker M function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="whittaker_m",
            func=torchscience.special_functions.whittaker_m,
            arity=3,
            input_specs=[
                InputSpec(
                    name="kappa",
                    position=0,
                    default_real_range=(0.1, 3.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="mu",
                    position=1,
                    default_real_range=(0.1, 3.0),
                    excluded_values={-0.5, -1.0, -1.5, -2.0, -2.5},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=2,
                    default_real_range=(0.1, 5.0),
                    supports_grad=True,
                ),
            ],
            tolerances=ToleranceConfig(
                float32_rtol=1e-4,
                float32_atol=1e-4,
                float64_rtol=1e-8,
                float64_atol=1e-8,
                gradcheck_rtol=1e-4,
                gradcheck_atol=1e-4,
                gradgradcheck_rtol=1e-3,
                gradgradcheck_atol=1e-3,
            ),
            skip_tests={
                "test_gradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                "test_quantized_basic",
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_low_precision_forward",
            },
            functional_identities=[
                IdentitySpec(
                    name="transformation_identity",
                    identity_fn=_transformation_identity,
                    description="M_kappa,mu(z) = e^(-z/2) * z^(mu+1/2) * M(a, b, z)",
                    rtol=1e-5,
                    atol=1e-5,
                ),
            ],
            special_values=[],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_transformation_identity(self):
        """Test M_kappa,mu(z) = e^(-z/2) * z^(mu+1/2) * M(a, b, z)."""
        kappa = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        mu = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Whittaker M
        result = self.descriptor.func(kappa, mu, z)

        # Transform parameters: a = mu - kappa + 1/2, b = 2*mu + 1
        a = mu - kappa + 0.5
        b = 2 * mu + 1

        # Expected: e^(-z/2) * z^(mu+1/2) * M(a, b, z)
        expected = (
            torch.exp(-z / 2)
            * torch.pow(z, mu + 0.5)
            * torchscience.special_functions.confluent_hypergeometric_m(
                a, b, z
            )
        )

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_large_z_behavior(self):
        """Test asymptotic behavior for large z.

        For large z, M_kappa,mu(z) ~ z^kappa * e^(z/2) / Gamma(mu - kappa + 1/2)
        We verify that the function grows approximately as expected.
        """
        kappa = torch.tensor([0.5], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)
        z_values = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float64)

        results = []
        for z in z_values:
            result = self.descriptor.func(kappa, mu, z.unsqueeze(0))
            results.append(result.item())

        # Check that the function is increasing for large positive z
        # (since it grows exponentially)
        for i in range(len(results) - 1):
            assert results[i + 1] > results[i], (
                f"Expected M to grow for large z, but "
                f"M(z={z_values[i + 1].item()}) = {results[i + 1]} <= "
                f"M(z={z_values[i].item()}) = {results[i]}"
            )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_reference(self):
        """Test against mpmath reference implementation."""
        test_cases = [
            (0.5, 1.0, 2.0),
            (1.0, 0.5, 1.0),
            (0.25, 0.75, 3.0),
            (1.5, 1.25, 2.5),
        ]
        for kappa_val, mu_val, z_val in test_cases:
            expected = complex(mpmath.whitm(kappa_val, mu_val, z_val)).real
            kappa = torch.tensor([kappa_val], dtype=torch.float64)
            mu = torch.tensor([mu_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(kappa, mu, z)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_positive_z_required(self):
        """Test that the function works correctly for z > 0."""
        kappa = torch.tensor([0.5], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)

        result = self.descriptor.func(kappa, mu, z)

        assert torch.isfinite(result).all()

    def test_symmetric_mu(self):
        """Test M_kappa,mu(z) = M_kappa,-mu(z) symmetry property."""
        kappa = torch.tensor([0.5], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result_positive = self.descriptor.func(kappa, mu, z)
        result_negative = self.descriptor.func(kappa, -mu, z)

        torch.testing.assert_close(
            result_positive, result_negative, rtol=1e-6, atol=1e-6
        )
