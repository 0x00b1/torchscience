import math

import scipy.special
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SingularitySpec,
    SpecialValue,
    ToleranceConfig,
)


def _legendre_relation(func):
    """Check E(m)*K(1-m) + E(1-m)*K(m) - K(m)*K(1-m) = pi/2."""
    m = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.7, 0.8], dtype=torch.float64)
    K_m = func(m)
    K_1m = func(1.0 - m)
    E_m = torchscience.special_functions.complete_legendre_elliptic_integral_e(
        m
    )
    E_1m = (
        torchscience.special_functions.complete_legendre_elliptic_integral_e(
            1.0 - m
        )
    )
    left = E_m * K_1m + E_1m * K_m - K_m * K_1m
    right = torch.full_like(left, math.pi / 2)
    return left, right


class TestCompleteLegendreEllipticIntegralK(OpTestCase):
    """Tests for the complete elliptic integral of the first kind K(m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="complete_legendre_elliptic_integral_k",
            func=torchscience.special_functions.complete_legendre_elliptic_integral_k,
            arity=1,
            input_specs=[
                InputSpec(
                    name="m",
                    position=0,
                    default_real_range=(0.05, 0.85),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                "test_gradgradcheck_complex",
                "test_pole_behavior",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=math.pi / 2,
                    description="K(0) = pi/2",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="legendre_relation",
                    identity_fn=_legendre_relation,
                    rtol=1e-8,
                    atol=1e-8,
                    description="E(m)K'(m) + E'(m)K(m) - K(m)K'(m) = pi/2",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: iter([1.0]),
                    expected_behavior="inf",
                    description="Logarithmic singularity at m=1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_scipy(self):
        """Compare against scipy.special.ellipk."""
        m_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        m = torch.tensor(m_vals, dtype=torch.float64)
        result = torchscience.special_functions.complete_legendre_elliptic_integral_k(
            m
        )
        expected = torch.tensor(
            [scipy.special.ellipk(v) for v in m_vals], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_relation_to_carlson_rf(self):
        """K(m) = R_F(0, 1-m, 1)."""
        m = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        K = torchscience.special_functions.complete_legendre_elliptic_integral_k(
            m
        )
        zero = torch.zeros_like(m)
        one = torch.ones_like(m)
        rf = torchscience.special_functions.carlson_elliptic_integral_r_f(
            zero, 1.0 - m, one
        )
        torch.testing.assert_close(K, rf, rtol=1e-8, atol=1e-10)
