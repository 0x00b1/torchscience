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
    SpecialValue,
    ToleranceConfig,
)


def _odd_function(func):
    """Check E(-phi, m) = -E(phi, m)."""
    phi = torch.tensor([0.1, 0.5, 1.0, 1.3], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    left = func(-phi, m)
    right = -func(phi, m)
    return left, right


class TestIncompleteLegendreEllipticIntegralE(OpTestCase):
    """Tests for the incomplete elliptic integral of the second kind E(phi, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="incomplete_legendre_elliptic_integral_e",
            func=torchscience.special_functions.incomplete_legendre_elliptic_integral_e,
            arity=2,
            input_specs=[
                InputSpec(
                    name="phi",
                    position=0,
                    default_real_range=(0.1, 1.5),
                ),
                InputSpec(
                    name="m",
                    position=1,
                    default_real_range=(0.05, 0.95),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="E(0, m) = 0",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="odd_function",
                    identity_fn=_odd_function,
                    description="E(-phi, m) = -E(phi, m)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_scipy(self):
        """Compare against scipy.special.ellipeinc."""
        phi_vals = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4]
        m_val = 0.5
        phi = torch.tensor(phi_vals, dtype=torch.float64)
        m = torch.tensor([m_val], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_e(
            phi, m
        )
        expected = torch.tensor(
            [scipy.special.ellipeinc(p, m_val) for p in phi_vals],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_m_zero_equals_phi(self):
        """E(phi, 0) = phi."""
        phi = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_e(
            phi, m
        )
        torch.testing.assert_close(result, phi, rtol=1e-10, atol=1e-10)

    def test_reduces_to_complete_at_pi_over_2(self):
        """E(pi/2, m) = E(m) (complete integral)."""
        phi = torch.tensor([math.pi / 2], dtype=torch.float64)
        m_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        for m_val in m_vals:
            m = torch.tensor([m_val], dtype=torch.float64)
            incomplete = torchscience.special_functions.incomplete_legendre_elliptic_integral_e(
                phi, m
            )
            complete = torchscience.special_functions.complete_legendre_elliptic_integral_e(
                m
            )
            torch.testing.assert_close(
                incomplete, complete, rtol=1e-6, atol=1e-8
            )
