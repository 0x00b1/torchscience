import math

import mpmath
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


class TestIncompleteLegendreEllipticIntegralPi(OpTestCase):
    """Tests for the incomplete elliptic integral of the third kind Pi(n, phi, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="incomplete_legendre_elliptic_integral_pi",
            func=torchscience.special_functions.incomplete_legendre_elliptic_integral_pi,
            arity=3,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(-2.0, 0.0),
                ),
                InputSpec(
                    name="phi",
                    position=1,
                    default_real_range=(0.1, 1.5),
                ),
                InputSpec(
                    name="m",
                    position=2,
                    default_real_range=(0.05, 0.95),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_vmap_over_batch",
                "test_compile_smoke",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.5, 0.0, 0.5),
                    expected=0.0,
                    description="Pi(n, 0, m) = 0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.ellippi(n, phi, m) -- only for n=0 due to convention difference."""
        test_cases = [
            (0.0, 0.5, 0.5),
            (0.0, 1.0, 0.3),
            (0.0, 0.3, 0.7),
        ]
        for n_val, phi_val, m_val in test_cases:
            n = torch.tensor([n_val], dtype=torch.float64)
            phi = torch.tensor([phi_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_pi(
                n, phi, m
            )
            expected = float(mpmath.ellippi(n_val, phi_val, m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-8,
            )

    def test_n_zero_equals_F(self):
        """Pi(0, phi, m) = F(phi, m)."""
        phi = torch.tensor([0.3, 0.5, 1.0, 1.3], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        n = torch.zeros_like(phi)
        pi_result = torchscience.special_functions.incomplete_legendre_elliptic_integral_pi(
            n, phi, m
        )
        f_result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        torch.testing.assert_close(pi_result, f_result, rtol=1e-8, atol=1e-10)

    def test_reduces_to_complete_at_pi_over_2(self):
        """Pi(n, pi/2, m) = Pi(n, m)."""
        phi = torch.tensor([math.pi / 2], dtype=torch.float64)
        n = torch.tensor([-0.5], dtype=torch.float64)
        m_vals = [0.1, 0.3, 0.5, 0.7]
        for m_val in m_vals:
            m = torch.tensor([m_val], dtype=torch.float64)
            incomplete = torchscience.special_functions.incomplete_legendre_elliptic_integral_pi(
                n, phi, m
            )
            complete = torchscience.special_functions.complete_legendre_elliptic_integral_pi(
                n, m
            )
            torch.testing.assert_close(
                incomplete, complete, rtol=1e-6, atol=1e-8
            )
