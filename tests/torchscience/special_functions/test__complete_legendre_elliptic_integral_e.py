import math

import scipy.special
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


class TestCompleteLegendreEllipticIntegralE(OpTestCase):
    """Tests for the complete elliptic integral of the second kind E(m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="complete_legendre_elliptic_integral_e",
            func=torchscience.special_functions.complete_legendre_elliptic_integral_e,
            arity=1,
            input_specs=[
                InputSpec(
                    name="m",
                    position=0,
                    default_real_range=(0.05, 0.95),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=math.pi / 2,
                    description="E(0) = pi/2",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_scipy(self):
        """Compare against scipy.special.ellipe."""
        m_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        m = torch.tensor(m_vals, dtype=torch.float64)
        result = torchscience.special_functions.complete_legendre_elliptic_integral_e(
            m
        )
        expected = torch.tensor(
            [scipy.special.ellipe(v) for v in m_vals], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_relation_to_carlson(self):
        """E(m) = R_F(0, 1-m, 1) - (m/3)*R_D(0, 1-m, 1)."""
        m = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        E = torchscience.special_functions.complete_legendre_elliptic_integral_e(
            m
        )
        zero = torch.zeros_like(m)
        one = torch.ones_like(m)
        rf = torchscience.special_functions.carlson_elliptic_integral_r_f(
            zero, 1.0 - m, one
        )
        rd = torchscience.special_functions.carlson_elliptic_integral_r_d(
            zero, 1.0 - m, one
        )
        expected = rf - (m / 3.0) * rd
        torch.testing.assert_close(E, expected, rtol=1e-8, atol=1e-10)

    def test_monotonically_decreasing(self):
        """E(m) is monotonically decreasing on [0, 1]."""
        m = torch.linspace(0.0, 0.99, 50, dtype=torch.float64)
        E = torchscience.special_functions.complete_legendre_elliptic_integral_e(
            m
        )
        assert (E[1:] < E[:-1]).all()
