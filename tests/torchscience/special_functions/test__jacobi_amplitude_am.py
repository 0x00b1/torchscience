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


class TestJacobiAmplitudeAm(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_amplitude_am",
            func=torchscience.special_functions.jacobi_amplitude_am,
            arity=2,
            input_specs=[
                InputSpec(name="u", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(name="m", position=1, default_real_range=(0.1, 0.9)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_nan_propagation_all_inputs",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5), expected=0.0, description="am(0, m) = 0"
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_relation_to_sn(self):
        """sin(am(u, m)) = sn(u, m)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        am = torchscience.special_functions.jacobi_amplitude_am(u, m)
        sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        torch.testing.assert_close(torch.sin(am), sn, rtol=1e-10, atol=1e-10)

    def test_circular_limit_m_zero(self):
        """am(u, 0) = u."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_amplitude_am(u, m)
        torch.testing.assert_close(result, u, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit_m_one(self):
        """am(u, 1) = 2*arctan(exp(u)) - pi/2 = gd(u) (Gudermannian)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_amplitude_am(u, m)
        expected = 2.0 * torch.atan(torch.exp(u)) - math.pi / 2
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_against_mpmath(self):
        """Compare against mpmath: am(u,m) = arcsin(sn(u,m))."""
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_amplitude_am(u, m)
            sn_val = float(mpmath.ellipfun("sn", u_val, m=m_val))
            expected = math.asin(sn_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_inverse_of_F(self):
        """am is inverse of F: F(am(u, m), m) = u."""
        u = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        am = torchscience.special_functions.jacobi_amplitude_am(u, m)
        F = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            am, m
        )
        torch.testing.assert_close(F, u, rtol=1e-8, atol=1e-8)
