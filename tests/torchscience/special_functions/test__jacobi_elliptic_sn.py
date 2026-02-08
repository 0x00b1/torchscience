import mpmath
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


def _pythagorean_sn_cn(func):
    """Check sn(u,m)^2 + cn(u,m)^2 = 1."""
    u = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
    cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
    left = sn**2 + cn**2
    right = torch.ones_like(left)
    return left, right


def _pythagorean_dn_sn(func):
    """Check dn(u,m)^2 + m*sn(u,m)^2 = 1."""
    u = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
    dn = torchscience.special_functions.jacobi_elliptic_dn(u, m)
    left = dn**2 + m * sn**2
    right = torch.ones_like(left)
    return left, right


class TestJacobiEllipticSn(OpTestCase):
    """Tests for the Jacobi elliptic function sn(u, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_sn",
            func=torchscience.special_functions.jacobi_elliptic_sn,
            arity=2,
            input_specs=[
                InputSpec(name="u", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(name="m", position=1, default_real_range=(0.1, 0.9)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradcheck_complex",
                "test_nan_propagation_all_inputs",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="sn(0, m) = 0",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="pythagorean_sn_cn",
                    identity_fn=_pythagorean_sn_cn,
                    description="sn^2 + cn^2 = 1",
                ),
                IdentitySpec(
                    name="pythagorean_dn_sn",
                    identity_fn=_pythagorean_dn_sn,
                    description="dn^2 + m*sn^2 = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.ellipfun('sn', u, m)."""
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_elliptic_sn(u, m)
            expected = float(mpmath.ellipfun("sn", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_circular_limit_m_zero(self):
        """sn(u, 0) = sin(u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        expected = torch.sin(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit_m_one(self):
        """sn(u, 1) = tanh(u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        expected = torch.tanh(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_odd_function(self):
        """sn(-u, m) = -sn(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        left = torchscience.special_functions.jacobi_elliptic_sn(-u, m)
        right = -torchscience.special_functions.jacobi_elliptic_sn(u, m)
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
