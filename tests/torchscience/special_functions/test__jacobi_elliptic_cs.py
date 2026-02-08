import mpmath
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


def _quotient_identity(func):
    """Check cs(u, m) = cn(u, m) / sn(u, m)."""
    u = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    cs = torchscience.special_functions.jacobi_elliptic_cs(u, m)
    cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
    sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
    left = cs
    right = cn / sn
    return left, right


class TestJacobiEllipticCs(OpTestCase):
    """Tests for the Jacobi elliptic function cs(u, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_cs",
            func=torchscience.special_functions.jacobi_elliptic_cs,
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
            functional_identities=[
                IdentitySpec(
                    name="quotient_identity",
                    identity_fn=_quotient_identity,
                    description="cs(u, m) = cn(u, m) / sn(u, m)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.ellipfun('cs', u, m)."""
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_elliptic_cs(u, m)
            expected = float(mpmath.ellipfun("cs", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_quotient_identity(self):
        """cs(u, m) = cn(u, m) / sn(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        cs = torchscience.special_functions.jacobi_elliptic_cs(u, m)
        cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
        sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        torch.testing.assert_close(cs, cn / sn, rtol=1e-10, atol=1e-10)

    def test_circular_limit(self):
        """cs(u, 0) = cot(u)."""
        u = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_cs(u, m)
        expected = torch.cos(u) / torch.sin(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit(self):
        """cs(u, 1) = csch(u) = 1/sinh(u)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_cs(u, m)
        expected = 1.0 / torch.sinh(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_odd_function(self):
        """cs(-u, m) = -cs(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        left = torchscience.special_functions.jacobi_elliptic_cs(-u, m)
        right = -torchscience.special_functions.jacobi_elliptic_cs(u, m)
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
