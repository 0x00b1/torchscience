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


def _quotient_identity(func):
    """Check cd(u, m) = cn(u, m) / dn(u, m)."""
    u = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    cd = torchscience.special_functions.jacobi_elliptic_cd(u, m)
    cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
    dn = torchscience.special_functions.jacobi_elliptic_dn(u, m)
    left = cd
    right = cn / dn
    return left, right


def _pythagorean_cd_sd(func):
    """Check cd(u, m)^2 + m * sd(u, m)^2 = 1."""
    u = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    m = torch.tensor([0.5], dtype=torch.float64)
    cd = torchscience.special_functions.jacobi_elliptic_cd(u, m)
    sd = torchscience.special_functions.jacobi_elliptic_sd(u, m)
    left = cd**2 + m * sd**2
    right = torch.ones_like(left)
    return left, right


class TestJacobiEllipticCd(OpTestCase):
    """Tests for the Jacobi elliptic function cd(u, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_cd",
            func=torchscience.special_functions.jacobi_elliptic_cd,
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
                    expected=1.0,
                    description="cd(0, m) = 1",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="quotient_identity",
                    identity_fn=_quotient_identity,
                    description="cd = cn / dn",
                ),
                IdentitySpec(
                    name="pythagorean_cd_sd",
                    identity_fn=_pythagorean_cd_sd,
                    description="cd^2 + m*sd^2 = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.ellipfun('cd', u, m)."""
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_elliptic_cd(u, m)
            expected = float(mpmath.ellipfun("cd", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_circular_limit_m_zero(self):
        """cd(u, 0) = cos(u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_cd(u, m)
        expected = torch.cos(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit_m_one(self):
        """cd(u, 1) = 1."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_cd(u, m)
        expected = torch.ones_like(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_even_function(self):
        """cd(-u, m) = cd(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        left = torchscience.special_functions.jacobi_elliptic_cd(-u, m)
        right = torchscience.special_functions.jacobi_elliptic_cd(u, m)
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
