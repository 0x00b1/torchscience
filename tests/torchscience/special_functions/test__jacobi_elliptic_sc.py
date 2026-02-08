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


class TestJacobiEllipticSc(OpTestCase):
    """Tests for the Jacobi elliptic function sc(u, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_sc",
            func=torchscience.special_functions.jacobi_elliptic_sc,
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
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="sc(0, m) = 0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.ellipfun('sc', u, m)."""
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_elliptic_sc(u, m)
            expected = float(mpmath.ellipfun("sc", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_quotient_identity(self):
        """sc(u, m) = sn(u, m) / cn(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        sc = torchscience.special_functions.jacobi_elliptic_sc(u, m)
        sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
        torch.testing.assert_close(sc, sn / cn, rtol=1e-10, atol=1e-10)

    def test_circular_limit(self):
        """sc(u, 0) = tan(u)."""
        u = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_sc(u, m)
        expected = torch.tan(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit(self):
        """sc(u, 1) = sinh(u)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_sc(u, m)
        expected = torch.sinh(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_odd_function(self):
        """sc(-u, m) = -sc(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        left = torchscience.special_functions.jacobi_elliptic_sc(-u, m)
        right = -torchscience.special_functions.jacobi_elliptic_sc(u, m)
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
