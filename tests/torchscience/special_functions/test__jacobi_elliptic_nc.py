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


class TestJacobiEllipticNc(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_nc",
            func=torchscience.special_functions.jacobi_elliptic_nc,
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
                "test_low_precision_forward",
                "test_nan_propagation_all_inputs",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5), expected=1.0, description="nc(0, m) = 1"
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        test_cases = [(0.5, 0.5), (1.0, 0.3), (2.0, 0.7)]
        for u_val, m_val in test_cases:
            u = torch.tensor([u_val], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.jacobi_elliptic_nc(u, m)
            expected = float(mpmath.ellipfun("nc", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_reciprocal_identity(self):
        """nc(u, m) = 1 / cn(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        nc = torchscience.special_functions.jacobi_elliptic_nc(u, m)
        cn = torchscience.special_functions.jacobi_elliptic_cn(u, m)
        torch.testing.assert_close(nc, 1.0 / cn, rtol=1e-10, atol=1e-10)

    def test_circular_limit(self):
        """nc(u, 0) = sec(u) = 1/cos(u)."""
        u = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_nc(u, m)
        expected = 1.0 / torch.cos(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit(self):
        """nc(u, 1) = cosh(u)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_nc(u, m)
        expected = torch.cosh(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
