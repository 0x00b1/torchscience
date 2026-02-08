import mpmath
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    ToleranceConfig,
)


class TestJacobiEllipticNs(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="jacobi_elliptic_ns",
            func=torchscience.special_functions.jacobi_elliptic_ns,
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
            result = torchscience.special_functions.jacobi_elliptic_ns(u, m)
            expected = float(mpmath.ellipfun("ns", u_val, m=m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_reciprocal_identity(self):
        """ns(u, m) = 1 / sn(u, m)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        ns = torchscience.special_functions.jacobi_elliptic_ns(u, m)
        sn = torchscience.special_functions.jacobi_elliptic_sn(u, m)
        torch.testing.assert_close(ns, 1.0 / sn, rtol=1e-10, atol=1e-10)

    def test_circular_limit(self):
        """ns(u, 0) = csc(u) = 1/sin(u)."""
        u = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_ns(u, m)
        expected = 1.0 / torch.sin(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_limit(self):
        """ns(u, 1) = coth(u) = 1/tanh(u)."""
        u = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.jacobi_elliptic_ns(u, m)
        expected = 1.0 / torch.tanh(u)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
