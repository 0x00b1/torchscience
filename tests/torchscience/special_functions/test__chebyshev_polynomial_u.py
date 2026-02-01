import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SpecialValue,
    ToleranceConfig,
)


def scipy_chebyshev_u(n: float, x: float) -> float:
    """Wrapper for scipy Chebyshev U function."""
    return scipy.special.eval_chebyu(n, x)


def reference_chebyshev_u(n, x):
    """Reference implementation using sin((n+1)*acos(x))/sin(acos(x))."""
    if torch.is_complex(n) or torch.is_complex(x):
        if not torch.is_complex(x):
            x = x.to(
                torch.complex128
                if x.dtype == torch.float64
                else torch.complex64
            )
        if not torch.is_complex(n):
            n = n.to(x.dtype)
    theta = torch.acos(x)
    sin_theta = torch.sin(theta)
    # Handle the case where sin(theta) is 0 (i.e., x = ±1)
    result = torch.where(
        torch.abs(sin_theta) < 1e-10,
        (n + 1) * torch.sign(torch.cos(theta)) ** n,
        torch.sin((n + 1) * theta) / sin_theta,
    )
    return result


def _check_recurrence(func) -> bool:
    """Check U_n(x) = 2x*U_{n-1}(x) - U_{n-2}(x)."""
    x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    for n in range(2, 10):
        n_n = torch.tensor([float(n)], dtype=torch.float64)
        n_n_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
        n_n_2 = torch.tensor([float(n - 2)], dtype=torch.float64)

        left = func(n_n, x)
        right = 2 * x * func(n_n_1, x) - func(n_n_2, x)
        if not torch.allclose(left, right, rtol=1e-10, atol=1e-10):
            return False
    return True


class TestChebyshevPolynomialU(OpTestCase):
    """Tests for Chebyshev polynomial of the second kind."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="chebyshev_polynomial_u",
            func=torchscience.special_functions.chebyshev_polynomial_u,
            arity=2,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(0.0, 10.0),
                    can_be_integer=True,
                    supports_grad=False,  # n gradient is always 0
                ),
                InputSpec(
                    name="x",
                    position=1,
                    default_real_range=(-0.99, 0.99),
                    supports_grad=True,
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_complex_dtypes",  # No complex kernel implementation
                "test_dtype_preservation",  # Includes complex dtypes which are not implemented
                "test_gradcheck_complex",  # No complex kernel implementation
                "test_gradgradcheck_complex",  # No complex kernel implementation
                "test_gradgradcheck_real",  # Relax tolerance for backward_backward
                "test_sympy_reference_complex",  # No complex kernel implementation
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="three_term_recurrence",
                    check_fn=_check_recurrence,
                    description="U_n(x) = 2x*U_{n-1}(x) - U_{n-2}(x)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=1.0,
                    description="U_0(x) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.5),
                    expected=1.0,
                    description="U_1(0.5) = 2*0.5 = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=0.0,
                    description="U_2(0.5) = 4*0.25 - 1 = 0",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    # =========================================================================
    # Chebyshev U specific tests
    # =========================================================================

    def test_integer_degree_polynomial_values(self):
        """Test exact polynomial values for integer degrees."""
        x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float64)

        # U_0(x) = 1
        n = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected)

        # U_1(x) = 2x
        n = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        expected = 2 * x
        torch.testing.assert_close(result, expected)

        # U_2(x) = 4x^2 - 1
        n = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        expected = 4 * x**2 - 1
        torch.testing.assert_close(result, expected)

        # U_3(x) = 8x^3 - 4x
        n = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        expected = 8 * x**3 - 4 * x
        torch.testing.assert_close(result, expected)

    def test_values_at_x_equals_1(self):
        """Test U_n(1) = n + 1."""
        x = torch.tensor([1.0], dtype=torch.float64)
        for degree in range(10):
            n = torch.tensor([float(degree)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x
            )
            expected = torch.tensor([float(degree + 1)], dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_values_at_x_equals_minus_1(self):
        """Test U_n(-1) = (-1)^n * (n + 1)."""
        x = torch.tensor([-1.0], dtype=torch.float64)
        for degree in range(10):
            n = torch.tensor([float(degree)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x
            )
            sign = 1.0 if degree % 2 == 0 else -1.0
            expected = torch.tensor([sign * (degree + 1)], dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_reference(self):
        """Test against scipy reference values."""
        x_vals = torch.tensor([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=torch.float64)
        for degree in range(10):
            n = torch.tensor([float(degree)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x_vals
            )
            expected = torch.tensor(
                [scipy_chebyshev_u(degree, x.item()) for x in x_vals],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_hyperbolic_continuation_z_greater_than_1(self):
        """Test U_n(x) = sinh((n+1)*acosh(x))/sinh(acosh(x)) for x > 1."""
        x = torch.tensor([1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        for degree in range(5):
            n = torch.tensor([float(degree)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x
            )
            expected = torch.tensor(
                [scipy_chebyshev_u(degree, xi.item()) for xi in x],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_hyperbolic_continuation_z_less_than_minus_1(self):
        """Test U_n(x) for x < -1."""
        x = torch.tensor([-1.5, -2.0, -3.0], dtype=torch.float64)
        for degree in range(5):
            n = torch.tensor([float(degree)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x
            )
            expected = torch.tensor(
                [scipy_chebyshev_u(degree, xi.item()) for xi in x],
                dtype=torch.float64,
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_gradcheck_standard_domain(self):
        """Test gradient correctness for |x| < 1."""
        x = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.chebyshev_polynomial_u(n, x)

        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z_greater_than_1(self):
        """Test gradient correctness for x > 1."""
        x = torch.tensor(
            [1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.chebyshev_polynomial_u(n, x)

        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z_less_than_minus_1(self):
        """Test gradient correctness for x < -1."""
        x = torch.tensor(
            [-1.5, -2.0, -3.0], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([2.0], dtype=torch.float64)

        def func(x):
            return torchscience.special_functions.chebyshev_polynomial_u(n, x)

        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_relationship_with_chebyshev_t(self):
        """Test dT_n/dx = n * U_{n-1}."""
        x = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n = torch.tensor([3.0], dtype=torch.float64)

        # Compute dT_n/dx
        t_n = torchscience.special_functions.chebyshev_polynomial_t(n, x)
        (grad_t,) = torch.autograd.grad(t_n.sum(), x)

        # Compute n * U_{n-1}
        n_minus_1 = torch.tensor([2.0], dtype=torch.float64)
        u_n_minus_1 = torchscience.special_functions.chebyshev_polynomial_u(
            n_minus_1, x.detach()
        )
        expected = n * u_n_minus_1

        torch.testing.assert_close(grad_t, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype):
        """Test forward pass with low-precision dtypes."""
        x = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
        n = torch.tensor([2.0], dtype=dtype)

        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        assert result.dtype == dtype

        # Compare against float32 reference
        x_f32 = x.to(torch.float32)
        n_f32 = n.to(torch.float32)
        expected = torchscience.special_functions.chebyshev_polynomial_u(
            n_f32, x_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    def test_large_batch_dimensions(self):
        """Test with large batch dimensions."""
        batch_sizes = [(1000,), (100, 100), (10, 10, 10)]

        for shape in batch_sizes:
            x = torch.rand(shape, dtype=torch.float64) * 1.98 - 0.99
            n = torch.tensor([2.0], dtype=torch.float64)

            result = torchscience.special_functions.chebyshev_polynomial_u(
                n, x
            )

            assert result.shape == shape
            # U_2(x) = 4x^2 - 1
            expected = 4 * x**2 - 1
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_broadcasting(self):
        """Test broadcasting between n and x."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        x = torch.tensor([[0.5, 0.7]], dtype=torch.float64)

        result = torchscience.special_functions.chebyshev_polynomial_u(n, x)
        assert result.shape == (3, 2)

        # Verify each element
        for i in range(3):
            for j in range(2):
                expected = scipy_chebyshev_u(n[i, 0].item(), x[0, j].item())
                torch.testing.assert_close(
                    result[i, j],
                    torch.tensor(expected, dtype=torch.float64),
                    rtol=1e-10,
                    atol=1e-10,
                )

    # =========================================================================
    # CUDA-specific tests
    # =========================================================================

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward_matches_cpu(self):
        """Test that CUDA forward pass matches CPU results."""
        x_cpu = torch.linspace(-0.99, 0.99, 100, dtype=torch.float64)
        n_cpu = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)

        x_cuda = x_cpu.cuda()
        n_cuda = n_cpu.cuda()

        for i in range(len(n_cpu)):
            n_i_cpu = n_cpu[i : i + 1]
            n_i_cuda = n_cuda[i : i + 1]

            result_cpu = torchscience.special_functions.chebyshev_polynomial_u(
                n_i_cpu, x_cpu
            )
            result_cuda = (
                torchscience.special_functions.chebyshev_polynomial_u(
                    n_i_cuda, x_cuda
                )
            )

            torch.testing.assert_close(
                result_cuda.cpu(),
                result_cpu,
                rtol=1e-12,
                atol=1e-12,
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_backward_matches_cpu(self):
        """Test that CUDA backward pass matches CPU results."""
        x_cpu = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        n_cpu = torch.tensor([2.0], dtype=torch.float64)

        x_cuda = x_cpu.detach().clone().cuda().requires_grad_(True)
        n_cuda = n_cpu.cuda()

        # Forward
        result_cpu = torchscience.special_functions.chebyshev_polynomial_u(
            n_cpu, x_cpu
        )
        result_cuda = torchscience.special_functions.chebyshev_polynomial_u(
            n_cuda, x_cuda
        )

        # Backward
        result_cpu.sum().backward()
        result_cuda.sum().backward()

        torch.testing.assert_close(
            x_cuda.grad.cpu(), x_cpu.grad, rtol=1e-10, atol=1e-10
        )
