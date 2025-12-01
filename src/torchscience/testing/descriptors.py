from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Set, Tuple

import torch


@dataclass
class ToleranceConfig:
    """Tolerance configuration for numerical comparisons."""

    # Default tolerances by dtype
    float16_rtol: float = 1e-2
    float16_atol: float = 1e-2
    bfloat16_rtol: float = 5e-2
    bfloat16_atol: float = 5e-2
    float32_rtol: float = 1e-5
    float32_atol: float = 1e-5
    float64_rtol: float = 1e-10
    float64_atol: float = 1e-10
    complex64_rtol: float = 1e-5
    complex64_atol: float = 1e-5
    complex128_rtol: float = 1e-10
    complex128_atol: float = 1e-10

    # Gradcheck tolerances
    gradcheck_eps: float = 1e-6
    gradcheck_rtol: float = 1e-4
    gradcheck_atol: float = 1e-4

    # Gradgradcheck tolerances (relaxed for second-order)
    gradgradcheck_eps: float = 1e-6
    gradgradcheck_rtol: float = 1e-3
    gradgradcheck_atol: float = 1e-3

    # SymPy reference tolerances
    sympy_rtol: float = 1e-10
    sympy_atol: float = 1e-10

    def get_tolerances(self, dtype: torch.dtype) -> Tuple[float, float]:
        """Return (rtol, atol) for the given dtype."""
        mapping = {
            torch.float16: (self.float16_rtol, self.float16_atol),
            torch.bfloat16: (self.bfloat16_rtol, self.bfloat16_atol),
            torch.float32: (self.float32_rtol, self.float32_atol),
            torch.float64: (self.float64_rtol, self.float64_atol),
            torch.complex64: (self.complex64_rtol, self.complex64_atol),
            torch.complex128: (self.complex128_rtol, self.complex128_atol),
        }
        return mapping.get(dtype, (1e-5, 1e-5))


@dataclass
class InputSpec:
    """Specification for an operator input."""

    name: str
    position: int = 0

    # Supported dtypes
    real_dtypes: list[torch.dtype] = field(
        default_factory=lambda: [torch.float32, torch.float64]
    )
    complex_dtypes: list[torch.dtype] = field(
        default_factory=lambda: [torch.complex64, torch.complex128]
    )
    low_precision_dtypes: list[torch.dtype] = field(
        default_factory=lambda: [torch.float16, torch.bfloat16]
    )

    # Domain constraints for Hypothesis strategies
    default_real_range: Tuple[float, float] = (-10.0, 10.0)
    default_imag_range: Tuple[float, float] = (-10.0, 10.0)
    excluded_values: Set[float] = field(default_factory=set)

    # Complex domain constraint: if set, generates complex values with |z| < max
    # This is useful for functions like incomplete_beta that require |z| < 1
    complex_magnitude_max: Optional[float] = None

    # Gradient support
    supports_grad: bool = True

    # Special value handling
    can_be_integer: bool = False


@dataclass
class RecurrenceSpec:
    """Specification for a recurrence relation test."""

    name: str
    check_fn: Callable[[Callable[..., torch.Tensor]], bool]
    description: str = ""


@dataclass
class IdentitySpec:
    """Specification for a functional identity test."""

    name: str
    identity_fn: Callable[
        [Callable[..., torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]
    ]
    rtol: float = 1e-10
    atol: float = 1e-10
    description: str = ""


@dataclass
class SingularitySpec:
    """Specification for singularity (pole/branch cut) behavior."""

    type: str  # "pole", "branch_cut", "essential"
    locations: Callable[[], Iterator[float | complex]]
    expected_behavior: str = "inf"  # "inf", "nan", "complex_valued"
    description: str = ""


@dataclass
class SpecialValue:
    """A special value test case."""

    inputs: Tuple[float | complex, ...]
    expected: float | complex
    rtol: float = 1e-10
    atol: float = 1e-10
    description: str = ""


@dataclass
class OperatorDescriptor:
    """Describes a PyTorch operator for testing."""

    # Basic info
    name: str
    func: Callable[..., torch.Tensor]
    arity: int  # 1 for unary, 2 for binary

    # Input specifications
    input_specs: list[InputSpec] = field(default_factory=list)

    # SymPy integration
    sympy_func: Optional[Callable[..., Any]] = None
    sympy_derivative_funcs: Optional[list[Callable[..., Any]]] = None

    # Tolerance configurations
    tolerances: ToleranceConfig = field(default_factory=ToleranceConfig)

    # Skip configurations
    skip_tests: Set[str] = field(default_factory=set)

    # Mathematical properties
    recurrence_relations: list[RecurrenceSpec] = field(default_factory=list)
    functional_identities: list[IdentitySpec] = field(default_factory=list)
    special_values: list[SpecialValue] = field(default_factory=list)
    singularities: list[SingularitySpec] = field(default_factory=list)

    # Sparse tensor support
    supports_sparse_coo: bool = True
    supports_sparse_csr: bool = True

    # Quantized tensor support
    supports_quantized: bool = True

    # Meta tensor support
    supports_meta: bool = True
