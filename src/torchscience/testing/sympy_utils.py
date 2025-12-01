from typing import Any, Callable, List, Optional

import numpy
import sympy
import torch
from sympy import I, N


class SymPyReference:
    """Compute reference values using SymPy for verification."""

    def __init__(
        self,
        sympy_func: Callable[..., Any],
        precision: int = 50,
    ):
        """Initialize SymPy reference.

        Args:
            sympy_func: SymPy function to use for reference computation.
            precision: Number of decimal digits for high-precision computation.
        """
        self.sympy_func = sympy_func
        self.precision = precision

    def evaluate_scalar(self, *args: float | complex) -> float | complex:
        """Evaluate the function at a single point."""
        # Convert to SymPy types
        sympy_args = []
        for arg in args:
            if isinstance(arg, complex):
                sympy_args.append(
                    sympy.Float(arg.real) + I * sympy.Float(arg.imag)
                )
            else:
                sympy_args.append(sympy.Float(arg))

        result = N(self.sympy_func(*sympy_args), self.precision)

        # Convert back to Python types
        if result.is_real:
            return float(result)
        return complex(result)

    def evaluate_real(self, *args: numpy.ndarray) -> numpy.ndarray:
        """Evaluate the function at real points."""
        result = numpy.zeros_like(args[0], dtype=numpy.float64)
        for idx in numpy.ndindex(args[0].shape):
            point = tuple(float(arg[idx]) for arg in args)
            result[idx] = float(N(self.sympy_func(*point), self.precision))
        return result

    def evaluate_complex(self, *args: numpy.ndarray) -> numpy.ndarray:
        """Evaluate the function at complex points."""
        result = numpy.zeros_like(args[0], dtype=numpy.complex128)
        for idx in numpy.ndindex(args[0].shape):
            point = []
            for arg in args:
                val = arg[idx]
                if numpy.iscomplexobj(arg):
                    # Convert to SymPy complex
                    point.append(
                        sympy.Float(val.real) + I * sympy.Float(val.imag)
                    )
                else:
                    point.append(sympy.Float(float(val)))
            sympy_result = N(self.sympy_func(*point), self.precision)
            result[idx] = complex(sympy_result)
        return result

    def to_torch(
        self,
        *tensors: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Compute reference values and return as torch tensor.

        Args:
            *tensors: Input tensors.
            dtype: Output dtype. If None, uses the dtype of the first tensor.

        Returns:
            Tensor with reference values.
        """
        if dtype is None:
            dtype = tensors[0].dtype

        numpy_args = [t.detach().cpu().numpy() for t in tensors]

        is_complex = any(t.is_complex() for t in tensors)
        if is_complex:
            result = self.evaluate_complex(*numpy_args)
        else:
            result = self.evaluate_real(*numpy_args)

        return torch.tensor(result, dtype=dtype)


class SymbolicDerivativeVerifier:
    """Verify PyTorch autograd against symbolic derivatives."""

    def __init__(
        self,
        sympy_expr: sympy.Expr,
        variables: List[sympy.Symbol],
        precision: int = 30,
    ):
        """Initialize derivative verifier.

        Args:
            sympy_expr: SymPy expression to differentiate.
            variables: List of SymPy symbols (variables) in the expression.
            precision: Number of decimal digits for evaluation.
        """
        self.expr = sympy_expr
        self.variables = variables
        self.precision = precision

        # Pre-compute symbolic derivatives
        self._first_derivatives = [
            sympy.diff(sympy_expr, var) for var in variables
        ]
        self._second_derivatives = [
            [sympy.diff(d, var2) for var2 in variables]
            for d in self._first_derivatives
        ]

    def evaluate_gradient(
        self,
        variable_idx: int,
        *point_values: float | complex,
    ) -> float | complex:
        """Evaluate first derivative at a point.

        Args:
            variable_idx: Index of the variable to differentiate with respect to.
            *point_values: Values at which to evaluate.

        Returns:
            Value of the derivative at the point.
        """
        symbolic_grad = self._first_derivatives[variable_idx]
        subs_dict = dict(zip(self.variables, point_values))
        result = N(symbolic_grad.subs(subs_dict), self.precision)
        if result.is_real:
            return float(result)
        return complex(result)

    def evaluate_hessian_element(
        self,
        var_idx1: int,
        var_idx2: int,
        *point_values: float | complex,
    ) -> float | complex:
        """Evaluate second derivative at a point.

        Args:
            var_idx1: First variable index.
            var_idx2: Second variable index.
            *point_values: Values at which to evaluate.

        Returns:
            Value of the second derivative at the point.
        """
        symbolic_hess = self._second_derivatives[var_idx1][var_idx2]
        subs_dict = dict(zip(self.variables, point_values))
        result = N(symbolic_hess.subs(subs_dict), self.precision)
        if result.is_real:
            return float(result)
        return complex(result)

    def verify_first_derivative(
        self,
        pytorch_grad: torch.Tensor,
        variable_idx: int,
        *point_values: float | complex,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> bool:
        """Verify first derivative at a point.

        Args:
            pytorch_grad: Gradient computed by PyTorch autograd.
            variable_idx: Index of the variable.
            *point_values: Values at which gradient was computed.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if the gradient matches within tolerance.
        """
        expected = self.evaluate_gradient(variable_idx, *point_values)
        actual = pytorch_grad.item()
        return numpy.allclose(actual, expected, rtol=rtol, atol=atol)

    def verify_hessian(
        self,
        pytorch_hessian: torch.Tensor,
        *point_values: float | complex,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> bool:
        """Verify Hessian at a point.

        Args:
            pytorch_hessian: Hessian computed by PyTorch.
            *point_values: Values at which Hessian was computed.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if the Hessian matches within tolerance.
        """
        n = len(self.variables)
        for i in range(n):
            for j in range(n):
                expected = self.evaluate_hessian_element(i, j, *point_values)
                actual = pytorch_hessian[i, j].item()
                if not numpy.allclose(actual, expected, rtol=rtol, atol=atol):
                    return False
        return True
