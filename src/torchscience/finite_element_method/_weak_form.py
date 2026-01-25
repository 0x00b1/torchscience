"""Weak form representation for variational formulations of PDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass
class WeakForm:
    """Variational formulation of a partial differential equation.

    Represents the weak form of a PDE for finite element discretization.
    A weak form consists of:

    - A bilinear form a(u, v) integrated over the domain
    - A linear form L(v) integrated over the domain
    - Optionally, boundary integrals for Neumann or Robin conditions

    The finite element method seeks u in V such that:
        a(u, v) = L(v) for all v in V

    Attributes
    ----------
    bilinear_form : Callable
        The bilinear form a(u, v, x) -> Tensor.
        Arguments:
            - u: Trial function (with .value and/or .grad attributes)
            - v: Test function (with .value and/or .grad attributes)
            - x: Physical coordinates at quadrature points, shape (n_points, dim)
        Returns:
            - Tensor of shape (n_points,) to be integrated with quadrature weights

    linear_form : Callable
        The linear form L(v, x) -> Tensor.
        Arguments:
            - v: Test function (with .value and/or .grad attributes)
            - x: Physical coordinates at quadrature points, shape (n_points, dim)
        Returns:
            - Tensor of shape (n_points,) to be integrated with quadrature weights

    boundary_form : Callable | None
        Optional boundary integral for Neumann or Robin conditions.
        Arguments:
            - v: Test function values on boundary
            - x: Boundary quadrature points, shape (n_boundary_points, dim)
            - n: Outward unit normals, shape (n_boundary_points, dim)
        Returns:
            - Tensor of shape (n_boundary_points,) to be integrated

    Examples
    --------
    Poisson equation: -nabla^2 u = f

    >>> def poisson_bilinear(u, v, x):
    ...     # Stiffness form: integral of grad(u) dot grad(v)
    ...     return (u.grad * v.grad).sum(dim=-1)
    ...
    >>> def poisson_linear(v, x):
    ...     # Source term: f(x) * v where f = 1
    ...     return v.value
    ...
    >>> poisson = WeakForm(
    ...     bilinear_form=poisson_bilinear,
    ...     linear_form=poisson_linear,
    ... )

    Heat equation with Neumann BC:

    >>> def heat_bilinear(u, v, x):
    ...     return u.value * v.value + 0.1 * (u.grad * v.grad).sum(dim=-1)
    ...
    >>> def heat_linear(v, x):
    ...     return torch.zeros_like(v.value)
    ...
    >>> def neumann_bc(v, x, n):
    ...     g = 1.0  # prescribed flux
    ...     return g * v.value
    ...
    >>> heat = WeakForm(
    ...     bilinear_form=heat_bilinear,
    ...     linear_form=heat_linear,
    ...     boundary_form=neumann_bc,
    ... )

    Notes
    -----
    The trial function `u` and test function `v` passed to the forms typically
    have the following attributes:
    - `value`: Function values at quadrature points, shape (n_points,) or
               (n_points, n_components) for vector problems
    - `grad`: Function gradients at quadrature points, shape (n_points, dim)
              or (n_points, n_components, dim) for vector problems

    For scalar problems like Poisson, these are scalar-valued at each point.
    For vector problems like elasticity, `value` and `grad` have additional
    component dimensions.

    This class uses a standard dataclass rather than tensorclass because
    the fields are callables rather than tensors. The callables themselves
    operate on tensor inputs and return tensor outputs.
    """

    bilinear_form: Callable[[object, object, Tensor], Tensor]
    linear_form: Callable[[object, Tensor], Tensor]
    boundary_form: Callable[[object, Tensor, Tensor], Tensor] | None = None
