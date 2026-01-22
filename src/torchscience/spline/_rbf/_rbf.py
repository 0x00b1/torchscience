"""RBF interpolator representation and convenience function."""

from typing import Callable, Optional

from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class RBFInterpolator:
    """Radial Basis Function interpolator for scattered data.

    RBF interpolation constructs a smooth function that passes through
    (or approximates) scattered data points by combining radial basis
    functions centered at the data points.

    The interpolant has the form:
    f(x) = Σ w_i φ(||x - c_i||) + p(x)

    where φ is the RBF kernel, w_i are weights, c_i are centers,
    and p(x) is an optional polynomial term for conditionally
    positive definite kernels.

    Attributes
    ----------
    centers : Tensor
        Center locations, shape (n, dim).
    weights : Tensor
        RBF weights, shape (n, *value_shape).
    polynomial_coeffs : Tensor
        Polynomial coefficients for CPD kernels, shape (m, *value_shape).
        For thin_plate in 2D, m = 3 (1, x, y).
    kernel : str
        Kernel name: "thin_plate", "gaussian", "multiquadric", etc.
    epsilon : float
        Shape parameter (for kernels that need it).
    """

    centers: Tensor
    weights: Tensor
    polynomial_coeffs: Tensor
    kernel: str
    epsilon: float


def rbf_interpolate(
    points: Tensor,
    values: Tensor,
    kernel: str = "thin_plate",
    epsilon: Optional[float] = None,
    smoothing: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Create an RBF interpolator for scattered data.

    This is a convenience function that fits an RBF interpolator and
    returns a callable that evaluates it.

    Parameters
    ----------
    points : Tensor
        Data point locations, shape (n, dim).
    values : Tensor
        Data values, shape (n,) or (n, *value_shape).
    kernel : str, optional
        RBF kernel. One of:

        - ``"thin_plate"``: r² log(r), good for smooth interpolation (default)
        - ``"gaussian"``: exp(-ε²r²), localized influence
        - ``"multiquadric"``: sqrt(1 + ε²r²), smooth global interpolation
        - ``"inverse_quadratic"``: 1/(1 + ε²r²), localized
        - ``"inverse_multiquadric"``: 1/sqrt(1 + ε²r²)
        - ``"cubic"``: r³
        - ``"linear"``: r

    epsilon : float, optional
        Shape parameter for Gaussian, multiquadric, and inverse kernels.
        If None, automatically estimated from data.
    smoothing : float, optional
        Smoothing parameter (0 for exact interpolation). Default is 0.

    Returns
    -------
    interpolator : Callable[[Tensor], Tensor]
        Function that evaluates the RBF at query points.

    Examples
    --------
    >>> import torch
    >>> # Scattered 2D data
    >>> points = torch.rand(20, 2)
    >>> values = torch.sin(points[:, 0]) * torch.cos(points[:, 1])
    >>> f = rbf_interpolate(points, values)
    >>> query = torch.tensor([[0.5, 0.5]])
    >>> f(query)
    """
    from ._rbf_evaluate import rbf_evaluate
    from ._rbf_fit import rbf_fit

    rbf = rbf_fit(
        points, values, kernel=kernel, epsilon=epsilon, smoothing=smoothing
    )
    return lambda q: rbf_evaluate(rbf, q)
