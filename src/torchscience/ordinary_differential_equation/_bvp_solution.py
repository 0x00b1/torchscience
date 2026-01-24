"""BVP solution data structure."""

from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.ordinary_differential_equation._interpolation import (
    hermite_interpolate,
)


@tensorclass
class BVPSolution:
    """Solution of a boundary value problem.

    Attributes
    ----------
    x : Tensor
        Mesh nodes, shape (n_nodes,). Strictly increasing from a to b.
    y : Tensor
        Solution values, shape (n_components, n_nodes).
    yp : Tensor
        Solution derivatives (dy/dx), shape (n_components, n_nodes).
        Used for cubic Hermite interpolation.
    p : Tensor
        Unknown parameters, shape (n_params,). Empty if no parameters.
    rms_residuals : Tensor
        RMS of normalized residuals at final iteration.
    n_iterations : int
        Number of Newton iterations taken.
    success : bool
        Whether the solver converged.
    """

    x: Tensor
    y: Tensor
    yp: Tensor
    p: Tensor
    rms_residuals: Tensor
    n_iterations: int
    success: bool

    def __call__(self, x_query: Tensor) -> Tensor:
        """Evaluate solution at query points using cubic Hermite interpolation.

        Parameters
        ----------
        x_query : Tensor
            Points to evaluate solution at, shape (*query_shape,).

        Returns
        -------
        Tensor
            Solution values, shape (n_components, *query_shape).

        Notes
        -----
        Uses cubic Hermite interpolation which matches both function values
        and derivatives at mesh nodes. This provides 4th-order accuracy,
        matching the collocation method's order.
        """
        query_shape = x_query.shape
        x_query_flat = x_query.flatten()

        # Use cubic Hermite interpolation
        y_interp = hermite_interpolate(self.x, self.y, self.yp, x_query_flat)

        # Reshape to match query shape
        result_shape = (self.y.shape[0],) + query_shape
        return y_interp.reshape(result_shape)
