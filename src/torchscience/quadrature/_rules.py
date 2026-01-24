"""Quadrature rule classes."""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from torchscience.quadrature._nodes import (
    gauss_kronrod_nodes_weights,
    gauss_legendre_nodes_weights,
)


class GaussLegendre:
    """
    Gauss-Legendre quadrature rule.

    Exact for polynomials of degree <= 2n-1.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Examples
    --------
    >>> rule = GaussLegendre(32)
    >>> nodes, weights = rule.nodes_and_weights(a=0, b=1)
    >>> result = rule.integrate(torch.sin, 0, torch.pi)  # approximately 2.0

    Attributes
    ----------
    n : int
        Number of points.
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self._cache: dict = {}

    def _get_base_nodes_weights(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Get cached base nodes/weights on [-1, 1]."""
        key = (str(dtype), str(device))
        if key not in self._cache:
            self._cache[key] = gauss_legendre_nodes_weights(
                self.n, dtype=dtype, device=device
            )
        return self._cache[key]

    def nodes_and_weights(
        self,
        a: Union[float, Tensor] = -1.0,
        b: Union[float, Tensor] = 1.0,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return nodes and weights scaled to [a, b].

        If a and b are tensors, returns batched nodes/weights.

        Parameters
        ----------
        a, b : float or Tensor
            Integration bounds. Can be batched.
        dtype : torch.dtype, optional
            Output dtype. Inferred from a/b if not specified.
        device : torch.device, optional
            Output device. Inferred from a/b if not specified.

        Returns
        -------
        nodes : Tensor
            Shape (*batch, n) if a/b are tensors, else (n,).
        weights : Tensor
            Shape (*batch, n) if a/b are tensors, else (n,).
        """
        # Infer dtype and device
        if isinstance(a, Tensor):
            dtype = dtype or a.dtype
            device = device or a.device
        elif isinstance(b, Tensor):
            dtype = dtype or b.dtype
            device = device or b.device
        else:
            dtype = dtype or torch.float64
            device = device or torch.device("cpu")

        # Ensure a and b are tensors
        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=dtype, device=device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=dtype, device=device)

        # Get base nodes/weights on [-1, 1]
        base_nodes, base_weights = self._get_base_nodes_weights(dtype, device)

        # Linear transformation from [-1, 1] to [a, b]
        # x' = (b - a) / 2 * x + (a + b) / 2
        # weights scale by (b - a) / 2
        half_width = (b - a) / 2
        center = (a + b) / 2

        # Handle batched limits
        if a.dim() > 0 or b.dim() > 0:
            # Broadcast and expand
            half_width = half_width.unsqueeze(-1)  # (*batch, 1)
            center = center.unsqueeze(-1)  # (*batch, 1)
            nodes = half_width * base_nodes + center
            weights = half_width * base_weights
        else:
            nodes = half_width * base_nodes + center
            weights = half_width * base_weights

        return nodes, weights

    def integrate(
        self,
        f: Callable[[Tensor], Tensor],
        a: Union[float, Tensor],
        b: Union[float, Tensor],
    ) -> Tensor:
        """
        Integrate f from a to b.

        Parameters
        ----------
        f : callable
            Integrand function. Takes tensor of shape (*batch, n), returns same.
        a, b : float or Tensor
            Integration bounds.

        Returns
        -------
        Tensor
            Integral value(s). Shape matches broadcast(a, b) or scalar.
        """
        nodes, weights = self.nodes_and_weights(a, b)
        values = f(nodes)
        return (values * weights).sum(dim=-1)


class GaussKronrod:
    """
    Gauss-Kronrod quadrature rule with embedded error estimation.

    Uses G(n)-K(2n+1) pairs: G7-K15, G10-K21.

    Parameters
    ----------
    order : int
        Kronrod order: 15 or 21.

    Examples
    --------
    >>> rule = GaussKronrod(15)  # G7-K15
    >>> result, error = rule.integrate_with_error(torch.sin, 0, torch.pi)

    Attributes
    ----------
    order : int
        Number of Kronrod points.
    """

    def __init__(self, order: int = 15):
        if order not in (15, 21):
            raise ValueError(f"order must be 15 or 21, got {order}")
        self.order = order
        self._cache: dict = {}

    def _get_nodes_weights(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get cached nodes and weights."""
        key = (str(dtype), str(device))
        if key not in self._cache:
            self._cache[key] = gauss_kronrod_nodes_weights(
                self.order, dtype=dtype, device=device
            )
        return self._cache[key]

    def integrate(
        self,
        f: Callable[[Tensor], Tensor],
        a: Union[float, Tensor],
        b: Union[float, Tensor],
    ) -> Tensor:
        """
        Integrate f from a to b using Kronrod rule.

        Parameters
        ----------
        f : callable
            Integrand function.
        a, b : float or Tensor
            Integration bounds.

        Returns
        -------
        Tensor
            Kronrod approximation.
        """
        result, _ = self.integrate_with_error(f, a, b)
        return result

    def integrate_with_error(
        self,
        f: Callable[[Tensor], Tensor],
        a: Union[float, Tensor],
        b: Union[float, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Integrate f with error estimate.

        Parameters
        ----------
        f : callable
            Integrand function.
        a, b : float or Tensor
            Integration bounds.

        Returns
        -------
        result : Tensor
            Kronrod approximation.
        error : Tensor
            Estimated error (|Kronrod - Gauss|).
        """
        # Infer dtype and device
        if isinstance(a, Tensor):
            dtype = a.dtype
            device = a.device
        elif isinstance(b, Tensor):
            dtype = b.dtype
            device = b.device
        else:
            dtype = torch.float64
            device = torch.device("cpu")

        # Ensure tensors
        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=dtype, device=device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=dtype, device=device)

        # Get nodes and weights on [-1, 1]
        nodes, k_weights, g_weights, g_indices = self._get_nodes_weights(
            dtype, device
        )

        # Transform to [a, b]
        half_width = (b - a) / 2
        center = (a + b) / 2

        scaled_nodes = half_width * nodes + center
        scaled_k_weights = half_width * k_weights
        scaled_g_weights = half_width * g_weights

        # Evaluate function at all nodes
        values = f(scaled_nodes)

        # Kronrod result (all nodes)
        kronrod_result = (values * scaled_k_weights).sum(dim=-1)

        # Gauss result (only at Gauss indices)
        gauss_values = values[..., g_indices]
        gauss_result = (gauss_values * scaled_g_weights).sum(dim=-1)

        error = torch.abs(kronrod_result - gauss_result)

        return kronrod_result, error
