"""Second-order sections (SOS) filter implementation."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from ._lfilter import lfilter


def sosfilt(
    sos: Tensor,
    x: Tensor,
    axis: int = -1,
    zi: Optional[Tensor] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Filter data along one dimension using cascaded second-order sections.

    Second-order sections (SOS) representation provides better numerical
    stability than direct transfer function (b, a) representation for
    high-order filters.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape ``(n_sections, 6)``. Each row contains
        ``[b0, b1, b2, a0, a1, a2]`` for one biquad section. Typically
        ``a0 = 1.0`` for each section.
    x : Tensor
        Input signal. Can be batched with arbitrary leading dimensions.
    axis : int, optional
        Axis along which to filter. Default is -1 (last axis).
    zi : Tensor, optional
        Initial conditions for each section, shape
        ``(..., n_sections, 2)`` where ``...`` matches the batch dimensions
        of ``x``. If provided, returns ``(y, zf)`` where ``zf`` is the
        final filter delays.

    Returns
    -------
    y : Tensor
        Filtered signal, same shape as ``x``.
    zf : Tensor, optional
        Final filter delays for each section (only if ``zi`` was provided).

    Notes
    -----
    The filter is implemented by cascading biquad sections:

    .. math::
        y = H_1(H_2(...H_K(x)...))

    where each section :math:`H_i` is a second-order IIR filter:

    .. math::
        H_i(z) = \\frac{b_{i,0} + b_{i,1} z^{-1} + b_{i,2} z^{-2}}
                      {a_{i,0} + a_{i,1} z^{-1} + a_{i,2} z^{-2}}

    The SOS representation avoids numerical issues that arise with
    high-order polynomial representations.

    Fully differentiable with respect to ``sos`` and ``x``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import sosfilt
    >>> # A 4th-order Butterworth lowpass filter (2 biquad sections)
    >>> sos = torch.tensor([
    ...     [0.067, 0.135, 0.067, 1.0, -1.143, 0.413],
    ...     [1.0, 2.0, 1.0, 1.0, -1.561, 0.641]
    ... ], dtype=torch.float64)
    >>> x = torch.randn(100, dtype=torch.float64)
    >>> y = sosfilt(sos, x)

    >>> # With initial conditions (for smooth continuation)
    >>> from torchscience.signal_processing.filter_design import sosfilt_zi
    >>> zi = sosfilt_zi(sos) * x[0]
    >>> y, zf = sosfilt(sos, x, zi=zi)
    """
    # Validate SOS shape
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    n_sections = sos.shape[0]

    # Get batch shape from x (excluding axis dimension)
    x_moved = torch.moveaxis(x, axis, -1)
    batch_shape = x_moved.shape[:-1]
    n_samples = x_moved.shape[-1]

    # Initialize output and final state storage
    y = x.clone()

    if zi is not None:
        # zi should have shape (..., n_sections, 2)
        zi_moved = torch.moveaxis(zi, axis if axis != -1 else -1, -1)
        # After moveaxis, zi_moved should have zi dimensions at end
        # We expect zi shape to be (batch..., n_sections, 2)
        expected_zi_shape = batch_shape + (n_sections, 2)

        # For batched input, zi could have shape (batch..., n_sections, 2)
        # or for unbatched zi with batch input, could be (n_sections, 2)
        if zi.shape == (n_sections, 2) and len(batch_shape) > 0:
            # Broadcast zi to batch shape
            zi_expanded = zi.unsqueeze(0).expand(batch_shape + (n_sections, 2))
            zi_moved = zi_expanded
        else:
            zi_moved = zi

        zf = torch.empty_like(zi_moved)

    # Cascade through sections
    for i in range(n_sections):
        # Extract b and a for this section
        b = sos[i, :3]
        a = sos[i, 3:]

        if zi is not None:
            # Extract zi for this section: shape (..., 2)
            zi_section = zi_moved[..., i, :]
            y, zf_section = lfilter(b, a, y, axis=axis, zi=zi_section)
            zf[..., i, :] = zf_section
        else:
            y = lfilter(b, a, y, axis=axis)

    if zi is not None:
        return y, zf

    return y
