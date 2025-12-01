"""Utility for normalizing second-order sections."""

from torch import Tensor


def sos_normalize(
    sos: Tensor,
) -> Tensor:
    """Normalize second-order sections so that a0 = 1 for all sections.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].

    Returns
    -------
    sos_normalized : Tensor
        Normalized SOS with a0 = 1 for all sections. Same shape as input.

    Notes
    -----
    This function divides each row by its a0 coefficient, converting
    potentially non-normalized SOS (from external sources or file I/O)
    to the normalized form expected by torchscience analysis functions.

    All filter design functions in torchscience produce already-normalized
    SOS, so this function is primarily useful for:
    - Loading SOS from external files
    - Converting SOS from other libraries that may not normalize
    - Ensuring normalization after manual coefficient manipulation

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import sos_normalize
    >>> # External SOS with a0 != 1
    >>> sos_external = torch.tensor([[0.5, 0.25, 0.125, 2.0, 0.2, 0.1]])
    >>> sos = sos_normalize(sos_external)
    >>> sos
    tensor([[0.2500, 0.1250, 0.0625, 1.0000, 0.1000, 0.0500]])

    >>> # Verify a0 = 1
    >>> sos[:, 3]
    tensor([1.])
    """
    if sos.numel() == 0:
        return sos

    # Get a0 coefficients (column 3)
    a0 = sos[:, 3:4]  # Keep dims for broadcasting

    # Divide each row by its a0
    sos_normalized = sos / a0

    return sos_normalized
