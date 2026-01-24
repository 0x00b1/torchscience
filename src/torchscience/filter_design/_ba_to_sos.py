"""Conversion from transfer function coefficients to second-order sections."""

from torch import Tensor

from ._ba_to_zpk import ba_to_zpk
from ._zpk_to_sos import zpk_to_sos


def ba_to_sos(
    numerator: Tensor,
    denominator: Tensor,
    pairing: str = "nearest",
) -> Tensor:
    """Convert transfer function coefficients to second-order sections.

    Parameters
    ----------
    numerator : Tensor
        Numerator polynomial coefficients in descending order.
    denominator : Tensor
        Denominator polynomial coefficients in descending order.
    pairing : str, default "nearest"
        Pairing strategy for poles and zeros: "nearest" or "keep_odd".

    Returns
    -------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].

    Notes
    -----
    This function converts via the ZPK representation:
    BA -> ZPK -> SOS

    This is done because direct BA to SOS conversion would require
    root finding anyway, and the ZPK intermediate form allows for
    optimal pole-zero pairing.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import ba_to_sos
    >>> b = torch.tensor([0.0675, 0.2025, 0.2025, 0.0675])
    >>> a = torch.tensor([1.0, -0.8178, 0.4536, -0.0697])
    >>> sos = ba_to_sos(b, a)
    """
    # Convert to ZPK first
    zeros, poles, gain = ba_to_zpk(numerator, denominator)

    # Then convert ZPK to SOS
    return zpk_to_sos(zeros, poles, gain, pairing=pairing)
