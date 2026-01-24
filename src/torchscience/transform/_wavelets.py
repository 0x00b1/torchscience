"""Wavelet filter coefficients for discrete wavelet transform.

This module provides filter coefficients for common wavelet families.
Each wavelet has four filters:
- dec_lo: Decomposition lowpass filter (scaling)
- dec_hi: Decomposition highpass filter (wavelet)
- rec_lo: Reconstruction lowpass filter
- rec_hi: Reconstruction highpass filter
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

# Wavelet filter coefficients
# All coefficients are normalized for energy preservation

# Haar wavelet (Daubechies-1)
_SQRT2_INV = 1.0 / math.sqrt(2.0)

WAVELET_FILTERS: dict[str, dict[str, Sequence[float]]] = {
    # Haar wavelet (simplest)
    "haar": {
        "dec_lo": [_SQRT2_INV, _SQRT2_INV],
        "dec_hi": [-_SQRT2_INV, _SQRT2_INV],
        "rec_lo": [_SQRT2_INV, _SQRT2_INV],
        "rec_hi": [_SQRT2_INV, -_SQRT2_INV],
    },
    # Daubechies-1 (same as Haar)
    "db1": {
        "dec_lo": [_SQRT2_INV, _SQRT2_INV],
        "dec_hi": [-_SQRT2_INV, _SQRT2_INV],
        "rec_lo": [_SQRT2_INV, _SQRT2_INV],
        "rec_hi": [_SQRT2_INV, -_SQRT2_INV],
    },
    # Daubechies-2
    "db2": {
        "dec_lo": [
            0.4829629131445341,
            0.8365163037378079,
            0.2241438680420134,
            -0.1294095225512604,
        ],
        "dec_hi": [
            -0.1294095225512604,
            -0.2241438680420134,
            0.8365163037378079,
            -0.4829629131445341,
        ],
        "rec_lo": [
            -0.1294095225512604,
            0.2241438680420134,
            0.8365163037378079,
            0.4829629131445341,
        ],
        "rec_hi": [
            -0.4829629131445341,
            0.8365163037378079,
            -0.2241438680420134,
            -0.1294095225512604,
        ],
    },
    # Daubechies-3
    "db3": {
        "dec_lo": [
            0.3326705529500826,
            0.8068915093110925,
            0.4598775021184915,
            -0.1350110200102546,
            -0.0854412738820267,
            0.0352262918857095,
        ],
        "dec_hi": [
            0.0352262918857095,
            0.0854412738820267,
            -0.1350110200102546,
            -0.4598775021184915,
            0.8068915093110925,
            -0.3326705529500826,
        ],
        "rec_lo": [
            0.0352262918857095,
            -0.0854412738820267,
            -0.1350110200102546,
            0.4598775021184915,
            0.8068915093110925,
            0.3326705529500826,
        ],
        "rec_hi": [
            -0.3326705529500826,
            0.8068915093110925,
            -0.4598775021184915,
            -0.1350110200102546,
            0.0854412738820267,
            0.0352262918857095,
        ],
    },
    # Daubechies-4
    "db4": {
        "dec_lo": [
            0.2303778133088965,
            0.7148465705529156,
            0.6308807679398587,
            -0.0279837694168599,
            -0.1870348117190930,
            0.0308413818355607,
            0.0328830116668852,
            -0.0105974017850690,
        ],
        "dec_hi": [
            -0.0105974017850690,
            -0.0328830116668852,
            0.0308413818355607,
            0.1870348117190930,
            -0.0279837694168599,
            -0.6308807679398587,
            0.7148465705529156,
            -0.2303778133088965,
        ],
        "rec_lo": [
            -0.0105974017850690,
            0.0328830116668852,
            0.0308413818355607,
            -0.1870348117190930,
            -0.0279837694168599,
            0.6308807679398587,
            0.7148465705529156,
            0.2303778133088965,
        ],
        "rec_hi": [
            -0.2303778133088965,
            0.7148465705529156,
            -0.6308807679398587,
            -0.0279837694168599,
            0.1870348117190930,
            0.0308413818355607,
            -0.0328830116668852,
            -0.0105974017850690,
        ],
    },
}


def get_wavelet_filters(
    wavelet: str,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Get wavelet filter coefficients as tensors.

    Parameters
    ----------
    wavelet : str
        Name of the wavelet. Supported wavelets:
        - ``"haar"``: Haar wavelet (simplest, 2 coefficients)
        - ``"db1"``: Daubechies-1 (same as Haar)
        - ``"db2"``: Daubechies-2 (4 coefficients)
        - ``"db3"``: Daubechies-3 (6 coefficients)
        - ``"db4"``: Daubechies-4 (8 coefficients)
    dtype : torch.dtype, optional
        Desired dtype of the filter tensors.
        Default: ``torch.float32``.
    device : torch.device or str, optional
        Desired device of the filter tensors.
        Default: ``"cpu"``.

    Returns
    -------
    tuple of Tensor
        A tuple ``(dec_lo, dec_hi, rec_lo, rec_hi)`` where:
        - ``dec_lo``: Decomposition lowpass filter (scaling function)
        - ``dec_hi``: Decomposition highpass filter (wavelet function)
        - ``rec_lo``: Reconstruction lowpass filter
        - ``rec_hi``: Reconstruction highpass filter

    Raises
    ------
    ValueError
        If the wavelet name is not recognized.

    Examples
    --------
    >>> dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("haar")
    >>> dec_lo
    tensor([0.7071, 0.7071])

    >>> dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db2", dtype=torch.float64)
    >>> dec_lo.dtype
    torch.float64
    """
    wavelet_lower = wavelet.lower()

    if wavelet_lower not in WAVELET_FILTERS:
        available = sorted(WAVELET_FILTERS.keys())
        raise ValueError(
            f"Unknown wavelet '{wavelet}'. Available wavelets: {available}"
        )

    if dtype is None:
        dtype = torch.float32

    filters = WAVELET_FILTERS[wavelet_lower]

    dec_lo = torch.tensor(filters["dec_lo"], dtype=dtype, device=device)
    dec_hi = torch.tensor(filters["dec_hi"], dtype=dtype, device=device)
    rec_lo = torch.tensor(filters["rec_lo"], dtype=dtype, device=device)
    rec_hi = torch.tensor(filters["rec_hi"], dtype=dtype, device=device)

    return dec_lo, dec_hi, rec_lo, rec_hi


def get_wavelet_filter_length(wavelet: str) -> int:
    """Get the filter length for a wavelet.

    Parameters
    ----------
    wavelet : str
        Name of the wavelet.

    Returns
    -------
    int
        Length of the wavelet filters.

    Raises
    ------
    ValueError
        If the wavelet name is not recognized.
    """
    wavelet_lower = wavelet.lower()

    if wavelet_lower not in WAVELET_FILTERS:
        available = sorted(WAVELET_FILTERS.keys())
        raise ValueError(
            f"Unknown wavelet '{wavelet}'. Available wavelets: {available}"
        )

    return len(WAVELET_FILTERS[wavelet_lower]["dec_lo"])
