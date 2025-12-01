"""Utility for computing number of SOS sections."""

from typing import Literal


def sos_sections_count(
    order: int,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
) -> int:
    """Compute the number of second-order sections for a filter.

    Parameters
    ----------
    order : int
        Filter order (number of poles for lowpass/highpass prototype).
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, default "lowpass"
        Type of filter.

    Returns
    -------
    n_sections : int
        Number of second-order sections in the SOS representation.

    Notes
    -----
    For lowpass and highpass filters:
        n_sections = ceil(order / 2)

    For bandpass and bandstop filters, the order doubles during the
    frequency transform, so:
        n_sections = 2 * ceil(order / 2) = order (for even order)
                   = 2 * ceil(order / 2) = order + 1 (for odd order)

    This function is useful for pre-allocating buffers or validating
    expected shapes before calling filter design functions.

    Examples
    --------
    >>> from torchscience.signal_processing.filter_design import sos_sections_count
    >>> sos_sections_count(4, "lowpass")
    2
    >>> sos_sections_count(4, "bandpass")
    4
    >>> sos_sections_count(5, "lowpass")
    3
    >>> sos_sections_count(5, "bandpass")
    6
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")

    # Base sections for lowpass/highpass: ceil(order / 2)
    base_sections = (order + 1) // 2

    if filter_type in ("lowpass", "highpass"):
        return base_sections
    elif filter_type in ("bandpass", "bandstop"):
        # Bandpass/bandstop doubles the order
        return 2 * base_sections
    else:
        raise ValueError(
            f"Invalid filter_type: {filter_type}. "
            "Must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'."
        )
