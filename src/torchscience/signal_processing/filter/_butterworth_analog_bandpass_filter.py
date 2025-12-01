"""Butterworth analog bandpass filter implementation."""

import math
from typing import Optional, Tuple, Union, overload

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


@overload
def butterworth_analog_bandpass_filter(
    n: int,
    passband: Tuple[Union[float, Tensor], Union[float, Tensor]],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Signature 1: Order n and passband frequencies."""
    ...


@overload
def butterworth_analog_bandpass_filter(
    n: int,
    center_q: Tuple[Tuple[Union[float, Tensor], Union[float, Tensor]]],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Signature 2: Order n, center frequency and Q factor."""
    ...


@overload
def butterworth_analog_bandpass_filter(
    spec: Tuple[
        Union[float, Tensor],
        Union[float, Tensor],
        Union[float, Tensor],
        Union[float, Tensor],
    ],
    attenuations: Tuple[Union[float, Tensor], Union[float, Tensor]],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Signature 3: Full specification."""
    ...


def butterworth_analog_bandpass_filter(
    n_or_spec: Union[
        int,
        Tuple[
            Union[float, Tensor],
            Union[float, Tensor],
            Union[float, Tensor],
            Union[float, Tensor],
        ],
    ],
    freqs_or_attenuations: Union[
        Tuple[Union[float, Tensor], Union[float, Tensor]],
        Tuple[Tuple[Union[float, Tensor], Union[float, Tensor]]],
    ],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Butterworth analog bandpass filter in SOS format.

    Computes the coefficients of an analog Butterworth bandpass filter
    and returns them in second-order sections (SOS) format.

    Mathematical Definition
    -----------------------
    The Butterworth filter is characterized by a maximally flat magnitude
    response in the passband. The analog bandpass filter is designed by:

    1. Creating a Butterworth lowpass prototype with poles at:

       .. math::
           s_k = e^{j \pi (2k + n - 1) / (2n)} \quad \text{for } k = 1, \ldots, n

    2. Applying the lowpass-to-bandpass transformation:

       .. math::
           s \to \frac{s^2 + \omega_0^2}{B \cdot s}

       where :math:`\omega_0 = \sqrt{\omega_{p1} \cdot \omega_{p2}}` is the
       center frequency and :math:`B = \omega_{p2} - \omega_{p1}` is the
       bandwidth.

    3. This produces 2n poles for an order-n bandpass filter, organized
       into n second-order sections.

    The transfer function of each section is:

    .. math::
        H_k(s) = \frac{b_0 s^2 + b_1 s + b_2}{a_0 s^2 + a_1 s + a_2}

    API Signatures (Wolfram/Mathematica style)
    ------------------------------------------
    1. ``n, (omega_p1, omega_p2)`` - Bandpass with order n and normalized
       passband edge frequencies (0-1 where 1 = Nyquist).

    2. ``n, ((omega, q),)`` - Center frequency omega and quality factor q.
       Q = omega_0 / bandwidth.

    3. ``(omega_s1, omega_p1, omega_p2, omega_s2), (a_s, a_p)`` - Full
       specification with stopband/passband frequencies and attenuations
       in dB. Filter order is computed automatically.

    Parameters
    ----------
    n_or_spec : int or tuple of 4 floats/Tensors
        For signatures 1 and 2: filter order (positive integer).
        For signature 3: tuple ``(omega_s1, omega_p1, omega_p2, omega_s2)``
        specifying lower stopband, lower passband, upper passband, and
        upper stopband edge frequencies.
    freqs_or_attenuations : tuple
        For signature 1: ``(omega_p1, omega_p2)`` passband edge frequencies.
        For signature 2: ``((omega, q),)`` center frequency and Q factor.
        For signature 3: ``(a_s, a_p)`` stopband and passband attenuations in dB.
    dtype : torch.dtype, optional
        Output tensor dtype. If ``None``, uses the default floating point type.
    device : torch.device, optional
        Output tensor device. If ``None``, uses CPU.

    Returns
    -------
    Tensor
        SOS format coefficients with shape ``(*batch_shape, n_sections, 6)``.
        Each row contains ``[b0, b1, b2, a0, a1, a2]`` for the transfer function:

        .. math::
            H_k(s) = \frac{b_0 s^2 + b_1 s + b_2}{a_0 s^2 + a_1 s + a_2}

        For scalar inputs, shape is ``(n, 6)``.
        For batched tensor inputs, shape is ``(*batch_shape, n, 6)``.

    Examples
    --------
    Order-4 bandpass with passband 0.2-0.5 (normalized):

    >>> sos = butterworth_analog_bandpass_filter(4, (0.2, 0.5))
    >>> sos.shape
    torch.Size([4, 6])

    Center frequency 0.3 with Q=2:

    >>> sos = butterworth_analog_bandpass_filter(2, ((0.3, 2.0),))
    >>> sos.shape
    torch.Size([2, 6])

    Full specification with 40dB stopband and 1dB passband ripple:

    >>> sos = butterworth_analog_bandpass_filter(
    ...     (0.1, 0.2, 0.5, 0.7),  # stopband/passband edges
    ...     (40.0, 1.0)            # attenuations in dB
    ... )

    Batched computation (multiple filters in parallel):

    >>> omega_p1 = torch.tensor([0.1, 0.15, 0.2])
    >>> omega_p2 = torch.tensor([0.4, 0.45, 0.5])
    >>> sos = butterworth_analog_bandpass_filter(3, (omega_p1, omega_p2))
    >>> sos.shape
    torch.Size([3, 3, 6])  # 3 filters, each with 3 sections, 6 coefficients

    Notes
    -----
    This returns analog (s-domain) filter coefficients. For digital
    filtering applications, use bilinear transformation to convert
    to z-domain coefficients.

    The filter is designed with unity gain at the center frequency
    :math:`\omega_0 = \sqrt{\omega_{p1} \cdot \omega_{p2}}`.

    For numerical stability with high-order filters, the SOS format
    cascades second-order sections rather than using a single high-order
    transfer function.

    Gradients are supported with respect to the frequency parameters
    ``omega_p1`` and ``omega_p2``, enabling gradient-based filter
    optimization.

    References
    ----------
    .. [1] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal Processing,"
           3rd ed., Prentice Hall, 2009.

    .. [2] S.K. Mitra, "Digital Signal Processing: A Computer-Based Approach,"
           4th ed., McGraw-Hill, 2011.

    .. [3] S. Butterworth, "On the Theory of Filter Amplifiers,"
           Wireless Engineer, vol. 7, pp. 536-541, 1930.

    See Also
    --------
    scipy.signal.butter : SciPy's Butterworth filter design function.
    """
    # Determine output options
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # Parse signature based on argument types
    if isinstance(n_or_spec, int):
        n = n_or_spec

        # Check for signature 2: n, ((omega, q),)
        if (
            isinstance(freqs_or_attenuations, tuple)
            and len(freqs_or_attenuations) == 1
            and isinstance(freqs_or_attenuations[0], tuple)
            and len(freqs_or_attenuations[0]) == 2
        ):
            # Signature 2: center frequency and Q factor
            omega, q = freqs_or_attenuations[0]
            omega_t = _to_tensor(omega, dtype, device)
            q_t = _to_tensor(q, dtype, device)

            # Convert Q to passband frequencies
            # Q = omega_0 / B where B = omega_p2 - omega_p1
            # omega_0 = sqrt(omega_p1 * omega_p2)
            # B = omega / q
            B = omega_t / q_t
            # From omega_0^2 = omega_p1 * omega_p2 and omega_p2 - omega_p1 = B
            # omega_p1 = -B/2 + sqrt(B^2/4 + omega^2)
            # omega_p2 = omega_p1 + B
            discriminant = (B * B) / 4 + omega_t * omega_t
            omega_p1 = -B / 2 + torch.sqrt(discriminant)
            omega_p2 = omega_p1 + B

            return torch.ops.torchscience.butterworth_analog_bandpass_filter(
                n, omega_p1, omega_p2
            )

        elif (
            isinstance(freqs_or_attenuations, tuple)
            and len(freqs_or_attenuations) == 2
        ):
            # Signature 1: n, (omega_p1, omega_p2)
            omega_p1, omega_p2 = freqs_or_attenuations
            omega_p1_t = _to_tensor(omega_p1, dtype, device)
            omega_p2_t = _to_tensor(omega_p2, dtype, device)

            return torch.ops.torchscience.butterworth_analog_bandpass_filter(
                n, omega_p1_t, omega_p2_t
            )
        else:
            raise ValueError(
                "Invalid frequency specification. Expected (omega_p1, omega_p2) "
                "or ((omega, q),)"
            )

    elif isinstance(n_or_spec, tuple) and len(n_or_spec) == 4:
        # Signature 3: full specification
        omega_s1, omega_p1, omega_p2, omega_s2 = n_or_spec

        if not (
            isinstance(freqs_or_attenuations, tuple)
            and len(freqs_or_attenuations) == 2
        ):
            raise ValueError(
                "For full specification, attenuations must be (a_s, a_p)"
            )

        a_s, a_p = freqs_or_attenuations

        # Convert to tensors
        omega_s1_t = _to_tensor(omega_s1, dtype, device)
        omega_p1_t = _to_tensor(omega_p1, dtype, device)
        omega_p2_t = _to_tensor(omega_p2, dtype, device)
        omega_s2_t = _to_tensor(omega_s2, dtype, device)
        a_s_t = _to_tensor(a_s, dtype, device)
        a_p_t = _to_tensor(a_p, dtype, device)

        # Compute minimum order from specifications
        n = _compute_butterworth_order(
            omega_s1_t, omega_p1_t, omega_p2_t, omega_s2_t, a_s_t, a_p_t
        )

        return torch.ops.torchscience.butterworth_analog_bandpass_filter(
            n, omega_p1_t, omega_p2_t
        )

    else:
        raise TypeError(
            f"First argument must be int (order) or tuple of 4 floats/Tensors "
            f"(frequency spec), got {type(n_or_spec)}"
        )


def _to_tensor(
    value: Union[float, Tensor], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Convert a scalar or tensor to a tensor with specified dtype and device."""
    if isinstance(value, Tensor):
        return value.to(dtype=dtype, device=device)
    else:
        return torch.tensor(value, dtype=dtype, device=device)


def _compute_butterworth_order(
    omega_s1: Tensor,
    omega_p1: Tensor,
    omega_p2: Tensor,
    omega_s2: Tensor,
    a_s: Tensor,
    a_p: Tensor,
) -> int:
    """Compute minimum Butterworth filter order from specifications.

    Uses the formula:
    n >= log10((10^(a_s/10) - 1) / (10^(a_p/10) - 1)) / (2 * log10(omega_s / omega_p))

    For bandpass, we transform to equivalent lowpass specification.
    """
    # Compute center frequency and bandwidth
    omega_0 = torch.sqrt(omega_p1 * omega_p2)
    B = omega_p2 - omega_p1

    # Transform stopband frequencies to lowpass prototype
    # omega_s_lp = |omega_s^2 - omega_0^2| / (B * omega_s)
    omega_s1_lp = torch.abs(omega_s1 * omega_s1 - omega_0 * omega_0) / (
        B * omega_s1
    )
    omega_s2_lp = torch.abs(omega_s2 * omega_s2 - omega_0 * omega_0) / (
        B * omega_s2
    )

    # Use the more restrictive (smaller) stopband frequency
    omega_s_lp = torch.min(omega_s1_lp, omega_s2_lp)

    # Compute epsilon values
    epsilon_p = torch.sqrt(torch.pow(torch.tensor(10.0), a_p / 10) - 1)
    epsilon_s = torch.sqrt(torch.pow(torch.tensor(10.0), a_s / 10) - 1)

    # Compute order
    n_exact = torch.log10(epsilon_s / epsilon_p) / torch.log10(omega_s_lp)
    n = int(math.ceil(n_exact.item()))

    return max(1, n)  # Ensure at least order 1
