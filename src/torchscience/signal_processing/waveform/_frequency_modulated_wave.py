from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def frequency_modulated_wave(
    n: Optional[int] = None,
    t: Optional[Tensor] = None,
    *,
    carrier_frequency: Union[float, Tensor] = 100.0,
    modulator_frequency: Optional[Union[float, Tensor]] = None,
    modulating_signal: Optional[Tensor] = None,
    modulation_index: Union[float, Tensor] = 1.0,
    sample_rate: float = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    phase: Union[float, Tensor] = 0.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Frequency-modulated waveform generator.

    Generates an FM signal where the instantaneous frequency varies according to
    a modulating signal. Two modes are supported:

    1. **Sinusoidal modulation**: When `modulator_frequency` is provided,
       uses a sinusoidal modulator at that frequency.

       .. math::
           y(t) = A \\cos(2\\pi f_c t + \\beta \\sin(2\\pi f_m t) + \\phi)

    2. **Arbitrary modulation**: When `modulating_signal` is provided,
       uses that tensor as the modulator, integrating it over time.

       .. math::
           y(t) = A \\cos\\left(2\\pi f_c t + \\beta \\int_0^t m(\\tau) d\\tau + \\phi\\right)

    Parameters
    ----------
    n : int, optional
        Number of samples. Mutually exclusive with t.
    t : Tensor, optional
        Explicit time tensor. Mutually exclusive with n.
    carrier_frequency : float or Tensor
        Carrier frequency in Hz. Tensor enables batched generation. Default is 100.0.
    modulator_frequency : float or Tensor, optional
        Modulator frequency in Hz for sinusoidal FM. Tensor enables batched
        generation. Mutually exclusive with modulating_signal.
    modulating_signal : Tensor, optional
        Arbitrary modulating waveform tensor with shape (..., n_samples).
        Mutually exclusive with modulator_frequency.
    modulation_index : float or Tensor
        Modulation index (beta). For sinusoidal FM, this equals the frequency
        deviation divided by the modulator frequency. Default is 1.0.
    sample_rate : float
        Samples per second. Ignored when t is provided. Default is 1.0.
    amplitude : float or Tensor
        Peak amplitude. Tensor enables batched generation. Default is 1.0.
    phase : float or Tensor
        Initial phase in radians. Tensor enables batched generation. Default is 0.0.
    dtype : torch.dtype, optional
        Output dtype.
    layout : torch.layout, optional
        Output layout.
    device : torch.device, optional
        Output device.
    requires_grad : bool
        If True, enables gradient computation. Default is False.

    Returns
    -------
    Tensor
        Shape (*broadcast_shape, n) or (*broadcast_shape, *t.shape)

    Raises
    ------
    ValueError
        If both n and t are provided, or neither is provided.
        If both modulator_frequency and modulating_signal are provided.
        If neither modulator_frequency nor modulating_signal is provided.

    Examples
    --------
    Generate FM signal with sinusoidal modulator:

    >>> result = frequency_modulated_wave(
    ...     n=1000,
    ...     carrier_frequency=100.0,
    ...     modulator_frequency=5.0,
    ...     modulation_index=2.0,
    ...     sample_rate=1000.0,
    ... )
    >>> result.shape
    torch.Size([1000])

    Generate FM with custom modulating signal:

    >>> t = torch.linspace(0, 1, 1000)
    >>> modulating_signal = torch.sin(2 * torch.pi * 5 * t)  # 5 Hz modulator
    >>> result = frequency_modulated_wave(
    ...     t=t,
    ...     carrier_frequency=100.0,
    ...     modulating_signal=modulating_signal,
    ...     modulation_index=2.0,
    ... )

    Generate batched FM with different carrier frequencies:

    >>> carriers = torch.tensor([100.0, 200.0, 300.0])
    >>> result = frequency_modulated_wave(
    ...     n=1000,
    ...     carrier_frequency=carriers,
    ...     modulator_frequency=5.0,
    ...     modulation_index=1.0,
    ...     sample_rate=1000.0,
    ... )
    >>> result.shape
    torch.Size([3, 1000])

    Notes
    -----
    Frequency modulation is widely used in radio broadcasting (FM radio),
    telecommunications, and audio synthesis. The modulation index determines
    the bandwidth of the FM signal according to Carson's rule:

    .. math::
        BW \\approx 2(\\Delta f + f_m) = 2 f_m (\\beta + 1)

    where :math:`\\Delta f = \\beta f_m` is the frequency deviation.
    """
    # Validate mutual exclusivity of n and t
    if n is not None and t is not None:
        raise ValueError("n and t are mutually exclusive - provide only one")
    if n is None and t is None:
        raise ValueError("Either n or t must be provided")

    # Validate mutual exclusivity of modulation modes
    if modulator_frequency is not None and modulating_signal is not None:
        raise ValueError(
            "modulator_frequency and modulating_signal are mutually exclusive - "
            "provide only one"
        )
    if modulator_frequency is None and modulating_signal is None:
        raise ValueError(
            "Either modulator_frequency or modulating_signal must be provided"
        )

    # Convert scalars to 0-D tensors
    if not isinstance(carrier_frequency, Tensor):
        carrier_frequency = torch.tensor(
            carrier_frequency, dtype=dtype or torch.float32, device=device
        )
    if not isinstance(modulation_index, Tensor):
        modulation_index = torch.tensor(
            modulation_index, dtype=dtype or torch.float32, device=device
        )
    if not isinstance(amplitude, Tensor):
        amplitude = torch.tensor(
            amplitude, dtype=dtype or torch.float32, device=device
        )
    if not isinstance(phase, Tensor):
        phase = torch.tensor(
            phase, dtype=dtype or torch.float32, device=device
        )

    # Determine common dtype
    if dtype is None:
        dtype = torch.result_type(carrier_frequency, modulation_index)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), amplitude)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), phase)
        if t is not None:
            dtype = torch.result_type(torch.empty(0, dtype=dtype), t)

    carrier_frequency = carrier_frequency.to(dtype=dtype, device=device)
    modulation_index = modulation_index.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)
    phase = phase.to(dtype=dtype, device=device)

    if modulator_frequency is not None:
        # Sinusoidal FM mode
        if not isinstance(modulator_frequency, Tensor):
            modulator_frequency = torch.tensor(
                modulator_frequency,
                dtype=dtype or torch.float32,
                device=device,
            )
        modulator_frequency = modulator_frequency.to(
            dtype=dtype, device=device
        )

        result = torch.ops.torchscience.frequency_modulated_wave(
            n,
            t,
            carrier_frequency=carrier_frequency,
            modulator_frequency=modulator_frequency,
            modulation_index=modulation_index,
            sample_rate=sample_rate,
            amplitude=amplitude,
            phase=phase,
            dtype=dtype,
            layout=layout,
            device=device,
        )
    else:
        # Arbitrary modulating signal FM mode
        modulating_signal = modulating_signal.to(dtype=dtype, device=device)

        result = torch.ops.torchscience.frequency_modulated_wave_arbitrary(
            n,
            t,
            carrier_frequency=carrier_frequency,
            modulating_signal=modulating_signal,
            modulation_index=modulation_index,
            sample_rate=sample_rate,
            amplitude=amplitude,
            phase=phase,
            dtype=dtype,
            layout=layout,
            device=device,
        )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
