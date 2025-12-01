from typing import Sequence

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def white_noise(
    size: Sequence[int],
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate white noise with flat power spectral density.

    White noise (also known as Gaussian white noise) has a constant power
    spectral density across all frequencies, meaning all frequencies are
    equally represented. This makes it useful as a baseline for audio
    synthesis, signal processing tests, and as a building block for
    generating colored noise.

    Mathematical Definition
    -----------------------
    White noise samples are independently drawn from a standard normal
    distribution:

        x[n] ~ N(0, 1)

    The power spectral density S(f) is flat:

        S(f) = constant

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. The last dimension is treated as the
        time/sample axis. Other dimensions are batch dimensions generating
        independent noise sequences.
    generator : torch.Generator, optional
        A pseudorandom number generator for sampling. If None, uses the default
        generator.
    out : Tensor, optional
    dtype : torch.dtype, optional
        The desired data type of the returned tensor. If None, uses the default
        floating point type.
    layout : torch.layout, optional
        The desired layout of the returned tensor. Default: torch.strided.
    device : torch.device, optional
        The desired device of the returned tensor. Default: CPU.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients. Default: False.
    pin_memory : bool, optional

    Returns
    -------
    Tensor
        A tensor of shape `size` containing white noise samples with
        zero mean and unit variance.

    Examples
    --------
    Generate 1D white noise with 1000 samples:

    >>> noise = white_noise([1000])
    >>> noise.shape
    torch.Size([1000])

    Generate batched white noise (4 channels, 1000 samples each):

    >>> noise = white_noise([4, 1000])
    >>> noise.shape
    torch.Size([4, 1000])

    Generate reproducible noise using a generator:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = white_noise([100], generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = white_noise([100], generator=g)
    >>> torch.allclose(noise1, noise2)
    True

    Generate on GPU with gradients:

    >>> noise = white_noise([1000], device='cuda', requires_grad=True)  # doctest: +SKIP

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    torch.randn : PyTorch's built-in Gaussian noise generator
    pink_noise : Generate noise with 1/f spectrum
    brown_noise : Generate noise with 1/f^2 spectrum

    Notes
    -----
    Spectral Properties
    ^^^^^^^^^^^^^^^^^^^
    The output has a flat power spectrum, meaning:
    - All frequencies have equal power
    - Samples are statistically independent
    - The autocorrelation is a delta function

    Normalization
    ^^^^^^^^^^^^^
    The output has zero mean and unit variance by construction,
    identical to torch.randn.

    Gradient Support
    ^^^^^^^^^^^^^^^^
    When requires_grad=True, gradients flow through the random number
    generation. This enables use in differentiable pipelines where
    the noise is treated as a fixed random variable.

    References
    ----------
    A. V. Oppenheim and R. W. Schafer, "Discrete-Time Signal Processing,"
    3rd ed. Pearson, 2010.
    """
    return torch.ops.torchscience.white_noise(
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
