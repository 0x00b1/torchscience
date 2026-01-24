from typing import Sequence, Union

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def shot_noise(
    size: Sequence[int],
    rate: Union[float, Tensor],
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
    Generate differentiable shot noise using Gaussian approximation.

    Shot noise models the statistical fluctuations in discrete counting
    processes (like photon detection) using a continuous, differentiable
    approximation. This enables gradient-based optimization in simulation
    and imaging pipelines.

    Mathematical Definition
    -----------------------
    Shot noise is generated using the Gaussian approximation to Poisson:

        X ~ max(0, N(λ, √λ))

    where λ is the rate parameter. For large λ (≥10), this closely
    approximates the true Poisson distribution. For smaller λ, it provides
    a differentiable relaxation.

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. All dimensions are treated as independent
        sample dimensions.
    rate : float or Tensor
        Rate parameter (λ) representing the expected count. Must be
        non-negative. If a tensor, it is broadcast with `size` to allow
        spatially-varying rates across the output.
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
        A tensor of shape `size` containing continuous shot noise samples.
        Mean ≈ rate, variance ≈ rate (clamped to non-negative values).

    Examples
    --------
    Generate shot noise with rate 100 (good Gaussian approximation):

    >>> noise = shot_noise([1000], rate=100.0)
    >>> noise.mean()  # approximately 100
    tensor(99.8765)

    Generate differentiable shot noise for training:

    >>> noise = shot_noise([100], rate=50.0, requires_grad=True)
    >>> loss = noise.sum()
    >>> loss.backward()  # gradients flow through

    Generate with spatially-varying rates:

    >>> rates = torch.tensor([[10.0, 50.0], [100.0, 200.0]])
    >>> noise = shot_noise([2, 2], rate=rates)
    >>> noise.shape
    torch.Size([2, 2])

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    poisson_noise : Discrete Poisson noise (not differentiable)
    white_noise : Gaussian white noise with mean 0

    Notes
    -----
    Differentiability
    ^^^^^^^^^^^^^^^^^
    Unlike `poisson_noise`, `shot_noise` is differentiable with respect to
    both the output and the rate parameter. Gradients flow through the
    Gaussian reparameterization and the ReLU clamping.

    Gaussian Approximation
    ^^^^^^^^^^^^^^^^^^^^^^
    The approximation X ~ N(λ, √λ) is:
    - Excellent for λ ≥ 10 (relative error < 1%)
    - Reasonable for λ ≥ 5 (relative error < 5%)
    - Poor but differentiable for λ < 5

    For applications requiring accurate low-rate statistics, consider using
    `poisson_noise` for evaluation and `shot_noise` for training.

    Non-Negativity
    ^^^^^^^^^^^^^^
    The output is clamped to non-negative values using ReLU, which is
    subgradient differentiable. This maintains the physical constraint
    that counts cannot be negative.

    Physical Applications
    ^^^^^^^^^^^^^^^^^^^^^
    - Differentiable photon-counting simulation
    - Noise-aware image reconstruction training
    - Poisson image denoising networks
    - Low-light imaging simulation

    References
    ----------
    D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes,"
    arXiv:1312.6114, 2013. (Reparameterization trick)

    E. Hasinoff et al., "Noise-optimal capture for high dynamic range
    photography," CVPR 2010. (Shot noise in imaging)
    """
    # Convert scalar rate to tensor
    if not isinstance(rate, Tensor):
        rate_tensor = torch.tensor(rate, dtype=torch.float32)
    else:
        rate_tensor = rate

    return torch.ops.torchscience.shot_noise(
        size,
        rate_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
