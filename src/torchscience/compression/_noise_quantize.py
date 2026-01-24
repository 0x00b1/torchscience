"""Noise-based quantization approximation for training."""

from __future__ import annotations

import torch
from torch import Tensor


def noise_quantize(
    x: Tensor,
    *,
    training: bool = True,
    noise_type: str = "uniform",
) -> Tensor:
    """Approximate quantization with additive noise for training.

    During training, quantization is approximated by adding noise from
    a distribution that matches the quantization error distribution.
    During inference, standard rounding is used.

    Parameters
    ----------
    x : Tensor
        Input tensor to quantize. Any shape.
    training : bool, default=True
        If True, add noise. If False, use hard quantization.
    noise_type : {"uniform", "triangular"}, default="uniform"
        Type of noise to add during training:
        - "uniform": U(-0.5, 0.5) - standard choice, matches quantization error.
        - "triangular": Triangular on [-1, 1] - smoother gradients.

    Returns
    -------
    Tensor
        Quantized (inference) or noisy (training) values. Same shape as input.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 64, 8, 8)
    >>> # Training: add noise
    >>> y_train = noise_quantize(x, training=True)
    >>> # Inference: round to integers
    >>> y_eval = noise_quantize(x, training=False)

    Notes
    -----
    Noise-based quantization is a key technique for training neural
    compression models. The idea is that uniform noise U(-0.5, 0.5)
    has the same first and second moments as rounding error, making
    it a good differentiable proxy.

    During training:
    - Forward: y = x + noise
    - Backward: dy/dx = 1 (gradients pass through)

    This provides unbiased gradient estimates for the rate-distortion
    objective, enabling end-to-end training.

    Triangular noise can provide smoother gradients and is sometimes
    preferred in audio compression.

    See Also
    --------
    scalar_quantize : Scalar quantization with STE gradients.
    entropy_bottleneck : Full entropy model with noise training.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    valid_noise = {"uniform", "triangular"}
    if noise_type not in valid_noise:
        raise ValueError(
            f"noise_type must be one of {valid_noise}, got '{noise_type}'"
        )

    if training:
        if noise_type == "uniform":
            # Uniform noise on [-0.5, 0.5)
            noise = torch.rand_like(x) - 0.5
        else:  # triangular
            # Triangular noise: sum of two uniform
            u1 = torch.rand_like(x) - 0.5
            u2 = torch.rand_like(x) - 0.5
            noise = u1 + u2  # Range: [-1, 1] with triangular PDF
            noise = noise / 2  # Scale to [-0.5, 0.5]

        return x + noise
    else:
        # Hard quantization
        return torch.round(x)


def ste_round(x: Tensor) -> Tensor:
    """Round with straight-through estimator gradient.

    Performs hard rounding in the forward pass but passes gradients
    through unchanged in the backward pass.

    Parameters
    ----------
    x : Tensor
        Input tensor to round.

    Returns
    -------
    Tensor
        Rounded values with STE gradients.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([1.2, 2.7, -0.3], requires_grad=True)
    >>> y = ste_round(x)
    >>> y.backward(torch.ones_like(y))
    >>> x.grad  # Gradients pass through unchanged
    tensor([1., 1., 1.])

    Notes
    -----
    The straight-through estimator (STE) was introduced by Bengio et al.
    for training networks with discrete operations. It's defined as:
    - Forward: y = round(x)
    - Backward: dy/dx = 1

    This is biased but works well in practice for many applications.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    return x + (torch.round(x) - x).detach()


def soft_round(x: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Soft approximation to rounding using temperature-scaled sigmoid.

    Provides a differentiable approximation to rounding that becomes
    sharper as temperature decreases.

    Parameters
    ----------
    x : Tensor
        Input tensor to soft-round.
    temperature : float, default=1.0
        Temperature parameter. Lower = sharper (closer to hard round).

    Returns
    -------
    Tensor
        Soft-rounded values.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(-0.5, 1.5, 5)
    >>> soft_round(x, temperature=0.1)  # Close to hard rounding
    >>> soft_round(x, temperature=10.0)  # Very soft

    Notes
    -----
    Soft rounding is computed as:
        y = floor(x) + sigmoid((x - floor(x) - 0.5) / temperature)

    As temperature -> 0, this approaches hard rounding.
    As temperature -> infinity, this approaches x (identity).

    This can be useful for annealing from soft to hard quantization
    during training.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    floor_x = torch.floor(x)
    frac = x - floor_x
    return floor_x + torch.sigmoid((frac - 0.5) / temperature)
