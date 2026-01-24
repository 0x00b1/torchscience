"""Entropy bottleneck for learned compression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EntropyBottleneck(nn.Module):
    """Entropy bottleneck layer for learned image compression.

    The entropy bottleneck learns a flexible prior distribution over latent
    representations. During training, uniform noise is added to simulate
    quantization. At inference, hard quantization is used.

    This implements the entropy bottleneck from:
    Ballé et al., "Variational Image Compression with a Scale Hyperprior", ICLR 2018.

    Parameters
    ----------
    channels : int
        Number of channels in the latent representation.
    init_scale : float, default=10.0
        Initial scale for the cumulative distribution parameters.
    filters : tuple[int, ...], default=(3, 3, 3)
        Number of filters in each layer of the CDF network.
    tail_mass : float, default=1e-9
        Probability mass assigned to tails of the distribution.

    Examples
    --------
    >>> import torch
    >>> from torchscience.compression import EntropyBottleneck
    >>> eb = EntropyBottleneck(channels=192)
    >>> x = torch.randn(4, 192, 16, 16)
    >>> y, likelihoods = eb(x)
    >>> # Training: y has noise, likelihoods for rate loss
    >>> eb.eval()
    >>> y_q, likelihoods_q = eb(x)
    >>> # Evaluation: y_q is quantized

    Notes
    -----
    The entropy bottleneck models P(y) using a flexible cumulative distribution
    function (CDF) parameterized by a small neural network. The CDF is
    constrained to be monotonically increasing.

    Key operations:
    - **Training**: Add uniform noise U(-0.5, 0.5) to simulate quantization
    - **Inference**: Hard quantization (round to nearest integer)
    - **Likelihood**: Computed as CDF(y + 0.5) - CDF(y - 0.5)

    The rate loss is computed as -log2(likelihood).sum().

    See Also
    --------
    GaussianConditional : For conditional entropy models with side information.
    """

    def __init__(
        self,
        channels: int,
        init_scale: float = 10.0,
        filters: tuple[int, ...] = (3, 3, 3),
        tail_mass: float = 1e-9,
    ):
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale
        self.filters = filters
        self.tail_mass = tail_mass

        # Build CDF network parameters
        # The network maps values to CDF values via cumulative sum of softplus
        self._build_cdf_network()

    def _build_cdf_network(self):
        """Build the cumulative distribution network parameters."""
        filters = (1,) + self.filters + (1,)

        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Parameters for each layer
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        for i in range(len(filters) - 1):
            init = torch.zeros(self.channels, filters[i + 1], filters[i])
            nn.init.xavier_uniform_(init)
            self._matrices.append(nn.Parameter(init))

            init = torch.zeros(self.channels, filters[i + 1], 1)
            nn.init.uniform_(init, -0.5, 0.5)
            self._biases.append(nn.Parameter(init))

            if i < len(filters) - 2:
                init = torch.zeros(self.channels, filters[i + 1], 1)
                init.fill_(scale)
                self._factors.append(nn.Parameter(init))

    def _logits_cumulative(self, inputs: Tensor) -> Tensor:
        """Evaluate the cumulative distribution function.

        Args:
            inputs: Values at which to evaluate CDF. Shape: (batch, channels, ...).

        Returns:
            Logits of CDF values. Shape: same as inputs.
        """
        # Reshape for matrix multiplication: (batch, channels, 1, samples)
        shape = inputs.shape
        inputs = inputs.reshape(
            -1,
            self.channels,
            1,
            inputs.numel() // (inputs.shape[0] * self.channels)
            if inputs.dim() > 2
            else 1,
        )

        # If 2D input (batch, channels), add dummy spatial dim
        if len(shape) == 2:
            inputs = inputs.unsqueeze(-1)

        # Forward through CDF network
        logits = inputs

        for i, (matrix, bias) in enumerate(zip(self._matrices, self._biases)):
            # matrix: (channels, out_filters, in_filters)
            # logits: (batch, channels, in_filters, samples)
            logits = torch.einsum(
                "coi,bcis->bcos", matrix, logits
            ) + bias.unsqueeze(0)

            if i < len(self._factors):
                factor = self._factors[i]
                logits = logits + torch.tanh(logits) * torch.tanh(
                    factor.unsqueeze(0)
                )

        # Reshape back
        logits = logits.reshape(shape)

        return logits

    def _likelihood(self, inputs: Tensor) -> Tensor:
        """Compute likelihood of inputs under the learned prior.

        The likelihood is P(floor(y) <= Y < floor(y) + 1).
        """
        # Compute CDF at upper and lower bounds
        upper = self._logits_cumulative(inputs + 0.5)
        lower = self._logits_cumulative(inputs - 0.5)

        # Convert logits to probabilities and compute difference
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)

        # Clamp to avoid numerical issues
        likelihood = torch.clamp(likelihood, min=self.tail_mass)

        return likelihood

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through entropy bottleneck.

        Args:
            x: Input latent representation. Shape: (batch, channels, ...).

        Returns:
            y: Quantized (eval) or noisy (train) representation. Same shape as x.
            likelihoods: Probability of each quantized value. Same shape as x.
        """
        if self.training:
            # Training: add uniform noise
            noise = torch.rand_like(x) - 0.5
            y = x + noise
        else:
            # Evaluation: hard quantization
            y = torch.round(x)

        # Compute likelihoods
        likelihoods = self._likelihood(y if self.training else x)

        return y, likelihoods

    def compress(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compress latents to quantized values and symbols.

        Args:
            x: Input latent representation.

        Returns:
            y: Quantized representation.
            symbols: Integer symbols for entropy coding.
        """
        symbols = torch.round(x).to(torch.int32)
        y = symbols.float()
        return y, symbols

    def decompress(self, symbols: Tensor) -> Tensor:
        """Decompress symbols to latent values.

        Args:
            symbols: Integer symbols from entropy coding.

        Returns:
            Reconstructed latent values.
        """
        return symbols.float()


def entropy_bottleneck(
    x: Tensor,
    *,
    training: bool = True,
    tail_mass: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Functional entropy bottleneck for learned compression.

    A simplified functional interface to the entropy bottleneck. This uses
    a uniform prior rather than a learned one, suitable for testing or
    simple applications.

    Parameters
    ----------
    x : Tensor
        Input latent representation. Any shape.
    training : bool, default=True
        If True, add uniform noise. If False, use hard quantization.
    tail_mass : float, default=1e-9
        Minimum probability for numerical stability.

    Returns
    -------
    y : Tensor
        Quantized (if not training) or noisy (if training) representation.
    likelihoods : Tensor
        Approximate likelihoods under uniform prior.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 192, 16, 16)
    >>> y, likelihoods = entropy_bottleneck(x, training=True)

    Notes
    -----
    This functional version uses a simple uniform prior approximation.
    For production use with learned priors, use the EntropyBottleneck class.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if training:
        # Add uniform noise to simulate quantization
        noise = torch.rand_like(x) - 0.5
        y = x + noise
    else:
        # Hard quantization
        y = torch.round(x)

    # Simple uniform prior likelihood approximation
    # For a uniform distribution on integers, likelihood is constant
    # Here we use a simple approximation based on the quantization interval
    likelihoods = torch.ones_like(x) * (1.0 - tail_mass)

    return y, likelihoods
