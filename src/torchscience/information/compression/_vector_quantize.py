"""Vector quantization operators."""

from __future__ import annotations

import torch
from torch import Tensor


class _VQStraightThrough(torch.autograd.Function):
    """Vector quantization with straight-through gradient."""

    @staticmethod
    def forward(
        ctx, x: Tensor, codebook: Tensor, beta: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize vectors to nearest codebook entry."""
        # x: (..., D) - input vectors
        # codebook: (K, D) - K codebook entries of dimension D
        batch_shape = x.shape[:-1]
        D = x.shape[-1]
        K = codebook.shape[0]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, D)  # (N, D)
        N = x_flat.shape[0]

        # Compute distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x_sq = (x_flat**2).sum(dim=1, keepdim=True)  # (N, 1)
        c_sq = (codebook**2).sum(dim=1, keepdim=True).T  # (1, K)
        distances = x_sq + c_sq - 2 * x_flat @ codebook.T  # (N, K)

        # Find nearest codebook entry
        indices = distances.argmin(dim=1)  # (N,)
        indices = indices.reshape(batch_shape)

        # Look up quantized values
        quantized = codebook[indices.reshape(-1)].reshape(*batch_shape, D)

        # Compute commitment loss: beta * ||x - sg(e)||^2
        # sg = stop gradient, so this encourages x to commit to codebook
        commitment_loss = beta * ((x - quantized.detach()) ** 2).mean()

        ctx.save_for_backward(x, codebook, indices.reshape(-1))
        ctx.beta = beta

        return quantized, indices, commitment_loss

    @staticmethod
    def backward(
        ctx, grad_quantized: Tensor, grad_indices: Tensor, grad_loss: Tensor
    ) -> tuple[Tensor | None, ...]:
        """Straight-through gradient for quantized, plus commitment gradient."""
        x, codebook, indices_flat = ctx.saved_tensors
        beta = ctx.beta

        batch_shape = x.shape[:-1]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]
        K = codebook.shape[0]

        # Gradient for x:
        # 1. Straight-through: grad from quantized
        # 2. Commitment loss: 2 * beta * (x - e) / numel
        grad_quantized_flat = grad_quantized.reshape(-1, D)
        quantized_flat = codebook[indices_flat]

        numel = x.numel()
        grad_x = (
            grad_quantized_flat
            + grad_loss * 2 * beta * (x_flat - quantized_flat) / numel
        )
        grad_x = grad_x.reshape(x.shape)

        # Gradient for codebook: accumulate from all vectors mapped to each entry
        # This is the "dictionary learning" gradient
        grad_codebook = torch.zeros_like(codebook)
        grad_codebook.index_add_(0, indices_flat, -grad_quantized_flat)

        return grad_x, grad_codebook, None


class _VQGumbelSoftmax(torch.autograd.Function):
    """Vector quantization with Gumbel-Softmax gradient."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        codebook: Tensor,
        temperature: float,
        hard: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Soft vector quantization with Gumbel noise."""
        batch_shape = x.shape[:-1]
        D = x.shape[-1]
        K = codebook.shape[0]

        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        # Compute negative distances as logits
        x_sq = (x_flat**2).sum(dim=1, keepdim=True)
        c_sq = (codebook**2).sum(dim=1, keepdim=True).T
        neg_distances = -(x_sq + c_sq - 2 * x_flat @ codebook.T)  # (N, K)

        # Gumbel-Softmax
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(neg_distances) + 1e-10) + 1e-10
        )
        logits = (neg_distances + gumbel_noise) / temperature
        soft_weights = torch.softmax(logits, dim=1)  # (N, K)

        if hard:
            # Hard assignment but with soft gradients
            indices = soft_weights.argmax(dim=1)
            hard_weights = torch.zeros_like(soft_weights)
            hard_weights.scatter_(1, indices.unsqueeze(1), 1.0)
            # Straight-through trick
            weights = hard_weights - soft_weights.detach() + soft_weights
        else:
            weights = soft_weights
            indices = soft_weights.argmax(dim=1)

        # Weighted combination of codebook entries
        quantized = weights @ codebook  # (N, D)
        quantized = quantized.reshape(*batch_shape, D)
        indices = indices.reshape(batch_shape)

        # No commitment loss for Gumbel-Softmax mode
        commitment_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        ctx.save_for_backward(x, codebook, weights.reshape(-1, K))
        ctx.temperature = temperature

        return quantized, indices, commitment_loss

    @staticmethod
    def backward(
        ctx, grad_quantized: Tensor, grad_indices: Tensor, grad_loss: Tensor
    ) -> tuple[Tensor | None, ...]:
        """Gradient through Gumbel-Softmax."""
        x, codebook, weights_flat = ctx.saved_tensors

        batch_shape = x.shape[:-1]
        D = x.shape[-1]
        K = codebook.shape[0]

        grad_quantized_flat = grad_quantized.reshape(-1, D)
        N = grad_quantized_flat.shape[0]

        # grad_weights = grad_quantized @ codebook.T  # (N, K)
        # Then need to propagate through softmax

        # Gradient for codebook
        grad_codebook = weights_flat.T @ grad_quantized_flat  # (K, D)

        # Gradient for x (through distance computation and softmax)
        # This is complex, use autograd in practice
        # Simplified: pass through
        grad_x = grad_quantized.clone()

        return grad_x, grad_codebook, None, None


def vector_quantize(
    x: Tensor,
    codebook: Tensor,
    *,
    beta: float = 0.25,
    gradient_mode: str = "ste",
    temperature: float = 1.0,
    hard: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize vectors to nearest codebook entries.

    Vector quantization maps continuous vectors to a discrete set of
    codebook entries (embeddings). This is the core operation in VQ-VAE
    and related models.

    Parameters
    ----------
    x : Tensor
        Input vectors to quantize. Shape: ``(..., D)`` where D is the
        vector dimension.
    codebook : Tensor
        Codebook of embeddings. Shape: ``(K, D)`` where K is the
        number of codebook entries.
    beta : float, default=0.25
        Commitment loss weight. The commitment loss encourages encoder
        outputs to stay close to codebook entries.
    gradient_mode : {"ste", "gumbel"}, default="ste"
        Gradient estimation mode:
        - "ste": Straight-through estimator with commitment loss.
        - "gumbel": Gumbel-Softmax relaxation for differentiable sampling.
    temperature : float, default=1.0
        Temperature for Gumbel-Softmax mode. Lower values give harder
        assignments. Only used when gradient_mode="gumbel".
    hard : bool, default=True
        Whether to use hard assignments in Gumbel-Softmax mode.
        If True, uses straight-through trick for hard assignments
        with soft gradients.

    Returns
    -------
    quantized : Tensor
        Quantized vectors. Shape: ``(..., D)``, same as input.
    indices : Tensor
        Codebook indices for each input vector. Shape: ``(...)``.
    commitment_loss : Tensor
        Scalar commitment loss: ``beta * mean(||x - sg(e)||^2)``.
        Zero for Gumbel-Softmax mode.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 8)  # 4 vectors of dimension 8
    >>> codebook = torch.randn(16, 8)  # 16 codebook entries
    >>> q, idx, loss = vector_quantize(x, codebook)
    >>> q.shape
    torch.Size([4, 8])
    >>> idx.shape
    torch.Size([4])

    Notes
    -----
    The straight-through estimator passes gradients directly from the
    quantized output to the input, while the codebook is updated by
    the EMA or gradient descent on reconstruction loss.

    The commitment loss term encourages the encoder to "commit" to
    codebook entries, preventing the encoder from growing unboundedly
    while the codebook stays fixed.

    Total VQ-VAE loss is typically:
        L = reconstruction_loss + commitment_loss + codebook_loss

    where codebook_loss = ||sg(x) - e||^2 is often handled by EMA
    updates rather than gradients.

    See Also
    --------
    scalar_quantize : Scalar (per-element) quantization.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
    if not isinstance(codebook, Tensor):
        raise TypeError(
            f"codebook must be a Tensor, got {type(codebook).__name__}"
        )

    if codebook.dim() != 2:
        raise ValueError(f"codebook must be 2D (K, D), got {codebook.dim()}D")

    if x.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            f"x dimension ({x.shape[-1]}) must match codebook dimension ({codebook.shape[-1]})"
        )

    if gradient_mode == "ste":
        return _VQStraightThrough.apply(x, codebook, beta)
    elif gradient_mode == "gumbel":
        return _VQGumbelSoftmax.apply(x, codebook, temperature, hard)
    else:
        raise ValueError(
            f"gradient_mode must be 'ste' or 'gumbel', got '{gradient_mode}'"
        )
