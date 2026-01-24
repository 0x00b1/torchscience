"""Kraskov entropy estimator (alias for Kozachenko-Leonenko with k=3)."""

from typing import Optional

from torch import Tensor

from ._kozachenko_leonenko_entropy import kozachenko_leonenko_entropy


def kraskov_entropy(
    samples: Tensor,
    *,
    k: int = 3,
    base: Optional[float] = None,
) -> Tensor:
    r"""Estimate differential entropy using the Kraskov estimator.

    This is an alias for :func:`kozachenko_leonenko_entropy` with default
    ``k=3``, which is the common choice used in the Kraskov-Stögbauer-Grassberger
    (KSG) mutual information estimator.

    Mathematical Definition
    -----------------------
    The estimator is:

    .. math::

        \hat{H} = \psi(n) - \psi(k) + \log(c_d) + \frac{d}{n} \sum_{i=1}^{n} \log(\rho_{k,i})

    where:

    - :math:`n` is the number of samples
    - :math:`k` is the number of nearest neighbors (default 3)
    - :math:`\psi` is the digamma function
    - :math:`c_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}` is the volume of the
      d-dimensional unit ball
    - :math:`\rho_{k,i}` is the distance from sample :math:`i` to its
      :math:`k`-th nearest neighbor
    - :math:`d` is the dimensionality of the data

    Parameters
    ----------
    samples : Tensor
        Input samples. Shape: ``(..., n_samples, n_dims)`` where ``...``
        represents optional batch dimensions, ``n_samples`` is the number
        of samples, and ``n_dims`` is the dimensionality.
    k : int, default=3
        Number of nearest neighbors to use. Default is 3, per the KSG paper.
        Must be at least 1 and less than the number of samples.
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)
        - ``10``: Base-10 logarithm (entropy in dits/hartleys)

    Returns
    -------
    Tensor
        Estimated differential entropy. Shape is the input shape with
        the last two dimensions (n_samples, n_dims) removed.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(42)
    >>> samples = torch.randn(1000, 2)  # 2D Gaussian
    >>> H = kraskov_entropy(samples)
    >>> # For N(0,I_2), true entropy is log(2*pi*e) ~ 2.84

    >>> # Equivalent to kozachenko_leonenko_entropy with k=3
    >>> from torchscience.information import kozachenko_leonenko_entropy
    >>> H_kl = kozachenko_leonenko_entropy(samples, k=3)
    >>> torch.allclose(H, H_kl)
    True

    Notes
    -----
    - This function is an alias for ``kozachenko_leonenko_entropy(samples, k=3)``.
    - The default k=3 was recommended by Kraskov et al. (2004) as a good
      balance between bias and variance.
    - For mutual information estimation, use :func:`kraskov_mutual_information`.

    See Also
    --------
    kozachenko_leonenko_entropy : The underlying estimator.
    kraskov_mutual_information : k-NN mutual information estimator.

    References
    ----------
    .. [1] Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating
           mutual information. Physical Review E, 69(6), 066138.
    """
    return kozachenko_leonenko_entropy(samples, k=k, base=base)
