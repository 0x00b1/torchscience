import torch
from torch import Tensor


def parabolic_cylinder_u(a: Tensor, z: Tensor) -> Tensor:
    r"""
    Parabolic cylinder function U(a, z).

    Computes the parabolic cylinder function U(a, z), the standard
    solution to Weber's differential equation that is recessive
    (decays) as z → +∞.

    Mathematical Definition
    -----------------------
    U(a, z) satisfies Weber's differential equation:

    .. math::

       \frac{d^2 w}{dz^2} - \left(\frac{z^2}{4} + a\right) w = 0

    It is the solution that decays exponentially as z → +∞.

    Parameters
    ----------
    a : Tensor
        Order parameter. Can be any real or complex number.
    z : Tensor
        Argument at which to evaluate the function.

    Returns
    -------
    Tensor
        The parabolic cylinder function U(a, z) evaluated at the input values.

    See Also
    --------
    parabolic_cylinder_v : Dominant solution V(a, z)

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Chapter 12.
           https://dlmf.nist.gov/12
    """
    return torch.ops.torchscience.parabolic_cylinder_u(a, z)
