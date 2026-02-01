import torch
from torch import Tensor


def weierstrass_eta(g2: Tensor, g3: Tensor) -> Tensor:
    r"""
    Weierstrass eta quasi-period eta1 = zeta(omega1).

    Computes the Weierstrass eta quasi-period evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Weierstrass eta quasi-period is defined as:

    .. math::

       \eta_1 = \zeta(\omega_1)

    where omega1 is the first half-period of the Weierstrass functions
    and zeta is the Weierstrass zeta function.

    The quasi-periods satisfy the fundamental relation:

    .. math::

       \zeta(z + 2\omega_i) = \zeta(z) + 2\eta_i

    Legendre Relation
    -----------------
    The quasi-periods and half-periods satisfy the Legendre relation:

    .. math::

       \eta_1 \omega_3 - \eta_3 \omega_1 = \frac{\pi i}{2}

    This is a fundamental identity in the theory of elliptic functions.

    Formula via Theta Functions
    ---------------------------
    The eta quasi-period can be computed using theta functions:

    .. math::

       \eta_1 = -\frac{\pi^2 \theta_1'''(0, q)}{12 \omega_1 \theta_1'(0, q)}

    where q is the nome and theta_1 is the first Jacobi theta function.

    Applications
    ------------
    The Weierstrass eta function appears in:

    - The quasi-periodicity of the Weierstrass zeta function
    - The addition formula for Weierstrass sigma
    - Elliptic curve theory and Abel's theorem
    - Computation of periods and quasi-periods
    - The Legendre relation connecting half-periods

    Relationship to Sigma Function
    ------------------------------
    The sigma function satisfies:

    .. math::

       \sigma(z + 2\omega_i) = -e^{2\eta_i(z + \omega_i)} \sigma(z)

    This shows how eta controls the quasi-periodicity of sigma.

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using numerical differentiation.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    g2 : Tensor
        The first invariant of the Weierstrass function.
        Broadcasting with g3 is supported.
    g3 : Tensor
        The second invariant of the Weierstrass function.
        Broadcasting with g2 is supported.

    Returns
    -------
    Tensor
        The Weierstrass eta quasi-period eta1(g2, g3) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> weierstrass_eta(g2, g3)
    tensor([1.5708])

    Lemniscatic case (g3=0):

    >>> g2 = torch.tensor([4.0])
    >>> g3 = torch.tensor([0.0])
    >>> eta = weierstrass_eta(g2, g3)
    >>> eta  # Related to elliptic integrals
    tensor([...])

    Broadcasting:

    >>> g2 = torch.tensor([[1.0], [2.0]])
    >>> g3 = torch.tensor([0.0, 0.5, 1.0])
    >>> result = weierstrass_eta(g2, g3)
    >>> result.shape
    torch.Size([2, 3])

    Complex input:

    >>> g2 = torch.tensor([1.0 + 0.0j])
    >>> g3 = torch.tensor([0.5 + 0.1j])
    >>> weierstrass_eta(g2, g3)
    tensor([...])

    Autograd:

    >>> g2 = torch.tensor([1.0], requires_grad=True)
    >>> g3 = torch.tensor([0.5])
    >>> eta = weierstrass_eta(g2, g3)
    >>> eta.backward()
    >>> g2.grad
    tensor([...])

    Notes
    -----
    - The function returns eta1, the quasi-period associated with omega1.
    - For degenerate cases (discriminant = 0), the result may be singular.
    - The Legendre relation can be used to verify numerical accuracy.
    - For numerical stability, theta function derivatives are computed
      using finite differences.

    See Also
    --------
    weierstrass_zeta : Weierstrass zeta function (has quasi-periodicity)
    weierstrass_p : Weierstrass elliptic P function (doubly periodic)
    weierstrass_sigma : Weierstrass sigma function
    """
    return torch.ops.torchscience.weierstrass_eta(g2, g3)
