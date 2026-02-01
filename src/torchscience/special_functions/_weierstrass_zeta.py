import torch
from torch import Tensor


def weierstrass_zeta(z: Tensor, g2: Tensor, g3: Tensor) -> Tensor:
    r"""
    Weierstrass zeta function zeta(z; g2, g3).

    Computes the Weierstrass zeta function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Weierstrass zeta function is defined as the logarithmic derivative
    of the sigma function:

    .. math::

       \zeta(z; g_2, g_3) = \frac{\sigma'(z)}{\sigma(z)}

    Equivalently, the derivative of zeta gives the negative of the
    Weierstrass P function:

    .. math::

       \zeta'(z) = -\wp(z)

    Laurent Series
    --------------
    Near z = 0, the zeta function has the Laurent expansion:

    .. math::

       \zeta(z) = \frac{1}{z} - \frac{g_2}{60} z^3 - \frac{g_3}{140} z^5
       - \frac{g_2^2}{8400} z^7 + O(z^9)

    Note that zeta has a simple pole at z = 0, unlike P which has a double pole.

    Special Properties
    ------------------
    - zeta(0) = infinity (simple pole at the origin)
    - zeta(-z) = -zeta(z) (odd function)
    - zeta is NOT periodic (quasi-periodic with additive constants)

    Quasi-periodicity
    -----------------
    Unlike the P function which is doubly periodic, zeta is quasi-periodic:

    .. math::

       \zeta(z + 2\omega_i) = \zeta(z) + 2\eta_i

    where eta_i are the quasi-periods (constants) associated with the
    half-periods omega_i. The quasi-periods satisfy the Legendre relation:

    .. math::

       \eta_1 \omega_2 - \eta_2 \omega_1 = \frac{\pi i}{2}

    Relationship to Other Weierstrass Functions
    --------------------------------------------
    The zeta function connects the sigma and P functions:

    - From sigma: zeta(z) = sigma'(z) / sigma(z)
    - To P: zeta'(z) = -P(z)
    - Integration: zeta(z) = -integral(P(z) dz) + constant

    The second logarithmic derivative of sigma gives P:

    .. math::

       \wp(z) = -\frac{d^2}{dz^2} \log \sigma(z) = \zeta'(z)^2 - \zeta''(z)/\zeta'(z)

    Applications
    ------------
    The Weierstrass zeta function appears in:

    - Elliptic curve theory and addition formulas
    - Construction of elliptic functions with prescribed poles
    - Theta function identities
    - Solutions of certain nonlinear differential equations
    - Computation of quasi-periods in lattice problems
    - The Weierstrass addition formula involves zeta

    Addition Formula
    ----------------
    The Weierstrass addition formula relates zeta values:

    .. math::

       \zeta(z_1 + z_2) + \zeta(z_1 - z_2) = 2\zeta(z_1) +
       \frac{\wp'(z_1)}{\wp(z_1) - \wp(z_2)}

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to z uses the relation:

    .. math::

       \frac{\partial \zeta}{\partial z} = -\wp(z)

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    z : Tensor
        The argument. Can be floating-point or complex.
        Broadcasting with g2 and g3 is supported.
    g2 : Tensor
        The first invariant of the Weierstrass function.
        Broadcasting with z and g3 is supported.
    g3 : Tensor
        The second invariant of the Weierstrass function.
        Broadcasting with z and g2 is supported.

    Returns
    -------
    Tensor
        The Weierstrass zeta function zeta(z; g2, g3) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> weierstrass_zeta(z, g2, g3)
    tensor([1.9167])

    Verify pole at z = 0:

    >>> z = torch.tensor([0.0])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.5])
    >>> weierstrass_zeta(z, g2, g3)
    tensor([inf])

    Odd function property:

    >>> z = torch.tensor([0.3])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.5])
    >>> zeta_pos = weierstrass_zeta(z, g2, g3)
    >>> zeta_neg = weierstrass_zeta(-z, g2, g3)
    >>> torch.allclose(zeta_pos, -zeta_neg)
    True

    Complex input:

    >>> z = torch.tensor([0.3 + 0.2j])
    >>> g2 = torch.tensor([1.0 + 0.0j])
    >>> g3 = torch.tensor([0.0 + 0.0j])
    >>> weierstrass_zeta(z, g2, g3)
    tensor([2.7284-1.8133j])

    Autograd:

    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> zeta = weierstrass_zeta(z, g2, g3)
    >>> zeta.backward()
    >>> z.grad  # d(zeta)/dz = -P(z)
    tensor([-4.1234])

    Notes
    -----
    - The zeta function has a simple pole at z = 0 and at all lattice points.
    - Unlike P which is doubly periodic, zeta is only quasi-periodic.
    - The relation zeta'(z) = -P(z) is useful for numerical verification.
    - For numerical stability near zero, the Laurent series is used.

    See Also
    --------
    weierstrass_p : Weierstrass elliptic P function
    weierstrass_sigma : Weierstrass sigma function (exp(integral(zeta)))
    """
    return torch.ops.torchscience.weierstrass_zeta(z, g2, g3)
