import torch
from torch import Tensor


def lambert_w(k: Tensor, z: Tensor) -> Tensor:
    r"""
    Lambert W function (product logarithm).

    Computes the Lambert W function :math:`W_k(z)` for each element of the
    input tensors.

    Mathematical Definition
    -----------------------
    The Lambert W function :math:`W(z)` is defined as the inverse of the
    function :math:`f(w) = w e^w`:

    .. math::

        W(z) \cdot e^{W(z)} = z

    The function is multi-valued in the complex plane, with infinitely many
    branches labeled by an integer index :math:`k`. The two real-valued
    branches are:

    - :math:`W_0(z)`: Principal branch, real for :math:`z \geq -1/e`
    - :math:`W_{-1}(z)`: Secondary branch, real for :math:`-1/e \leq z < 0`

    Domain
    ------
    - For real inputs:
        - :math:`k = 0` (principal branch): :math:`z \geq -1/e \approx -0.3679`
        - :math:`k = -1` (secondary branch): :math:`-1/e \leq z < 0`
    - For complex inputs: All :math:`z \in \mathbb{C}`, any integer :math:`k`

    Special Values
    --------------
    - :math:`W(0) = 0` for all branches
    - :math:`W_0(e) = 1`
    - :math:`W_0(-1/e) = -1` (branch point)
    - :math:`W_{-1}(-1/e) = -1` (branch point)
    - :math:`W_0(1) \approx 0.5671`

    Algorithm
    ---------
    Uses Halley's method with carefully chosen initial approximations:

    - Near :math:`z = 0`: :math:`W(z) \approx z - z^2 + \frac{3z^3}{2}`
    - For large :math:`z`: :math:`W(z) \approx \ln(z) - \ln(\ln(z))`
    - Near branch point: :math:`W(z) \approx -1 + \sqrt{2(ez + 1)}`

    Halley's iteration provides cubic convergence:

    .. math::

        w_{n+1} = w_n - \frac{w_n e^{w_n} - z}{e^{w_n}(w_n + 1) -
        \frac{(w_n + 2)(w_n e^{w_n} - z)}{2w_n + 2}}

    Applications
    ------------
    The Lambert W function appears in many mathematical and scientific contexts:

    - Solutions to equations of the form :math:`x = a e^{bx}`
    - Combinatorics (tree enumeration)
    - Physics (Wien displacement law, quantum mechanics)
    - Enzyme kinetics (Michaelis-Menten equation)
    - Electrical engineering (diode equation)
    - Delay differential equations

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The derivative with respect to :math:`z` is:

    .. math::

        \frac{d}{dz} W(z) = \frac{W(z)}{z (1 + W(z))} \quad \text{for } z \neq 0

    .. math::

        \frac{d}{dz} W(z) \Big|_{z=0} = 1

    Note that :math:`k` is a discrete branch index, so its gradient is always zero.

    Second-order derivatives (gradgradcheck) are also supported:

    .. math::

        \frac{d^2}{dz^2} W(z) = -\frac{W(z)(2 + W(z))}{z^2 (1 + W(z))^3}

    Parameters
    ----------
    k : Tensor
        Branch index. Common values are:
        - 0: Principal branch (most common, default in most libraries)
        - -1: Secondary real branch
        Values are rounded to the nearest integer internally.
    z : Tensor
        Input argument. Broadcasting with k is supported.

    Returns
    -------
    Tensor
        The Lambert W function :math:`W_k(z)` evaluated at the input values.
        Returns NaN for values outside the domain of the specified branch
        (for real inputs).

    Examples
    --------
    Principal branch (k=0):

    >>> import torch
    >>> from torchscience.special_functions import lambert_w
    >>> k = torch.tensor([0.0])
    >>> z = torch.tensor([1.0])
    >>> lambert_w(k, z)
    tensor([0.5671])

    W(e) = 1:

    >>> import math
    >>> k = torch.tensor([0.0])
    >>> z = torch.tensor([math.e])
    >>> lambert_w(k, z)
    tensor([1.0000])

    Secondary branch (k=-1):

    >>> k = torch.tensor([-1.0])
    >>> z = torch.tensor([-0.1])
    >>> lambert_w(k, z)
    tensor([-3.5772])

    Branch point W(-1/e) = -1:

    >>> k = torch.tensor([0.0])
    >>> z = torch.tensor([-1.0 / math.e])
    >>> lambert_w(k, z)
    tensor([-1.0000])

    Verification: W(z) * exp(W(z)) = z:

    >>> k = torch.tensor([0.0])
    >>> z = torch.tensor([2.0])
    >>> w = lambert_w(k, z)
    >>> w * torch.exp(w)
    tensor([2.0000])

    Autograd:

    >>> k = torch.tensor([0.0])
    >>> z = torch.tensor([1.0], requires_grad=True)
    >>> y = lambert_w(k, z)
    >>> y.backward()
    >>> z.grad
    tensor([0.3618])

    Complex input:

    >>> k = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 1.0j])
    >>> lambert_w(k, z)
    tensor([0.6562+0.4324j])

    .. warning:: Branch selection

       For real negative inputs in the range :math:`[-1/e, 0)`, both branches
       :math:`k=0` and :math:`k=-1` give valid real results. The principal
       branch :math:`W_0` returns values in :math:`[-1, 0)`, while the secondary
       branch :math:`W_{-1}` returns values in :math:`(-\infty, -1]`.

    .. warning:: Singularity at w = -1

       The derivative is singular at the branch point :math:`z = -1/e` where
       :math:`W = -1`. Gradient computations near this point may be inaccurate.

    Notes
    -----
    - The function uses Halley's method which typically converges in 3-5
      iterations for double precision.
    - For computing :math:`x = W(z) e^{W(z)/2}` (which equals :math:`\sqrt{z}
      e^{W(z)/2}`), consider using the identity directly for better numerical
      stability.

    See Also
    --------
    scipy.special.lambertw : SciPy's Lambert W function
    """
    return torch.ops.torchscience.lambert_w(k, z)
