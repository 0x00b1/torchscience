"""Parks-McClellan (Remez) optimal FIR filter design."""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor


def remez(
    num_taps: int,
    bands: Sequence[float],
    desired: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    filter_type: Literal["bandpass", "differentiator", "hilbert"] = "bandpass",
    maxiter: int = 25,
    grid_density: int = 16,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design an optimal FIR filter using the Parks-McClellan (Remez) algorithm.

    The Parks-McClellan algorithm computes the optimal (in the Chebyshev sense)
    FIR filter that minimizes the maximum error between the desired frequency
    response and the actual frequency response over a set of bands.

    Parameters
    ----------
    num_taps : int
        Number of FIR filter taps (filter length). Must be at least 3.
    bands : sequence of float
        Band edges as pairs [start1, end1, start2, end2, ...]. Frequencies are
        normalized to the Nyquist frequency (0 to 0.5, where 0.5 is Nyquist).
        Must have even length with monotonically increasing values.
    desired : sequence of float
        Desired gain in each band. Length must equal len(bands) // 2.
    weights : sequence of float, optional
        Weight for error in each band. Length must equal len(bands) // 2.
        Default is equal weighting (all ones).
    filter_type : {"bandpass", "differentiator", "hilbert"}, optional
        Type of filter:
        - "bandpass": Standard FIR filter (default)
        - "differentiator": Differentiator with 1/f weighting
        - "hilbert": Hilbert transformer (90-degree phase shift)
    maxiter : int, optional
        Maximum number of Remez exchange iterations. Default is 25.
    grid_density : int, optional
        Grid density for frequency sampling. The number of frequency points
        is approximately grid_density * num_taps. Default is 16.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    h : Tensor
        FIR filter coefficients with shape (num_taps,). The filter has
        linear phase (symmetric coefficients for bandpass, antisymmetric
        for differentiator/hilbert).

    Notes
    -----
    The Remez exchange algorithm iteratively:
    1. Initializes extremal frequencies on a dense grid
    2. Solves for optimal polynomial coefficients via Lagrange interpolation
    3. Finds new extremal points where error is maximized
    4. Repeats until convergence (extremal points stop changing)

    The resulting filter has equiripple behavior: the approximation error
    oscillates between equal-magnitude extrema within each band.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import remez
    >>> # Design a lowpass filter with cutoff at 0.2 * Nyquist
    >>> h = remez(51, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0])
    >>> h.shape
    torch.Size([51])

    >>> # Design a bandpass filter
    >>> h = remez(65, [0.0, 0.1, 0.2, 0.35, 0.4, 0.5], [0.0, 1.0, 0.0])
    """
    # Validate parameters
    if num_taps < 3:
        raise ValueError(f"num_taps must be at least 3, got {num_taps}")

    if len(bands) % 2 != 0:
        raise ValueError(f"bands must have even length, got {len(bands)}")

    num_bands = len(bands) // 2
    if len(desired) != num_bands:
        raise ValueError(
            f"Number of desired values ({len(desired)}) must equal "
            f"number of bands ({num_bands})"
        )

    if weights is not None and len(weights) != num_bands:
        raise ValueError(
            f"Number of weights ({len(weights)}) must equal "
            f"number of bands ({num_bands})"
        )

    # Validate band edges
    for i, freq in enumerate(bands):
        if freq < 0 or freq > 0.5:
            raise ValueError(
                f"Band edge {freq} at index {i} must be between 0 and 0.5"
            )

    for i in range(1, len(bands)):
        if bands[i] < bands[i - 1]:
            raise ValueError(
                f"Band edges must be monotonically increasing. "
                f"Got {bands[i]} after {bands[i - 1]} at index {i}"
            )

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert to numpy/scipy and use their implementation,
    # then convert back to torch tensor
    # This ensures numerical correctness while we work on a pure-torch version
    try:
        from scipy.signal import remez as scipy_remez

        # Map filter_type
        type_map = {
            "bandpass": "bandpass",
            "differentiator": "differentiator",
            "hilbert": "hilbert",
        }

        # Call scipy with fs=1.0 (our bands are already normalized to [0, 0.5])
        h_np = scipy_remez(
            numtaps=num_taps,
            bands=list(bands),
            desired=list(desired),
            weight=list(weights) if weights is not None else None,
            type=type_map[filter_type],
            maxiter=maxiter,
            grid_density=grid_density,
            fs=1.0,
        )

        return torch.from_numpy(h_np).to(dtype=dtype, device=device)

    except ImportError:
        # Fall back to pure PyTorch implementation
        pass

    # Pure PyTorch implementation (fallback)
    bands_tensor = torch.tensor(bands, dtype=torch.float64, device=device)
    desired_tensor = torch.tensor(desired, dtype=torch.float64, device=device)

    if weights is None:
        weights_tensor = torch.ones(
            num_bands, dtype=torch.float64, device=device
        )
    else:
        weights_tensor = torch.tensor(
            weights, dtype=torch.float64, device=device
        )

    # Determine filter symmetry type
    if filter_type == "bandpass":
        if num_taps % 2 == 1:
            ftype = 1  # Type I: odd, symmetric
        else:
            ftype = 2  # Type II: even, symmetric
    else:  # differentiator or hilbert
        if num_taps % 2 == 1:
            ftype = 3  # Type III: odd, antisymmetric
        else:
            ftype = 4  # Type IV: even, antisymmetric

    h = _remez_core(
        num_taps=num_taps,
        bands=bands_tensor,
        desired=desired_tensor,
        weights=weights_tensor,
        filter_type=filter_type,
        ftype=ftype,
        maxiter=maxiter,
        grid_density=grid_density,
        device=device,
    )

    return h.to(dtype)


def _remez_core(
    num_taps: int,
    bands: Tensor,
    desired: Tensor,
    weights: Tensor,
    filter_type: str,
    ftype: int,
    maxiter: int,
    grid_density: int,
    device: torch.device,
) -> Tensor:
    """
    Pure PyTorch implementation of the Remez exchange algorithm.

    This is a fallback for when scipy is not available.
    """
    # Number of cosine basis functions
    if ftype == 1:  # Type I: odd, symmetric
        nfcns = (num_taps + 1) // 2
    elif ftype == 2:  # Type II: even, symmetric
        nfcns = num_taps // 2
    elif ftype == 3:  # Type III: odd, antisymmetric
        nfcns = (num_taps - 1) // 2
    else:  # Type IV: even, antisymmetric
        nfcns = num_taps // 2

    # Number of extremal frequencies
    r = nfcns + 1

    # Build frequency grid
    grid, des_grid, wt_grid = _build_grid(
        bands,
        desired,
        weights,
        filter_type,
        ftype,
        num_taps,
        grid_density,
        device,
    )
    ngrid = len(grid)

    # Initialize extremal indices
    iext = torch.linspace(0, ngrid - 1, r, dtype=torch.long, device=device)

    # Remez exchange loop
    converged = False
    for _ in range(maxiter):
        # Compute Lagrange interpolation coefficients
        x = grid[iext]
        ad = _compute_ad(x, r, device)

        # Compute deviation
        dnum = torch.zeros(1, dtype=torch.float64, device=device)
        dden = torch.zeros(1, dtype=torch.float64, device=device)
        sign = 1.0
        for j in range(r):
            dnum = dnum + ad[j] * des_grid[iext[j]]
            dden = dden + sign * ad[j] / wt_grid[iext[j]]
            sign = -sign
        dev = dnum / dden

        # Compute y values (desired + deviation)
        y = torch.zeros(r, dtype=torch.float64, device=device)
        sign = 1.0
        for j in range(r):
            y[j] = des_grid[iext[j]] - sign * dev / wt_grid[iext[j]]
            sign = -sign

        # Compute response on grid using barycentric interpolation
        A = _compute_A_barycentric(grid, x, ad, y, ngrid, r, device)

        # Compute weighted error
        E = wt_grid * (des_grid - A)

        # Find new extremals
        new_iext = _find_extremals_remez(E, ngrid, r, device)

        # Check convergence
        if torch.equal(iext, new_iext):
            converged = True
            break

        iext = new_iext

    if not converged:
        warnings.warn(
            f"Remez algorithm did not converge after {maxiter} iterations. "
            "Try increasing maxiter or relaxing specifications.",
            RuntimeWarning,
        )

    # Convert to impulse response
    h = _grid_to_impulse(grid, A, num_taps, ftype, device)

    return h


def _build_grid(
    bands: Tensor,
    desired: Tensor,
    weights: Tensor,
    filter_type: str,
    ftype: int,
    num_taps: int,
    grid_density: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the dense frequency grid for Remez algorithm."""
    num_bands = len(desired)
    all_freqs = []
    all_des = []
    all_wts = []

    for i in range(num_bands):
        f_low = bands[2 * i].item()
        f_high = bands[2 * i + 1].item()

        if f_high <= f_low:
            continue

        # Number of points for this band
        npts = max(2, int(grid_density * num_taps * (f_high - f_low) / 0.5))
        freq = torch.linspace(
            f_low, f_high, npts, dtype=torch.float64, device=device
        )

        # Desired response
        if filter_type == "differentiator":
            des = desired[i] * freq * 2  # Linear in frequency
        else:
            des = torch.full_like(freq, desired[i])

        # Weights
        if filter_type == "differentiator":
            wt = torch.where(
                freq > 1e-10,
                weights[i] / (freq * 2),
                torch.full_like(freq, weights[i] * 1e10),
            )
        else:
            wt = torch.full_like(freq, weights[i])

        all_freqs.append(freq)
        all_des.append(des)
        all_wts.append(wt)

    grid = torch.cat(all_freqs)
    des_grid = torch.cat(all_des)
    wt_grid = torch.cat(all_wts)

    # Sort by frequency
    sort_idx = torch.argsort(grid)
    grid = grid[sort_idx]
    des_grid = des_grid[sort_idx]
    wt_grid = wt_grid[sort_idx]

    # Transform for filter types II, III, IV
    omega = 2 * math.pi * grid  # Angular frequency

    if ftype == 2:
        # Type II: multiply by cos(omega/2)
        factor = torch.cos(omega / 2)
        factor = torch.clamp(factor.abs(), min=1e-10) * torch.sign(
            factor + 1e-20
        )
        des_grid = des_grid / factor
        wt_grid = wt_grid * factor.abs()
    elif ftype == 3:
        # Type III: multiply by sin(omega)
        factor = torch.sin(omega)
        factor = torch.clamp(factor.abs(), min=1e-10) * torch.sign(
            factor + 1e-20
        )
        des_grid = des_grid / factor
        wt_grid = wt_grid * factor.abs()
    elif ftype == 4:
        # Type IV: multiply by sin(omega/2)
        factor = torch.sin(omega / 2)
        factor = torch.clamp(factor.abs(), min=1e-10) * torch.sign(
            factor + 1e-20
        )
        des_grid = des_grid / factor
        wt_grid = wt_grid * factor.abs()

    # Convert to cosine domain: x = cos(omega)
    x = torch.cos(omega)

    return x, des_grid, wt_grid


def _compute_ad(x: Tensor, r: int, device: torch.device) -> Tensor:
    """Compute Lagrange interpolation coefficients."""
    ad = torch.ones(r, dtype=torch.float64, device=device)
    for j in range(r):
        for k in range(r):
            if k != j:
                ad[j] = ad[j] / (x[j] - x[k])
    return ad


def _compute_A_barycentric(
    grid: Tensor,
    x: Tensor,
    ad: Tensor,
    y: Tensor,
    ngrid: int,
    r: int,
    device: torch.device,
) -> Tensor:
    """Compute approximation using barycentric interpolation."""
    A = torch.zeros(ngrid, dtype=torch.float64, device=device)

    for i in range(ngrid):
        # Check if grid[i] is close to any x[j]
        diffs = (grid[i] - x).abs()
        min_diff, min_idx = diffs.min(dim=0)

        if min_diff < 1e-10:
            A[i] = y[min_idx]
        else:
            numer = torch.zeros(1, dtype=torch.float64, device=device)
            denom = torch.zeros(1, dtype=torch.float64, device=device)
            for j in range(r):
                c = ad[j] / (grid[i] - x[j])
                numer = numer + c * y[j]
                denom = denom + c
            A[i] = numer / denom

    return A


def _find_extremals_remez(
    E: Tensor,
    ngrid: int,
    r: int,
    device: torch.device,
) -> Tensor:
    """Find r extremal points in the error function."""
    E_abs = E.abs()

    # Find all local maxima of |E|
    local_max = torch.zeros(ngrid, dtype=torch.bool, device=device)
    if ngrid > 1:
        local_max[0] = E_abs[0] >= E_abs[1]
        local_max[-1] = E_abs[-1] >= E_abs[-2]
    if ngrid > 2:
        local_max[1:-1] = (E_abs[1:-1] >= E_abs[:-2]) & (
            E_abs[1:-1] >= E_abs[2:]
        )

    candidates = torch.where(local_max)[0]

    if len(candidates) < r:
        # Not enough local maxima, add points with largest |E|
        _, top_idx = torch.topk(E_abs, min(r * 2, ngrid))
        candidates = torch.unique(torch.cat([candidates, top_idx]))
        candidates = torch.sort(candidates)[0]

    if len(candidates) <= r:
        # Pad with evenly spaced if still not enough
        if len(candidates) < r:
            extra = torch.linspace(
                0, ngrid - 1, r, dtype=torch.long, device=device
            )
            candidates = torch.unique(torch.cat([candidates, extra]))
        return torch.sort(candidates[:r])[0]

    # Select r candidates ensuring alternation
    candidates = torch.sort(candidates)[0]
    selected = []
    last_sign = 0

    for idx in candidates.tolist():
        err_val = E[idx].item()
        sign = 1 if err_val > 0 else (-1 if err_val < 0 else last_sign)

        if sign != last_sign or len(selected) == 0:
            selected.append(idx)
            last_sign = sign
            if len(selected) >= r:
                break

    # If not enough with alternation, add by magnitude
    if len(selected) < r:
        remaining = [c.item() for c in candidates if c.item() not in selected]
        remaining.sort(key=lambda i: -E_abs[i].item())
        for idx in remaining:
            selected.append(idx)
            if len(selected) >= r:
                break

    selected = sorted(selected)[:r]
    return torch.tensor(selected, dtype=torch.long, device=device)


def _grid_to_impulse(
    x: Tensor,
    A: Tensor,
    num_taps: int,
    ftype: int,
    device: torch.device,
) -> Tensor:
    """Convert frequency response to impulse response via IDFT."""
    # Use a dense frequency grid for IDFT
    N = max(num_taps * 4, 512)
    omega = torch.linspace(0, math.pi, N, dtype=torch.float64, device=device)
    x_dense = torch.cos(omega)

    # Interpolate A to dense grid
    # Sort x for interpolation (x is cos(omega), so it's decreasing)
    sort_idx = torch.argsort(x, descending=True)
    x_sorted = x[sort_idx]
    A_sorted = A[sort_idx]

    # Linear interpolation
    A_dense = torch.zeros(N, dtype=torch.float64, device=device)
    for i in range(N):
        xi = x_dense[i]
        # Find bracketing indices
        idx = torch.searchsorted(x_sorted.flip(0), xi)
        idx = N - 1 - idx.item() if hasattr(idx, "item") else N - 1 - idx

        if idx <= 0:
            A_dense[i] = A_sorted[0]
        elif idx >= len(x_sorted) - 1:
            A_dense[i] = A_sorted[-1]
        else:
            # Linear interpolation
            x0, x1 = x_sorted[idx], x_sorted[idx + 1]
            A0, A1 = A_sorted[idx], A_sorted[idx + 1]
            t = (xi - x0) / (x1 - x0 + 1e-20)
            A_dense[i] = A0 + t * (A1 - A0)

    # Transform back for filter type
    if ftype == 2:
        A_dense = A_dense * torch.cos(omega / 2)
    elif ftype == 3:
        A_dense = A_dense * torch.sin(omega)
    elif ftype == 4:
        A_dense = A_dense * torch.sin(omega / 2)

    # IDFT to get impulse response
    # H(omega) = sum_{n=0}^{N-1} h[n] * exp(-j*omega*n)
    # For symmetric (Type I/II): H(omega) = h[M] + 2*sum_{k=1}^{M} h[M-k]*cos(k*omega)
    # For antisymmetric (Type III/IV): H(omega) = 2j*sum_{k=1}^{M} h[M-k]*sin(k*omega)

    h = torch.zeros(num_taps, dtype=torch.float64, device=device)

    if ftype == 1:
        # Type I: odd, symmetric
        M = (num_taps - 1) // 2
        # h[M] = (1/pi) * integral_0^pi H(omega) d_omega
        h[M] = torch.trapezoid(A_dense, omega) / math.pi
        for k in range(1, M + 1):
            # h[M-k] = h[M+k] = (2/pi) * integral_0^pi H(omega)*cos(k*omega) d_omega
            coeff = (
                torch.trapezoid(A_dense * torch.cos(k * omega), omega)
                / math.pi
            )
            h[M - k] = coeff
            h[M + k] = coeff

    elif ftype == 2:
        # Type II: even, symmetric
        L = num_taps // 2
        for k in range(L):
            # Coefficients for cos((k+0.5)*omega) basis
            coeff = (
                torch.trapezoid(A_dense * torch.cos((k + 0.5) * omega), omega)
                * 2
                / math.pi
            )
            h[L - 1 - k] = coeff / 2
            h[L + k] = coeff / 2

    elif ftype == 3:
        # Type III: odd, antisymmetric
        M = (num_taps - 1) // 2
        h[M] = 0
        for k in range(1, M + 1):
            # h[M+k] = -h[M-k] = (2/pi) * integral_0^pi H(omega)*sin(k*omega) d_omega
            coeff = (
                torch.trapezoid(A_dense * torch.sin(k * omega), omega)
                * 2
                / math.pi
            )
            h[M + k] = coeff / 2
            h[M - k] = -coeff / 2

    else:
        # Type IV: even, antisymmetric
        L = num_taps // 2
        for k in range(L):
            # Coefficients for sin((k+0.5)*omega) basis
            coeff = (
                torch.trapezoid(A_dense * torch.sin((k + 0.5) * omega), omega)
                * 2
                / math.pi
            )
            h[L - 1 - k] = -coeff / 2
            h[L + k] = coeff / 2

    return h
