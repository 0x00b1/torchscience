"""Benchmark polynomial multiplication.

Compares direct (O(n^2)) vs FFT-based (O(n log n)) multiplication
across different polynomial degrees to validate performance characteristics.
"""

import time

import torch

from torchscience.polynomial import (
    polynomial,
    polynomial_multiply,
    polynomial_multiply_fft,
)
from torchscience.polynomial._polynomial._polynomial_multiply import (
    _multiply_direct,
)


def benchmark_multiply(
    degree: int,
    n_iterations: int = 100,
    device: str = "cpu",
    method: str = "auto",
) -> float:
    """Benchmark multiplication at given degree.

    Parameters
    ----------
    degree : int
        Degree of polynomials to multiply.
    n_iterations : int
        Number of iterations for timing.
    device : str
        Device to run on ('cpu' or 'cuda').
    method : str
        'auto', 'direct', or 'fft'.

    Returns
    -------
    float
        Average time per multiplication in milliseconds.
    """
    a = polynomial(torch.randn(degree + 1, device=device, dtype=torch.float64))
    b = polynomial(torch.randn(degree + 1, device=device, dtype=torch.float64))

    # Select multiplication function
    if method == "auto":
        multiply_fn = polynomial_multiply
    elif method == "direct":
        multiply_fn = _multiply_direct
    elif method == "fft":
        multiply_fn = polynomial_multiply_fft
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(10):
        _ = multiply_fn(a, b)

    # Synchronize before timing (important for CUDA)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = multiply_fn(a, b)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / n_iterations * 1000  # ms


def main():
    """Run multiplication benchmarks across degrees."""
    degrees = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    print("Polynomial Multiplication Benchmark")
    print("=" * 70)
    print(
        f"{'Degree':>8} {'Direct (ms)':>14} {'FFT (ms)':>14} {'Auto (ms)':>14}"
    )
    print("-" * 70)

    for degree in degrees:
        # Direct method
        try:
            ms_direct = benchmark_multiply(degree, method="direct")
        except Exception as e:
            ms_direct = float("nan")
            print(f"Direct failed for degree {degree}: {e}")

        # FFT method
        try:
            ms_fft = benchmark_multiply(degree, method="fft")
        except Exception as e:
            ms_fft = float("nan")
            print(f"FFT failed for degree {degree}: {e}")

        # Auto method (adaptive selection)
        try:
            ms_auto = benchmark_multiply(degree, method="auto")
        except Exception as e:
            ms_auto = float("nan")
            print(f"Auto failed for degree {degree}: {e}")

        print(
            f"{degree:>8} {ms_direct:>14.4f} {ms_fft:>14.4f} {ms_auto:>14.4f}"
        )

    print()
    print("Notes:")
    print("- Direct uses O(n^2) convolution via C++ kernel")
    print("- FFT uses O(n log n) FFT-based convolution")
    print("- Auto switches from Direct to FFT at output size >= 64")


if __name__ == "__main__":
    main()
