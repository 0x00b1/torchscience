"""Benchmark polynomial root finding.

Compares companion matrix (O(n^3)) vs Aberth-Ehrlich (O(n^2 * iterations))
methods across different polynomial degrees to validate performance characteristics.
"""

import time

import torch

from torchscience.polynomial import polynomial, polynomial_roots


def benchmark_roots(
    degree: int, n_iterations: int = 10, method: str = "auto"
) -> float:
    """Benchmark root finding at given degree.

    Parameters
    ----------
    degree : int
        Degree of polynomial (number of roots to find).
    n_iterations : int
        Number of iterations for timing.
    method : str
        'auto', 'companion', or 'aberth'.

    Returns
    -------
    float
        Average time per root finding in milliseconds.
    """
    # Random monic polynomial
    coeffs = torch.randn(degree + 1, dtype=torch.float64)
    coeffs[-1] = 1.0  # Monic (leading coeff = 1)
    p = polynomial(coeffs)

    # Warmup
    for _ in range(3):
        _ = polynomial_roots(p, method=method)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = polynomial_roots(p, method=method)

    elapsed = time.perf_counter() - start
    return elapsed / n_iterations * 1000  # ms


def main():
    """Run root finding benchmarks across degrees."""
    degrees = [8, 16, 32, 64, 128, 256, 512]

    print("Polynomial Root Finding Benchmark")
    print("=" * 70)
    print(
        f"{'Degree':>8} {'Companion (ms)':>16} {'Aberth (ms)':>16} {'Auto (ms)':>16}"
    )
    print("-" * 70)

    for degree in degrees:
        # Companion matrix method
        try:
            ms_companion = benchmark_roots(degree, method="companion")
        except Exception as e:
            ms_companion = float("nan")
            print(f"Companion failed for degree {degree}: {e}")

        # Aberth-Ehrlich method
        try:
            ms_aberth = benchmark_roots(degree, method="aberth")
        except Exception as e:
            ms_aberth = float("nan")
            print(f"Aberth failed for degree {degree}: {e}")

        # Auto method (adaptive selection)
        try:
            ms_auto = benchmark_roots(degree, method="auto")
        except Exception as e:
            ms_auto = float("nan")
            print(f"Auto failed for degree {degree}: {e}")

        print(
            f"{degree:>8} {ms_companion:>16.4f} {ms_aberth:>16.4f} {ms_auto:>16.4f}"
        )

    print()
    print("Notes:")
    print("- Companion: O(n^3) via eigenvalue decomposition")
    print("- Aberth: O(n^2 * k) where k is iterations (typically 10-20)")
    print("- Auto switches from Companion to Aberth at degree > 64")


if __name__ == "__main__":
    main()
