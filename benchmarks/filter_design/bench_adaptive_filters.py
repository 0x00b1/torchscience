"""Benchmarks for adaptive filter functions.

This module benchmarks torchscience adaptive filter functions (LMS, NLMS, RLS, etc.)
and demonstrates their convergence characteristics.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

# torchscience imports
from torchscience.filter_design import (
    kalman_filter,
    leaky_lms,
    lms,
    nlms,
    rls,
)


def benchmark(
    func: Callable,
    *args: Any,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """Run a simple benchmark on a function.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args : Any
        Positional arguments to pass to func.
    warmup : int, optional
        Number of warmup iterations. Default is 3.
    iterations : int, optional
        Number of timed iterations. Default is 10.
    **kwargs : Any
        Keyword arguments to pass to func.

    Returns
    -------
    dict
        Dictionary with timing statistics:
        - 'mean': Mean time in seconds
        - 'std': Standard deviation in seconds
        - 'min': Minimum time in seconds
        - 'max': Maximum time in seconds
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.3f}us"
    elif seconds < 1:
        return f"{seconds * 1e3:.3f}ms"
    else:
        return f"{seconds:.3f}s"


def print_result(name: str, ts_time: dict[str, float]) -> None:
    """Print benchmark result."""
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"  Time: {format_time(ts_time['mean'])} +/- {format_time(ts_time['std'])}"
    )


def print_comparison(
    name: str,
    times: dict[str, dict[str, float]],
) -> None:
    """Print benchmark comparison results for multiple methods."""
    print(f"\n{name}")
    print("-" * len(name))

    # Find fastest method
    fastest_name = min(times.keys(), key=lambda k: times[k]["mean"])
    fastest_time = times[fastest_name]["mean"]

    for method_name, ts_time in times.items():
        slowdown = ts_time["mean"] / fastest_time
        if slowdown > 1.01:
            suffix = f" ({slowdown:.2f}x slower)"
        else:
            suffix = " (fastest)"
        print(
            f"  {method_name}: {format_time(ts_time['mean'])} +/- {format_time(ts_time['std'])}{suffix}"
        )


def generate_system_id_data(
    num_samples: int,
    num_taps: int,
    snr_db: float = 30.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data for system identification benchmark.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    num_taps : int
        Number of filter taps in the unknown system.
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 30.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x : Tensor
        Input signal.
    d : Tensor
        Desired output (system output + noise).
    h_true : Tensor
        True system impulse response.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate true system
    h_true = torch.randn(num_taps, dtype=torch.float64)
    h_true = h_true / h_true.norm()  # Normalize

    # Generate input signal
    x = torch.randn(num_samples, dtype=torch.float64)

    # Generate system output using convolution
    # Pad x and convolve
    x_padded = torch.nn.functional.pad(x, (num_taps - 1, 0))
    d_clean = torch.nn.functional.conv1d(
        x_padded.view(1, 1, -1),
        h_true.view(1, 1, -1).flip(-1),
    ).squeeze()[:num_samples]

    # Add noise
    signal_power = d_clean.var()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(d_clean) * torch.sqrt(noise_power)
    d = d_clean + noise

    return x, d, h_true


class BenchAdaptiveFilters:
    """Benchmarks for adaptive filter functions."""

    def __init__(self, warmup: int = 3, iterations: int = 10):
        """Initialize benchmark runner.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup iterations. Default is 3.
        iterations : int, optional
            Number of timed iterations. Default is 10.
        """
        self.warmup = warmup
        self.iterations = iterations

    def _bench(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Run benchmark with configured settings."""
        return benchmark(
            func,
            *args,
            warmup=self.warmup,
            iterations=self.iterations,
            **kwargs,
        )

    def bench_lms(
        self, num_samples: int = 10000, num_taps: int = 32, mu: float = 0.01
    ) -> None:
        """Benchmark LMS adaptive filter.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 10000.
        num_taps : int, optional
            Number of filter taps. Default is 32.
        mu : float, optional
            Step size. Default is 0.01.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        ts_time = self._bench(lms, x, d, num_taps, mu=mu)

        print_result(f"LMS (samples={num_samples}, taps={num_taps})", ts_time)

    def bench_nlms(
        self, num_samples: int = 10000, num_taps: int = 32, mu: float = 0.5
    ) -> None:
        """Benchmark NLMS adaptive filter.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 10000.
        num_taps : int, optional
            Number of filter taps. Default is 32.
        mu : float, optional
            Step size. Default is 0.5.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        ts_time = self._bench(nlms, x, d, num_taps, mu=mu)

        print_result(f"NLMS (samples={num_samples}, taps={num_taps})", ts_time)

    def bench_leaky_lms(
        self,
        num_samples: int = 10000,
        num_taps: int = 32,
        mu: float = 0.01,
        gamma: float = 0.001,
    ) -> None:
        """Benchmark Leaky LMS adaptive filter.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 10000.
        num_taps : int, optional
            Number of filter taps. Default is 32.
        mu : float, optional
            Step size. Default is 0.01.
        gamma : float, optional
            Leakage factor. Default is 0.001.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        ts_time = self._bench(leaky_lms, x, d, num_taps, mu=mu, gamma=gamma)

        print_result(
            f"Leaky LMS (samples={num_samples}, taps={num_taps})", ts_time
        )

    def bench_rls(
        self,
        num_samples: int = 1000,
        num_taps: int = 32,
        lam: float = 0.99,
    ) -> None:
        """Benchmark RLS adaptive filter.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 1000 (RLS is more expensive).
        num_taps : int, optional
            Number of filter taps. Default is 32.
        lam : float, optional
            Forgetting factor. Default is 0.99.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        ts_time = self._bench(rls, x, d, num_taps, lam=lam)

        print_result(f"RLS (samples={num_samples}, taps={num_taps})", ts_time)

    def bench_kalman(
        self,
        num_samples: int = 1000,
        state_dim: int = 4,
        obs_dim: int = 2,
    ) -> None:
        """Benchmark Kalman filter.

        Parameters
        ----------
        num_samples : int, optional
            Number of time steps. Default is 1000.
        state_dim : int, optional
            State dimension. Default is 4.
        obs_dim : int, optional
            Observation dimension. Default is 2.
        """
        torch.manual_seed(42)

        # State transition matrix (random stable)
        F = 0.9 * torch.eye(state_dim, dtype=torch.float64)
        F += 0.05 * torch.randn(state_dim, state_dim, dtype=torch.float64)

        # Observation matrix
        H = torch.randn(obs_dim, state_dim, dtype=torch.float64)

        # Covariances
        Q = 0.01 * torch.eye(state_dim, dtype=torch.float64)
        R = 0.1 * torch.eye(obs_dim, dtype=torch.float64)
        P0 = torch.eye(state_dim, dtype=torch.float64)

        # Generate observations
        x = torch.zeros(num_samples, state_dim, dtype=torch.float64)
        z = torch.zeros(num_samples, obs_dim, dtype=torch.float64)

        state = torch.randn(state_dim, dtype=torch.float64)
        for t in range(num_samples):
            x[t] = state
            z[t] = H @ state + 0.1 * torch.randn(obs_dim, dtype=torch.float64)
            state = F @ state + 0.01 * torch.randn(
                state_dim, dtype=torch.float64
            )

        x0 = torch.zeros(state_dim, dtype=torch.float64)

        ts_time = self._bench(kalman_filter, z, F, H, Q, R, x0, P0)

        print_result(
            f"Kalman (samples={num_samples}, state_dim={state_dim})", ts_time
        )

    def bench_compare_lms_variants(
        self, num_samples: int = 5000, num_taps: int = 32
    ) -> None:
        """Compare LMS, NLMS, and Leaky LMS.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 5000.
        num_taps : int, optional
            Number of filter taps. Default is 32.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        times = {
            "LMS": self._bench(lms, x, d, num_taps, mu=0.01),
            "NLMS": self._bench(nlms, x, d, num_taps, mu=0.5),
            "Leaky LMS": self._bench(
                leaky_lms, x, d, num_taps, mu=0.01, gamma=0.001
            ),
        }

        print_comparison(
            f"LMS Variants (samples={num_samples}, taps={num_taps})", times
        )

    def bench_compare_lms_vs_rls(
        self, num_samples: int = 1000, num_taps: int = 16
    ) -> None:
        """Compare LMS vs RLS (RLS is O(L^2) vs LMS O(L)).

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. Default is 1000.
        num_taps : int, optional
            Number of filter taps. Default is 16.
        """
        x, d, _ = generate_system_id_data(num_samples, num_taps, seed=42)

        times = {
            "LMS": self._bench(lms, x, d, num_taps, mu=0.01),
            "NLMS": self._bench(nlms, x, d, num_taps, mu=0.5),
            "RLS": self._bench(rls, x, d, num_taps, lam=0.99),
        }

        print_comparison(
            f"LMS vs RLS (samples={num_samples}, taps={num_taps})", times
        )

    def bench_batched_lms(
        self, batch_size: int = 32, num_samples: int = 1000, num_taps: int = 16
    ) -> None:
        """Benchmark batched LMS (parallel adaptation).

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 32.
        num_samples : int, optional
            Number of samples per batch. Default is 1000.
        num_taps : int, optional
            Number of filter taps. Default is 16.
        """
        torch.manual_seed(42)

        # Generate batched data
        x = torch.randn(batch_size, num_samples, dtype=torch.float64)
        d = torch.randn(batch_size, num_samples, dtype=torch.float64)

        ts_time = self._bench(lms, x, d, num_taps, mu=0.01)

        print_result(
            f"Batched LMS (batch={batch_size}, samples={num_samples}, taps={num_taps})",
            ts_time,
        )

    def run_all(self) -> None:
        """Run all adaptive filter benchmarks."""
        print("=" * 60)
        print("ADAPTIVE FILTER BENCHMARKS")
        print("=" * 60)

        # Individual adaptive filters
        print("\n--- Individual Adaptive Filters ---")
        self.bench_lms()
        self.bench_nlms()
        self.bench_leaky_lms()
        self.bench_rls()
        self.bench_kalman()

        # Comparisons
        print("\n--- Adaptive Filter Comparisons ---")
        self.bench_compare_lms_variants()
        self.bench_compare_lms_vs_rls()

        # Batched operations
        print("\n--- Batched Operations ---")
        self.bench_batched_lms()

    def run_scaling(self) -> None:
        """Run scaling benchmarks with varying parameters."""
        print("=" * 60)
        print("SCALING BENCHMARKS")
        print("=" * 60)

        # Number of samples scaling
        print("\n--- Sample Count Scaling (LMS) ---")
        for num_samples in [1000, 5000, 10000, 50000]:
            self.bench_lms(num_samples=num_samples)

        # Number of taps scaling
        print("\n--- Filter Taps Scaling (LMS) ---")
        for num_taps in [8, 16, 32, 64, 128]:
            self.bench_lms(num_taps=num_taps, num_samples=5000)

        # RLS taps scaling (shows O(L^2) behavior)
        print("\n--- Filter Taps Scaling (RLS) ---")
        for num_taps in [8, 16, 32, 64]:
            self.bench_rls(num_taps=num_taps, num_samples=500)

        # Batch size scaling
        print("\n--- Batch Size Scaling ---")
        for batch_size in [1, 8, 32, 64]:
            self.bench_batched_lms(batch_size=batch_size)


if __name__ == "__main__":
    bench = BenchAdaptiveFilters(warmup=3, iterations=10)
    bench.run_all()
    print("\n")
    bench.run_scaling()
