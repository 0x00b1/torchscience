"""Benchmarks for filter application functions.

This module compares torchscience filter application functions (lfilter, sosfilt,
fftfilt, etc.) against scipy baselines.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

# scipy imports - handle optional dependency
try:
    from scipy import signal as scipy_signal
    from scipy.signal import fftconvolve

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# torchscience imports
from torchscience.signal_processing.filter import (
    butterworth_design,
    fftfilt,
    filtfilt,
    firwin,
    lfilter,
    lfilter_zi,
    sosfilt,
    sosfilt_zi,
    sosfiltfilt,
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


def print_comparison(
    name: str,
    ts_time: dict[str, float],
    scipy_time: dict[str, float] | None = None,
) -> None:
    """Print benchmark comparison results."""
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"  torchscience: {format_time(ts_time['mean'])} +/- {format_time(ts_time['std'])}"
    )
    if scipy_time is not None:
        print(
            f"  scipy:        {format_time(scipy_time['mean'])} +/- {format_time(scipy_time['std'])}"
        )
        speedup = scipy_time["mean"] / ts_time["mean"]
        if speedup >= 1:
            print(f"  Speedup:      {speedup:.2f}x faster")
        else:
            print(f"  Speedup:      {1 / speedup:.2f}x slower")


class BenchFilterApplication:
    """Benchmarks for filter application functions."""

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

    def bench_lfilter(
        self, signal_length: int = 10000, order: int = 4
    ) -> None:
        """Benchmark lfilter vs scipy.signal.lfilter.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 4.
        """
        # Design filter and get BA coefficients
        b_ts, a_ts = butterworth_design(order, 0.3, output="ba")
        x_ts = torch.randn(signal_length, dtype=torch.float64)

        ts_time = self._bench(lfilter, b_ts, a_ts, x_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            b_np, a_np = scipy_signal.butter(order, 0.3, output="ba")
            x_np = x_ts.numpy()
            scipy_time = self._bench(scipy_signal.lfilter, b_np, a_np, x_np)

        print_comparison(
            f"lfilter (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_lfilter_with_zi(
        self, signal_length: int = 10000, order: int = 4
    ) -> None:
        """Benchmark lfilter with initial conditions.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 4.
        """
        # Design filter and get BA coefficients
        b_ts, a_ts = butterworth_design(order, 0.3, output="ba")
        x_ts = torch.randn(signal_length, dtype=torch.float64)
        zi_ts = lfilter_zi(b_ts, a_ts) * x_ts[0]

        ts_time = self._bench(lfilter, b_ts, a_ts, x_ts, zi=zi_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            b_np, a_np = scipy_signal.butter(order, 0.3, output="ba")
            x_np = x_ts.numpy()
            zi_np = scipy_signal.lfilter_zi(b_np, a_np) * x_np[0]
            scipy_time = self._bench(
                scipy_signal.lfilter, b_np, a_np, x_np, zi=zi_np
            )

        print_comparison(
            f"lfilter with zi (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_sosfilt(
        self, signal_length: int = 10000, order: int = 8
    ) -> None:
        """Benchmark sosfilt vs scipy.signal.sosfilt.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 8.
        """
        sos_ts = butterworth_design(order, 0.3, output="sos")
        x_ts = torch.randn(signal_length, dtype=torch.float64)

        ts_time = self._bench(sosfilt, sos_ts, x_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            sos_np = scipy_signal.butter(order, 0.3, output="sos")
            x_np = x_ts.numpy()
            scipy_time = self._bench(scipy_signal.sosfilt, sos_np, x_np)

        print_comparison(
            f"sosfilt (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_sosfilt_with_zi(
        self, signal_length: int = 10000, order: int = 8
    ) -> None:
        """Benchmark sosfilt with initial conditions.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 8.
        """
        sos_ts = butterworth_design(order, 0.3, output="sos")
        x_ts = torch.randn(signal_length, dtype=torch.float64)
        zi_ts = sosfilt_zi(sos_ts) * x_ts[0]

        ts_time = self._bench(sosfilt, sos_ts, x_ts, zi=zi_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            sos_np = scipy_signal.butter(order, 0.3, output="sos")
            x_np = x_ts.numpy()
            zi_np = scipy_signal.sosfilt_zi(sos_np) * x_np[0]
            scipy_time = self._bench(
                scipy_signal.sosfilt, sos_np, x_np, zi=zi_np
            )

        print_comparison(
            f"sosfilt with zi (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_fftfilt(
        self, signal_length: int = 100000, num_taps: int = 101
    ) -> None:
        """Benchmark fftfilt vs scipy.signal.fftconvolve.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 100000.
        num_taps : int, optional
            Number of filter taps. Default is 101.
        """
        h_ts = firwin(num_taps, 0.3)
        x_ts = torch.randn(signal_length, dtype=torch.float64)

        ts_time = self._bench(fftfilt, h_ts, x_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            h_np = h_ts.numpy()
            x_np = x_ts.numpy()
            scipy_time = self._bench(fftconvolve, h_np, x_np, mode="same")

        print_comparison(
            f"fftfilt (length={signal_length}, taps={num_taps})",
            ts_time,
            scipy_time,
        )

    def bench_filtfilt(
        self, signal_length: int = 10000, order: int = 4
    ) -> None:
        """Benchmark filtfilt vs scipy.signal.filtfilt.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 4.
        """
        b_ts, a_ts = butterworth_design(order, 0.3, output="ba")
        x_ts = torch.randn(signal_length, dtype=torch.float64)

        ts_time = self._bench(filtfilt, b_ts, a_ts, x_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            b_np, a_np = scipy_signal.butter(order, 0.3, output="ba")
            x_np = x_ts.numpy()
            scipy_time = self._bench(scipy_signal.filtfilt, b_np, a_np, x_np)

        print_comparison(
            f"filtfilt (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_sosfiltfilt(
        self, signal_length: int = 10000, order: int = 8
    ) -> None:
        """Benchmark sosfiltfilt vs scipy.signal.sosfiltfilt.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 10000.
        order : int, optional
            Filter order. Default is 8.
        """
        sos_ts = butterworth_design(order, 0.3, output="sos")
        x_ts = torch.randn(signal_length, dtype=torch.float64)

        ts_time = self._bench(sosfiltfilt, sos_ts, x_ts)

        scipy_time = None
        if SCIPY_AVAILABLE:
            sos_np = scipy_signal.butter(order, 0.3, output="sos")
            x_np = x_ts.numpy()
            scipy_time = self._bench(scipy_signal.sosfiltfilt, sos_np, x_np)

        print_comparison(
            f"sosfiltfilt (length={signal_length}, order={order})",
            ts_time,
            scipy_time,
        )

    def bench_batched_lfilter(
        self, batch_size: int = 32, signal_length: int = 1000, order: int = 4
    ) -> None:
        """Benchmark batched lfilter (torchscience advantage).

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 32.
        signal_length : int, optional
            Length of each signal. Default is 1000.
        order : int, optional
            Filter order. Default is 4.
        """
        b_ts, a_ts = butterworth_design(order, 0.3, output="ba")
        x_ts = torch.randn(batch_size, signal_length, dtype=torch.float64)

        ts_time = self._bench(lfilter, b_ts, a_ts, x_ts, axis=-1)

        scipy_time = None
        if SCIPY_AVAILABLE:
            b_np, a_np = scipy_signal.butter(order, 0.3, output="ba")
            x_np = x_ts.numpy()
            scipy_time = self._bench(
                scipy_signal.lfilter, b_np, a_np, x_np, axis=-1
            )

        print_comparison(
            f"batched lfilter (batch={batch_size}, length={signal_length})",
            ts_time,
            scipy_time,
        )

    def bench_batched_sosfilt(
        self, batch_size: int = 32, signal_length: int = 1000, order: int = 8
    ) -> None:
        """Benchmark batched sosfilt (torchscience advantage).

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 32.
        signal_length : int, optional
            Length of each signal. Default is 1000.
        order : int, optional
            Filter order. Default is 8.
        """
        sos_ts = butterworth_design(order, 0.3, output="sos")
        x_ts = torch.randn(batch_size, signal_length, dtype=torch.float64)

        ts_time = self._bench(sosfilt, sos_ts, x_ts, axis=-1)

        scipy_time = None
        if SCIPY_AVAILABLE:
            sos_np = scipy_signal.butter(order, 0.3, output="sos")
            x_np = x_ts.numpy()
            scipy_time = self._bench(
                scipy_signal.sosfilt, sos_np, x_np, axis=-1
            )

        print_comparison(
            f"batched sosfilt (batch={batch_size}, length={signal_length})",
            ts_time,
            scipy_time,
        )

    def run_all(self) -> None:
        """Run all filter application benchmarks."""
        print("=" * 60)
        print("FILTER APPLICATION BENCHMARKS")
        print("=" * 60)

        # Basic filter application
        print("\n--- IIR Filter Application ---")
        self.bench_lfilter()
        self.bench_lfilter_with_zi()
        self.bench_sosfilt()
        self.bench_sosfilt_with_zi()

        # Zero-phase filtering
        print("\n--- Zero-Phase Filtering ---")
        self.bench_filtfilt()
        self.bench_sosfiltfilt()

        # FFT-based filtering
        print("\n--- FFT-Based Filtering ---")
        self.bench_fftfilt()

        # Batched operations
        print("\n--- Batched Operations ---")
        self.bench_batched_lfilter()
        self.bench_batched_sosfilt()

    def run_scaling(self) -> None:
        """Run scaling benchmarks with varying parameters."""
        print("=" * 60)
        print("SCALING BENCHMARKS")
        print("=" * 60)

        # Signal length scaling
        print("\n--- Signal Length Scaling (lfilter) ---")
        for length in [1000, 10000, 100000, 1000000]:
            self.bench_lfilter(signal_length=length)

        # Batch size scaling
        print("\n--- Batch Size Scaling ---")
        for batch_size in [1, 8, 32, 128]:
            self.bench_batched_lfilter(batch_size=batch_size)

        # FFT filter scaling
        print("\n--- FFT Filter Scaling ---")
        for num_taps in [11, 101, 1001]:
            self.bench_fftfilt(num_taps=num_taps)


if __name__ == "__main__":
    bench = BenchFilterApplication(warmup=5, iterations=20)
    bench.run_all()
    print("\n")
    bench.run_scaling()
