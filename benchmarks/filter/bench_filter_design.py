"""Benchmarks for filter design functions.

This module compares torchscience filter design functions against scipy baselines.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

# scipy imports - handle optional dependency
try:
    from scipy import signal as scipy_signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# torchscience imports
from torchscience.filter import (
    bessel_design,
    butterworth_design,
    chebyshev_type_1_design,
    chebyshev_type_2_design,
    elliptic_design,
    firwin,
    firwin2,
    minimum_phase,
    remez,
    savgol_coeffs,
    yule_walker,
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


class BenchFilterDesign:
    """Benchmarks for IIR filter design functions."""

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

    def bench_butterworth_design(
        self, order: int = 8, cutoff: float = 0.3
    ) -> None:
        """Benchmark butterworth_design vs scipy.signal.butter.

        Parameters
        ----------
        order : int, optional
            Filter order. Default is 8.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        """
        ts_time = self._bench(butterworth_design, order, cutoff, output="sos")

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.butter, order, cutoff, output="sos"
            )

        print_comparison(f"Butterworth (order={order})", ts_time, scipy_time)

    def bench_chebyshev_type_1_design(
        self, order: int = 8, cutoff: float = 0.3, ripple: float = 0.5
    ) -> None:
        """Benchmark chebyshev_type_1_design vs scipy.signal.cheby1.

        Parameters
        ----------
        order : int, optional
            Filter order. Default is 8.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        ripple : float, optional
            Passband ripple in dB. Default is 0.5.
        """
        ts_time = self._bench(
            chebyshev_type_1_design, order, ripple, cutoff, output="sos"
        )

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.cheby1, order, ripple, cutoff, output="sos"
            )

        print_comparison(
            f"Chebyshev Type 1 (order={order})", ts_time, scipy_time
        )

    def bench_chebyshev_type_2_design(
        self, order: int = 8, cutoff: float = 0.3, attenuation: float = 40
    ) -> None:
        """Benchmark chebyshev_type_2_design vs scipy.signal.cheby2.

        Parameters
        ----------
        order : int, optional
            Filter order. Default is 8.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        attenuation : float, optional
            Stopband attenuation in dB. Default is 40.
        """
        ts_time = self._bench(
            chebyshev_type_2_design, order, attenuation, cutoff, output="sos"
        )

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.cheby2, order, attenuation, cutoff, output="sos"
            )

        print_comparison(
            f"Chebyshev Type 2 (order={order})", ts_time, scipy_time
        )

    def bench_elliptic_design(
        self,
        order: int = 8,
        cutoff: float = 0.3,
        ripple: float = 0.5,
        attenuation: float = 40,
    ) -> None:
        """Benchmark elliptic_design vs scipy.signal.ellip.

        Parameters
        ----------
        order : int, optional
            Filter order. Default is 8.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        ripple : float, optional
            Passband ripple in dB. Default is 0.5.
        attenuation : float, optional
            Stopband attenuation in dB. Default is 40.
        """
        ts_time = self._bench(
            elliptic_design, order, ripple, attenuation, cutoff, output="sos"
        )

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.ellip,
                order,
                ripple,
                attenuation,
                cutoff,
                output="sos",
            )

        print_comparison(f"Elliptic (order={order})", ts_time, scipy_time)

    def bench_bessel_design(self, order: int = 8, cutoff: float = 0.3) -> None:
        """Benchmark bessel_design vs scipy.signal.bessel.

        Parameters
        ----------
        order : int, optional
            Filter order. Default is 8.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        """
        ts_time = self._bench(bessel_design, order, cutoff, output="sos")

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.bessel, order, cutoff, output="sos"
            )

        print_comparison(f"Bessel (order={order})", ts_time, scipy_time)

    def bench_firwin(self, num_taps: int = 101, cutoff: float = 0.3) -> None:
        """Benchmark firwin vs scipy.signal.firwin.

        Parameters
        ----------
        num_taps : int, optional
            Number of filter taps. Default is 101.
        cutoff : float, optional
            Cutoff frequency (normalized). Default is 0.3.
        """
        ts_time = self._bench(firwin, num_taps, cutoff)

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(scipy_signal.firwin, num_taps, cutoff)

        print_comparison(f"firwin (num_taps={num_taps})", ts_time, scipy_time)

    def bench_firwin2(self, num_taps: int = 101) -> None:
        """Benchmark firwin2 vs scipy.signal.firwin2.

        Parameters
        ----------
        num_taps : int, optional
            Number of filter taps. Default is 101.
        """
        freqs = [0.0, 0.2, 0.3, 0.5]
        gains = [1.0, 1.0, 0.0, 0.0]

        ts_time = self._bench(firwin2, num_taps, freqs, gains)

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.firwin2, num_taps, freqs, gains
            )

        print_comparison(f"firwin2 (num_taps={num_taps})", ts_time, scipy_time)

    def bench_remez(self, num_taps: int = 51) -> None:
        """Benchmark remez vs scipy.signal.remez.

        Parameters
        ----------
        num_taps : int, optional
            Number of filter taps. Default is 51.
        """
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]

        ts_time = self._bench(remez, num_taps, bands, desired)

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.remez, num_taps, bands, desired, fs=1.0
            )

        print_comparison(f"remez (num_taps={num_taps})", ts_time, scipy_time)

    def bench_savgol_coeffs(
        self, window_length: int = 51, polyorder: int = 3
    ) -> None:
        """Benchmark savgol_coeffs vs scipy.signal.savgol_coeffs.

        Parameters
        ----------
        window_length : int, optional
            Window length. Default is 51.
        polyorder : int, optional
            Polynomial order. Default is 3.
        """
        ts_time = self._bench(savgol_coeffs, window_length, polyorder)

        scipy_time = None
        if SCIPY_AVAILABLE:
            scipy_time = self._bench(
                scipy_signal.savgol_coeffs, window_length, polyorder
            )

        print_comparison(
            f"savgol_coeffs (window={window_length}, poly={polyorder})",
            ts_time,
            scipy_time,
        )

    def bench_yule_walker(
        self, signal_length: int = 1000, order: int = 10
    ) -> None:
        """Benchmark yule_walker autoregressive filter design.

        Parameters
        ----------
        signal_length : int, optional
            Length of input signal. Default is 1000.
        order : int, optional
            AR model order. Default is 10.
        """
        x = torch.randn(signal_length)
        ts_time = self._bench(yule_walker, x, order)

        print_comparison(
            f"yule_walker (length={signal_length}, order={order})", ts_time
        )

    def bench_minimum_phase(self, num_taps: int = 101) -> None:
        """Benchmark minimum_phase FIR conversion.

        Parameters
        ----------
        num_taps : int, optional
            Number of filter taps. Default is 101.
        """
        h = firwin(num_taps, 0.3)
        ts_time = self._bench(minimum_phase, h)

        scipy_time = None
        if SCIPY_AVAILABLE:
            h_np = h.numpy()
            scipy_time = self._bench(scipy_signal.minimum_phase, h_np)

        print_comparison(
            f"minimum_phase (num_taps={num_taps})", ts_time, scipy_time
        )

    def run_all(self) -> None:
        """Run all filter design benchmarks."""
        print("=" * 60)
        print("FILTER DESIGN BENCHMARKS")
        print("=" * 60)

        # IIR filter design
        print("\n--- IIR Filter Design ---")
        self.bench_butterworth_design()
        self.bench_chebyshev_type_1_design()
        self.bench_chebyshev_type_2_design()
        self.bench_elliptic_design()
        self.bench_bessel_design()

        # FIR filter design
        print("\n--- FIR Filter Design ---")
        self.bench_firwin()
        self.bench_firwin2()
        self.bench_remez()
        self.bench_savgol_coeffs()
        self.bench_yule_walker()
        self.bench_minimum_phase()

    def run_scaling(self) -> None:
        """Run scaling benchmarks with varying parameters."""
        print("=" * 60)
        print("SCALING BENCHMARKS")
        print("=" * 60)

        # Butterworth scaling with order
        print("\n--- Butterworth Order Scaling ---")
        for order in [2, 4, 8, 16, 32]:
            self.bench_butterworth_design(order=order)

        # FIR scaling with number of taps
        print("\n--- FIR Taps Scaling ---")
        for num_taps in [11, 51, 101, 501, 1001]:
            self.bench_firwin(num_taps=num_taps)

        # Remez scaling with number of taps
        print("\n--- Remez Taps Scaling ---")
        for num_taps in [11, 31, 51, 101]:
            self.bench_remez(num_taps=num_taps)


if __name__ == "__main__":
    bench = BenchFilterDesign(warmup=5, iterations=20)
    bench.run_all()
    print("\n")
    bench.run_scaling()
