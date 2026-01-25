"""High-level filter design interface."""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch
from torch import Tensor

from ._bessel_design import bessel_design
from ._butterworth_design import butterworth_design
from ._chebyshev_type_1_design import chebyshev_type_1_design
from ._chebyshev_type_2_design import chebyshev_type_2_design
from ._elliptic_design import elliptic_design
from ._fftfilt import fftfilt
from ._firwin import firwin
from ._iirnotch import iirnotch
from ._iirpeak import iirpeak
from ._sosfilt import sosfilt
from ._sosfiltfilt import sosfiltfilt

# Filter type aliases
IIRMethod = Literal[
    "butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"
]
FIRMethod = Literal["firwin"]
FilterMethod = Union[IIRMethod, FIRMethod]


class Filter:
    """Represents a designed filter with methods for application and analysis.

    A Filter object encapsulates filter coefficients and provides methods for
    applying the filter to signals, computing frequency response, and converting
    between different filter representations.

    Parameters
    ----------
    sos : Tensor, optional
        Second-order sections representation, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    ba : tuple of Tensors, optional
        Numerator and denominator coefficients (b, a).
    zpk : tuple of Tensors, optional
        Zeros, poles, and gain (z, p, k).
    fir : Tensor, optional
        FIR filter coefficients.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import Filter
    >>> # Create a filter from SOS
    >>> sos = torch.tensor([[0.1, 0.2, 0.1, 1.0, -0.5, 0.2]])
    >>> filt = Filter(sos=sos)
    >>> filt.is_iir
    True
    """

    def __init__(
        self,
        sos: Optional[Tensor] = None,
        ba: Optional[tuple[Tensor, Tensor]] = None,
        zpk: Optional[tuple[Tensor, Tensor, Tensor]] = None,
        fir: Optional[Tensor] = None,
    ):
        self.sos = sos
        self.ba = ba
        self.zpk = zpk
        self.fir = fir

    @property
    def is_iir(self) -> bool:
        """Return True if this is an IIR filter, False for FIR."""
        return self.fir is None

    def apply(
        self,
        x: Tensor,
        axis: int = -1,
        zi: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Apply the filter to a signal.

        Parameters
        ----------
        x : Tensor
            Input signal.
        axis : int, optional
            Axis along which to filter. Default is -1 (last axis).
        zi : Tensor, optional
            Initial conditions. Only applicable for IIR filters with SOS.

        Returns
        -------
        y : Tensor
            Filtered signal, same shape as x.
        zf : Tensor, optional
            Final filter states (only returned if zi is provided).

        Examples
        --------
        >>> import torch
        >>> from torchscience.signal_processing.filter import FilterDesigner
        >>> designer = FilterDesigner()
        >>> filt = designer.lowpass(cutoff=0.3, order=4)
        >>> x = torch.randn(100)
        >>> y = filt.apply(x)
        """
        if self.fir is not None:
            return fftfilt(self.fir, x, axis=axis)
        elif self.sos is not None:
            return sosfilt(self.sos, x, axis=axis, zi=zi)
        elif self.ba is not None:
            # For BA format, use lfilter
            from ._lfilter import lfilter

            b, a = self.ba
            return lfilter(b, a, x, axis=axis, zi=zi)
        else:
            raise ValueError("No filter coefficients available")

    def apply_zero_phase(
        self,
        x: Tensor,
        axis: int = -1,
        padtype: Optional[Literal["odd", "even", "constant"]] = "odd",
        padlen: Optional[int] = None,
    ) -> Tensor:
        """Apply the filter with zero phase (forward-backward).

        Zero-phase filtering eliminates phase distortion by applying the filter
        twice: once forward and once backward. This doubles the filter order
        and squares the magnitude response.

        Parameters
        ----------
        x : Tensor
            Input signal.
        axis : int, optional
            Axis along which to filter. Default is -1.
        padtype : {"odd", "even", "constant", None}, optional
            Padding type to reduce edge effects. Default is "odd".
        padlen : int, optional
            Number of samples to pad at each end.

        Returns
        -------
        y : Tensor
            Filtered signal with zero phase distortion.

        Examples
        --------
        >>> import torch
        >>> from torchscience.signal_processing.filter import FilterDesigner
        >>> designer = FilterDesigner()
        >>> filt = designer.lowpass(cutoff=0.3, order=4)
        >>> x = torch.randn(100)
        >>> y = filt.apply_zero_phase(x)
        """
        if self.fir is not None:
            # FIR zero-phase: apply twice (forward and backward)
            from ._filtfilt import filtfilt

            # FIR has denominator = [1]
            a = torch.ones(1, dtype=self.fir.dtype, device=self.fir.device)
            return filtfilt(
                self.fir, a, x, axis=axis, padtype=padtype, padlen=padlen
            )
        elif self.sos is not None:
            return sosfiltfilt(
                self.sos, x, axis=axis, padtype=padtype, padlen=padlen
            )
        elif self.ba is not None:
            from ._filtfilt import filtfilt

            b, a = self.ba
            return filtfilt(b, a, x, axis=axis, padtype=padtype, padlen=padlen)
        else:
            raise ValueError("No filter coefficients available")

    def frequency_response(
        self,
        n_points: int = 512,
        whole: bool = False,
        sampling_frequency: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the frequency response of the filter.

        Parameters
        ----------
        n_points : int, optional
            Number of frequency points. Default is 512.
        whole : bool, optional
            If True, compute from 0 to sampling frequency.
            If False (default), compute from 0 to Nyquist.
        sampling_frequency : float, optional
            The sampling frequency. If not specified, returns normalized
            frequencies in [0, 1] where 1 = Nyquist.

        Returns
        -------
        w : Tensor
            Frequency points.
        H : Tensor
            Complex frequency response.

        Examples
        --------
        >>> import torch
        >>> from torchscience.signal_processing.filter import FilterDesigner
        >>> designer = FilterDesigner()
        >>> filt = designer.lowpass(cutoff=0.3, order=4)
        >>> w, H = filt.frequency_response()
        >>> magnitude_db = 20 * torch.log10(torch.abs(H))
        """
        from . import (
            frequency_response,
            frequency_response_fir,
            frequency_response_sos,
        )

        if self.fir is not None:
            return frequency_response_fir(
                self.fir,
                frequencies=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        elif self.sos is not None:
            return frequency_response_sos(
                self.sos,
                frequencies=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        elif self.ba is not None:
            b, a = self.ba
            return frequency_response(
                b,
                a,
                frequencies=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        elif self.zpk is not None:
            from . import (
                frequency_response_zpk,
            )

            z, p, k = self.zpk
            return frequency_response_zpk(
                z,
                p,
                k,
                frequencies=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        else:
            raise ValueError("No filter coefficients available")

    def group_delay(
        self,
        n_points: int = 512,
        whole: bool = False,
        sampling_frequency: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the group delay of the filter.

        Group delay is the negative derivative of the phase response with
        respect to frequency. For linear phase FIR filters, it equals
        (num_taps - 1) / 2 samples.

        Parameters
        ----------
        n_points : int, optional
            Number of frequency points. Default is 512.
        whole : bool, optional
            If True, compute from 0 to 2*pi. If False (default), compute to pi.
        sampling_frequency : float, optional
            The sampling frequency. If specified, frequencies are in Hz.

        Returns
        -------
        w : Tensor
            Frequency points.
        gd : Tensor
            Group delay in samples.

        Examples
        --------
        >>> import torch
        >>> from torchscience.signal_processing.filter import FilterDesigner
        >>> designer = FilterDesigner()
        >>> filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")
        >>> w, gd = filt.group_delay()
        >>> # Linear phase FIR has constant group delay of (num_taps - 1) / 2
        """
        from . import (
            group_delay,
            group_delay_sos,
        )

        if self.fir is not None:
            return group_delay(
                self.fir,
                denominator=None,
                n_points=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        elif self.sos is not None:
            return group_delay_sos(
                self.sos,
                n_points=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        elif self.ba is not None:
            b, a = self.ba
            return group_delay(
                b,
                denominator=a,
                n_points=n_points,
                whole=whole,
                sampling_frequency=sampling_frequency,
            )
        else:
            raise ValueError("No filter coefficients available")

    def to_sos(self) -> Tensor:
        """Convert filter to second-order sections format.

        Returns
        -------
        sos : Tensor
            Second-order sections, shape (n_sections, 6).

        Raises
        ------
        ValueError
            If filter cannot be converted to SOS (e.g., FIR filters).
        """
        if self.sos is not None:
            return self.sos
        elif self.zpk is not None:
            from ._zpk_to_sos import zpk_to_sos

            z, p, k = self.zpk
            return zpk_to_sos(z, p, k)
        elif self.ba is not None:
            from ._ba_to_sos import ba_to_sos

            b, a = self.ba
            return ba_to_sos(b, a)
        elif self.fir is not None:
            raise ValueError("Cannot convert FIR filter to SOS format")
        else:
            raise ValueError("No filter coefficients available")

    def to_ba(self) -> tuple[Tensor, Tensor]:
        """Convert filter to numerator/denominator format.

        Returns
        -------
        b : Tensor
            Numerator coefficients.
        a : Tensor
            Denominator coefficients.
        """
        if self.ba is not None:
            return self.ba
        elif self.sos is not None:
            from ._sos_to_ba import sos_to_ba

            return sos_to_ba(self.sos)
        elif self.zpk is not None:
            from ._zpk_to_ba import zpk_to_ba

            z, p, k = self.zpk
            return zpk_to_ba(z, p, k)
        elif self.fir is not None:
            # FIR filter: b = coefficients, a = [1]
            a = torch.ones(1, dtype=self.fir.dtype, device=self.fir.device)
            return self.fir, a
        else:
            raise ValueError("No filter coefficients available")

    def to_zpk(self) -> tuple[Tensor, Tensor, Tensor]:
        """Convert filter to zeros/poles/gain format.

        Returns
        -------
        z : Tensor
            Filter zeros.
        p : Tensor
            Filter poles.
        k : Tensor
            Filter gain.
        """
        if self.zpk is not None:
            return self.zpk
        elif self.sos is not None:
            from ._sos_to_zpk import sos_to_zpk

            return sos_to_zpk(self.sos)
        elif self.ba is not None:
            from ._ba_to_zpk import ba_to_zpk

            b, a = self.ba
            return ba_to_zpk(b, a)
        elif self.fir is not None:
            from ._ba_to_zpk import ba_to_zpk

            # FIR filter: a = [1]
            a = torch.ones(1, dtype=self.fir.dtype, device=self.fir.device)
            return ba_to_zpk(self.fir, a)
        else:
            raise ValueError("No filter coefficients available")


class FilterDesigner:
    """High-level interface for filter design.

    FilterDesigner provides a fluent, unified API for designing IIR and FIR
    filters. It simplifies common filter design workflows and provides methods
    for filter analysis.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Default output dtype for designed filters.
    device : torch.device, optional
        Default output device for designed filters.

    Examples
    --------
    Design a 4th-order Butterworth lowpass filter:

    >>> from torchscience.signal_processing.filter import FilterDesigner
    >>> designer = FilterDesigner()
    >>> filt = designer.lowpass(cutoff=0.3, order=4, method='butterworth')
    >>> y = filt.apply(x)

    Design an FIR lowpass filter:

    >>> filt = designer.lowpass(cutoff=0.3, num_taps=51, method='firwin')

    Design a bandpass filter with explicit sampling frequency:

    >>> filt = designer.bandpass(low=100.0, high=500.0, order=4,
    ...                          sampling_frequency=8000.0)
    """

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.dtype = dtype
        self.device = device

    def _validate_params(
        self,
        order: Optional[int],
        num_taps: Optional[int],
        method: str,
    ) -> None:
        """Validate filter design parameters."""
        if order is not None and num_taps is not None:
            raise ValueError("Cannot specify both order and num_taps")

        if order is None and num_taps is None:
            raise ValueError(
                "Must specify either order (for IIR) or num_taps (for FIR)"
            )

        iir_methods = {
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "elliptic",
            "bessel",
        }
        fir_methods = {"firwin"}

        if method not in iir_methods and method not in fir_methods:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported IIR methods: {iir_methods}. "
                f"Supported FIR methods: {fir_methods}."
            )

        if method in fir_methods and num_taps is None:
            raise ValueError(
                f"FIR method '{method}' requires num_taps, not order"
            )

        if method in iir_methods and order is None:
            raise ValueError(
                f"IIR method '{method}' requires order, not num_taps"
            )

    def _validate_iir_params(self, method: str, kwargs: dict) -> None:
        """Validate IIR-specific parameters."""
        if method == "chebyshev1" and "ripple" not in kwargs:
            raise ValueError(
                "Chebyshev Type I filter requires 'ripple' parameter (passband ripple in dB)"
            )

        if method == "chebyshev2" and "attenuation" not in kwargs:
            raise ValueError(
                "Chebyshev Type II filter requires 'attenuation' parameter (stopband attenuation in dB)"
            )

        if method == "elliptic":
            if "ripple" not in kwargs or "attenuation" not in kwargs:
                raise ValueError(
                    "Elliptic filter requires both 'ripple' (passband ripple in dB) "
                    "and 'attenuation' (stopband attenuation in dB) parameters"
                )

    def lowpass(
        self,
        cutoff: float,
        order: Optional[int] = None,
        num_taps: Optional[int] = None,
        method: FilterMethod = "butterworth",
        sampling_frequency: Optional[float] = None,
        **kwargs,
    ) -> Filter:
        """Design a lowpass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency. If sampling_frequency is not specified, this is
            a normalized frequency in [0, 1] where 1 = Nyquist. Otherwise,
            in the same units as sampling_frequency.
        order : int, optional
            Filter order (for IIR filters).
        num_taps : int, optional
            Number of taps (for FIR filters).
        method : str, optional
            Design method. IIR: 'butterworth' (default), 'chebyshev1',
            'chebyshev2', 'elliptic', 'bessel'. FIR: 'firwin'.
        sampling_frequency : float, optional
            Sampling frequency in Hz.
        **kwargs
            Additional method-specific parameters:
            - ripple: Passband ripple in dB (for chebyshev1, elliptic)
            - attenuation: Stopband attenuation in dB (for chebyshev2, elliptic)
            - window: Window function (for firwin)

        Returns
        -------
        Filter
            Designed filter object.

        Examples
        --------
        >>> designer = FilterDesigner()
        >>> filt = designer.lowpass(cutoff=0.3, order=4, method='butterworth')
        """
        self._validate_params(order, num_taps, method)

        if method in (
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "elliptic",
            "bessel",
        ):
            self._validate_iir_params(method, kwargs)
            return self._design_iir_filter(
                "lowpass", cutoff, order, method, sampling_frequency, kwargs
            )
        elif method == "firwin":
            return self._design_fir_filter(
                "lowpass", cutoff, num_taps, sampling_frequency, kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def highpass(
        self,
        cutoff: float,
        order: Optional[int] = None,
        num_taps: Optional[int] = None,
        method: FilterMethod = "butterworth",
        sampling_frequency: Optional[float] = None,
        **kwargs,
    ) -> Filter:
        """Design a highpass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency.
        order : int, optional
            Filter order (for IIR filters).
        num_taps : int, optional
            Number of taps (for FIR filters). Must be odd.
        method : str, optional
            Design method. Default is 'butterworth'.
        sampling_frequency : float, optional
            Sampling frequency in Hz.
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        Filter
            Designed filter object.
        """
        self._validate_params(order, num_taps, method)

        if method in (
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "elliptic",
            "bessel",
        ):
            self._validate_iir_params(method, kwargs)
            return self._design_iir_filter(
                "highpass", cutoff, order, method, sampling_frequency, kwargs
            )
        elif method == "firwin":
            return self._design_fir_filter(
                "highpass", cutoff, num_taps, sampling_frequency, kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def bandpass(
        self,
        low: float,
        high: float,
        order: Optional[int] = None,
        num_taps: Optional[int] = None,
        method: FilterMethod = "butterworth",
        sampling_frequency: Optional[float] = None,
        **kwargs,
    ) -> Filter:
        """Design a bandpass filter.

        Parameters
        ----------
        low : float
            Lower cutoff frequency.
        high : float
            Upper cutoff frequency.
        order : int, optional
            Filter order (for IIR filters). The resulting filter will have
            order 2*order (bandpass transformation doubles the order).
        num_taps : int, optional
            Number of taps (for FIR filters).
        method : str, optional
            Design method. Default is 'butterworth'.
        sampling_frequency : float, optional
            Sampling frequency in Hz.
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        Filter
            Designed filter object.
        """
        self._validate_params(order, num_taps, method)
        cutoff = [low, high]

        if method in (
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "elliptic",
            "bessel",
        ):
            self._validate_iir_params(method, kwargs)
            return self._design_iir_filter(
                "bandpass", cutoff, order, method, sampling_frequency, kwargs
            )
        elif method == "firwin":
            return self._design_fir_filter(
                "bandpass", cutoff, num_taps, sampling_frequency, kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def bandstop(
        self,
        low: float,
        high: float,
        order: Optional[int] = None,
        num_taps: Optional[int] = None,
        method: FilterMethod = "butterworth",
        sampling_frequency: Optional[float] = None,
        **kwargs,
    ) -> Filter:
        """Design a bandstop (notch) filter.

        Parameters
        ----------
        low : float
            Lower cutoff frequency.
        high : float
            Upper cutoff frequency.
        order : int, optional
            Filter order (for IIR filters).
        num_taps : int, optional
            Number of taps (for FIR filters). Must be odd.
        method : str, optional
            Design method. Default is 'butterworth'.
        sampling_frequency : float, optional
            Sampling frequency in Hz.
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        Filter
            Designed filter object.
        """
        self._validate_params(order, num_taps, method)
        cutoff = [low, high]

        if method in (
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "elliptic",
            "bessel",
        ):
            self._validate_iir_params(method, kwargs)
            return self._design_iir_filter(
                "bandstop", cutoff, order, method, sampling_frequency, kwargs
            )
        elif method == "firwin":
            return self._design_fir_filter(
                "bandstop", cutoff, num_taps, sampling_frequency, kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def notch(
        self,
        frequency: float,
        quality_factor: float = 30.0,
        sampling_frequency: float = 2.0,
    ) -> Filter:
        """Design a notch filter.

        A notch filter is a narrow band-reject filter that attenuates
        frequencies near the notch frequency while passing all others.

        Parameters
        ----------
        frequency : float
            Notch frequency. If sampling_frequency=2.0 (default), this is
            normalized frequency in [0, 1) where 1 = Nyquist.
        quality_factor : float, optional
            Quality factor Q. Higher Q means narrower notch. Default is 30.
        sampling_frequency : float, optional
            Sampling frequency. Default is 2.0 (normalized).

        Returns
        -------
        Filter
            Designed notch filter (second-order IIR).

        Examples
        --------
        >>> designer = FilterDesigner()
        >>> filt = designer.notch(frequency=60.0, quality_factor=30.0,
        ...                       sampling_frequency=1000.0)
        """
        b, a = iirnotch(
            frequency,
            quality_factor,
            sampling_frequency=sampling_frequency,
            dtype=self.dtype,
            device=self.device,
        )
        return Filter(ba=(b, a))

    def peak(
        self,
        frequency: float,
        quality_factor: float = 30.0,
        sampling_frequency: float = 2.0,
    ) -> Filter:
        """Design a peak (resonator) filter.

        A peak filter is a narrow bandpass filter that passes frequencies
        near the peak frequency while attenuating all others.

        Parameters
        ----------
        frequency : float
            Peak frequency. If sampling_frequency=2.0 (default), this is
            normalized frequency in [0, 1) where 1 = Nyquist.
        quality_factor : float, optional
            Quality factor Q. Higher Q means narrower peak. Default is 30.
        sampling_frequency : float, optional
            Sampling frequency. Default is 2.0 (normalized).

        Returns
        -------
        Filter
            Designed peak filter (second-order IIR).
        """
        b, a = iirpeak(
            frequency,
            quality_factor,
            sampling_frequency=sampling_frequency,
            dtype=self.dtype,
            device=self.device,
        )
        return Filter(ba=(b, a))

    def _design_iir_filter(
        self,
        filter_type: str,
        cutoff: Union[float, list[float]],
        order: int,
        method: str,
        sampling_frequency: Optional[float],
        kwargs: dict,
    ) -> Filter:
        """Design an IIR filter using the specified method."""
        common_kwargs = {
            "dtype": self.dtype,
            "device": self.device,
            "sampling_frequency": sampling_frequency,
        }

        if method == "butterworth":
            sos = butterworth_design(
                order=order,
                cutoff=cutoff,
                filter_type=filter_type,
                output="sos",
                **common_kwargs,
            )
        elif method == "chebyshev1":
            sos = chebyshev_type_1_design(
                order=order,
                cutoff=cutoff,
                passband_ripple_db=kwargs["ripple"],
                filter_type=filter_type,
                output="sos",
                **common_kwargs,
            )
        elif method == "chebyshev2":
            sos = chebyshev_type_2_design(
                order=order,
                cutoff=cutoff,
                stopband_attenuation_db=kwargs["attenuation"],
                filter_type=filter_type,
                output="sos",
                **common_kwargs,
            )
        elif method == "elliptic":
            sos = elliptic_design(
                order=order,
                cutoff=cutoff,
                passband_ripple_db=kwargs["ripple"],
                stopband_attenuation_db=kwargs["attenuation"],
                filter_type=filter_type,
                output="sos",
                **common_kwargs,
            )
        elif method == "bessel":
            sos = bessel_design(
                order=order,
                cutoff=cutoff,
                filter_type=filter_type,
                output="sos",
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown IIR method: {method}")

        return Filter(sos=sos)

    def _design_fir_filter(
        self,
        filter_type: str,
        cutoff: Union[float, list[float]],
        num_taps: int,
        sampling_frequency: Optional[float],
        kwargs: dict,
    ) -> Filter:
        """Design an FIR filter using firwin."""
        # Extract firwin-specific kwargs
        window = kwargs.get("window", "hamming")

        fir = firwin(
            num_taps=num_taps,
            cutoff=cutoff,
            filter_type=filter_type,
            window=window,
            sampling_frequency=sampling_frequency,
            dtype=self.dtype,
            device=self.device,
        )

        return Filter(fir=fir)
