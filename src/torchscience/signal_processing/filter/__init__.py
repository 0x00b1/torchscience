"""Filter design functions for IIR and FIR filters."""

from ._ba_to_sos import ba_to_sos
from ._ba_to_zpk import ba_to_zpk
from ._batched import (
    batched_butterworth_design,
    batched_chebyshev_type_1_design,
    batched_chebyshev_type_2_design,
    batched_filter_apply,
    batched_firwin,
)
from ._bessel_analog import bessel_analog
from ._bessel_design import bessel_design
from ._bessel_prototype import bessel_prototype
from ._bilinear_transform_ba import bilinear_transform_ba
from ._bilinear_transform_zpk import bilinear_transform_zpk
from ._butterworth_analog import butterworth_analog
from ._butterworth_analog_bandpass_filter import (
    butterworth_analog_bandpass_filter,
)
from ._butterworth_design import butterworth_design
from ._butterworth_minimum_order import butterworth_minimum_order
from ._butterworth_prototype import butterworth_prototype
from ._cascade_sos import cascade_sos
from ._chebyshev_type_1_analog import chebyshev_type_1_analog
from ._chebyshev_type_1_design import chebyshev_type_1_design
from ._chebyshev_type_1_minimum_order import chebyshev_type_1_minimum_order
from ._chebyshev_type_1_prototype import chebyshev_type_1_prototype
from ._chebyshev_type_2_analog import chebyshev_type_2_analog
from ._chebyshev_type_2_design import chebyshev_type_2_design
from ._chebyshev_type_2_minimum_order import chebyshev_type_2_minimum_order
from ._chebyshev_type_2_prototype import chebyshev_type_2_prototype
from ._constants import (
    Q_BUTTERWORTH,
    Q_MEDIUM,
    Q_NARROW,
    Q_WIDE,
)
from ._elliptic_analog import elliptic_analog
from ._elliptic_design import elliptic_design
from ._elliptic_minimum_order import elliptic_minimum_order
from ._elliptic_prototype import elliptic_prototype
from ._exceptions import (
    ConvergenceError,
    FilterDesignError,
    FrequencyOrderError,
    InvalidCutoffError,
    InvalidNumTapsError,
    InvalidOrderError,
    NyquistViolationError,
    SOSNormalizationError,
    SpecificationError,
)
from ._fftfilt import fftfilt
from ._filter_designer import Filter, FilterDesigner
from ._filtfilt import filtfilt
from ._firwin import firwin
from ._firwin2 import firwin2
from ._freqs_ba import freqs_ba
from ._freqs_zpk import freqs_zpk

# Filter analysis functions (merged from filter_analysis)
from ._frequency_response import frequency_response
from ._frequency_response_fir import frequency_response_fir
from ._frequency_response_sos import frequency_response_sos
from ._frequency_response_zpk import frequency_response_zpk
from ._group_delay import group_delay, group_delay_sos
from ._iirnotch import iirnotch
from ._iirpeak import iirpeak
from ._impulse_response import (
    impulse_response,
    impulse_response_sos,
    step_response,
)
from ._kalman_filter import kalman_filter
from ._leaky_lms import leaky_lms
from ._lfilter import lfilter, lfiltic
from ._lfilter_zi import lfilter_zi
from ._lms import lms
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._minimum_phase import minimum_phase
from ._nlms import nlms
from ._remez import remez
from ._rls import rls
from ._savgol_coeffs import savgol_coeffs
from ._sos_normalize import sos_normalize
from ._sos_sections_count import sos_sections_count
from ._sos_to_ba import sos_to_ba
from ._sos_to_zpk import sos_to_zpk
from ._sosfilt import sosfilt
from ._sosfilt_zi import sosfilt_zi
from ._sosfiltfilt import sosfiltfilt
from ._yule_walker import yule_walker
from ._zpk_to_ba import zpk_to_ba
from ._zpk_to_sos import zpk_to_sos

__all__ = [
    # High-level interface
    "Filter",
    "FilterDesigner",
    # Batched design functions
    "batched_butterworth_design",
    "batched_chebyshev_type_1_design",
    "batched_chebyshev_type_2_design",
    "batched_filter_apply",
    "batched_firwin",
    # Design functions - Digital
    "bessel_design",
    "bessel_prototype",
    "butterworth_design",
    "butterworth_minimum_order",
    "butterworth_prototype",
    "chebyshev_type_1_design",
    "chebyshev_type_1_minimum_order",
    "chebyshev_type_1_prototype",
    "chebyshev_type_2_design",
    "chebyshev_type_2_minimum_order",
    "chebyshev_type_2_prototype",
    "elliptic_design",
    "elliptic_minimum_order",
    "elliptic_prototype",
    # Design functions - Analog
    "bessel_analog",
    "butterworth_analog",
    "butterworth_analog_bandpass_filter",
    "chebyshev_type_1_analog",
    "chebyshev_type_2_analog",
    "elliptic_analog",
    # FIR design functions
    "fftfilt",
    "filtfilt",
    "firwin",
    "firwin2",
    "freqs_ba",
    "freqs_zpk",
    "remez",
    "iirnotch",
    "iirpeak",
    "kalman_filter",
    "leaky_lms",
    "lfilter",
    "lfilter_zi",
    "lfiltic",
    "lms",
    "minimum_phase",
    "nlms",
    "rls",
    "savgol_coeffs",
    "yule_walker",
    # Transforms
    "bilinear_transform_ba",
    "bilinear_transform_zpk",
    "lowpass_to_bandpass_zpk",
    "lowpass_to_bandstop_zpk",
    "lowpass_to_highpass_zpk",
    "lowpass_to_lowpass_zpk",
    # Conversions
    "ba_to_sos",
    "ba_to_zpk",
    "sos_to_ba",
    "sos_to_zpk",
    "zpk_to_ba",
    "zpk_to_sos",
    # SOS utilities
    "cascade_sos",
    "sos_normalize",
    "sos_sections_count",
    "sosfilt",
    "sosfilt_zi",
    "sosfiltfilt",
    # Constants
    "Q_BUTTERWORTH",
    "Q_MEDIUM",
    "Q_NARROW",
    "Q_WIDE",
    # Exceptions
    "ConvergenceError",
    "FilterDesignError",
    "FrequencyOrderError",
    "InvalidCutoffError",
    "InvalidNumTapsError",
    "InvalidOrderError",
    "NyquistViolationError",
    "SOSNormalizationError",
    "SpecificationError",
    # Filter analysis (merged from filter_analysis)
    "frequency_response",
    "frequency_response_fir",
    "frequency_response_sos",
    "frequency_response_zpk",
    "group_delay",
    "group_delay_sos",
    "impulse_response",
    "impulse_response_sos",
    "step_response",
]
