"""Signal processing operations."""

from torchscience.signal_processing import (
    filter,
    noise,
    spectral_estimation,
    waveform,
)
from torchscience.signal_processing._constants import (
    SAMPLE_RATE_CD,
    SAMPLE_RATE_DAT,
    SAMPLE_RATE_DEFAULT_AUDIO,
    SAMPLE_RATE_HIGH_RES,
    SAMPLE_RATE_PROFESSIONAL,
    SAMPLE_RATE_TELEPHONY,
    SAMPLE_RATE_WIDEBAND,
)

__all__ = [
    # Submodules
    "filter",
    "noise",
    "spectral_estimation",
    "waveform",
    # Constants
    "SAMPLE_RATE_CD",
    "SAMPLE_RATE_DAT",
    "SAMPLE_RATE_DEFAULT_AUDIO",
    "SAMPLE_RATE_HIGH_RES",
    "SAMPLE_RATE_PROFESSIONAL",
    "SAMPLE_RATE_TELEPHONY",
    "SAMPLE_RATE_WIDEBAND",
]
