import hypothesis.strategies
import torch


def available_devices() -> hypothesis.strategies.SearchStrategy[str]:
    """Strategy for available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return hypothesis.strategies.sampled_from(devices)
