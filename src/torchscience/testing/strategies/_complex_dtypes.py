import hypothesis.strategies
import torch

complex_dtypes = hypothesis.strategies.sampled_from(
    [torch.complex64, torch.complex128]
)
