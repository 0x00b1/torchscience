import hypothesis.strategies
import torch

all_complex_dtypes = hypothesis.strategies.sampled_from(
    [torch.complex64, torch.complex128]
)
