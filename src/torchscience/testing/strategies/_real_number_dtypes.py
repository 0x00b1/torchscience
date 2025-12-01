import hypothesis.strategies
import torch

real_number_dtypes = hypothesis.strategies.sampled_from(
    [torch.float32, torch.float64]
)
