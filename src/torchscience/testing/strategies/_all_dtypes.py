import hypothesis.strategies
import torch

all_dtypes = hypothesis.strategies.sampled_from(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ]
)
