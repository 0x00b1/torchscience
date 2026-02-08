import hypothesis.strategies
import torch

low_precision_dtypes = hypothesis.strategies.sampled_from(
    [torch.float16, torch.bfloat16]
)
