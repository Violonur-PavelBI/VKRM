import numpy as np
import torch

NPTYPES = {
    "IEEE_FP16": np.float16,
    "IEEE_FP32": np.float32,
    "IEEE_FP64": np.float64,
    "IEEE_INT8": np.int8,
}
TTYPES = {
    torch.int8: "IEEE_INT8",
    torch.float32: "IEEE_FP32",
    torch.float64: "IEEE_FP64",
    torch.float16: "IEEE_FP16",
    torch.bfloat16: "BRA_FP16",
}

TORCH_TYPES = {
    torch.float16: "IEEE_FP16",
    torch.float32: "IEEE_FP32",
    torch.float64: "IEEE_FP64",
    torch.bfloat16: "BRA_FP16",
}
