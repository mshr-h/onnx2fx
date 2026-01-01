# SPDX-License-Identifier: Apache-2.0
"""Data type mapping between ONNX and PyTorch."""

from typing import Dict, Optional

import onnx
import torch

# ONNX TensorProto data type to PyTorch dtype mapping
DTYPE_MAP: Dict[int, torch.dtype] = {
    onnx.TensorProto.FLOAT: torch.float32,
    onnx.TensorProto.FLOAT16: torch.float16,
    onnx.TensorProto.BFLOAT16: torch.bfloat16,
    onnx.TensorProto.DOUBLE: torch.float64,
    onnx.TensorProto.INT8: torch.int8,
    onnx.TensorProto.INT16: torch.int16,
    onnx.TensorProto.INT32: torch.int32,
    onnx.TensorProto.INT64: torch.int64,
    onnx.TensorProto.UINT8: torch.uint8,
    onnx.TensorProto.BOOL: torch.bool,
    onnx.TensorProto.COMPLEX64: torch.complex64,
    onnx.TensorProto.COMPLEX128: torch.complex128,
}

# Reverse mapping: PyTorch dtype to ONNX TensorProto data type
TORCH_TO_ONNX_DTYPE: Dict[torch.dtype, int] = {v: k for k, v in DTYPE_MAP.items()}


def onnx_dtype_to_torch(onnx_dtype: int) -> Optional[torch.dtype]:
    """Convert ONNX TensorProto data type to PyTorch dtype.

    Parameters
    ----------
    onnx_dtype : int
        ONNX TensorProto data type enum value.

    Returns
    -------
    Optional[torch.dtype]
        Corresponding PyTorch dtype, or None if not supported.
    """
    return DTYPE_MAP.get(onnx_dtype)


def torch_dtype_to_onnx(torch_dtype: torch.dtype) -> Optional[int]:
    """Convert PyTorch dtype to ONNX TensorProto data type.

    Parameters
    ----------
    torch_dtype : torch.dtype
        PyTorch dtype.

    Returns
    -------
    Optional[int]
        Corresponding ONNX TensorProto data type, or None if not supported.
    """
    return TORCH_TO_ONNX_DTYPE.get(torch_dtype)


__all__ = [
    "DTYPE_MAP",
    "TORCH_TO_ONNX_DTYPE",
    "onnx_dtype_to_torch",
    "torch_dtype_to_onnx",
]
