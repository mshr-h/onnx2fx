# SPDX-License-Identifier: Apache-2.0
"""Utility modules for ONNX model handling.

This package provides utilities for:
- ONNX attribute parsing
- ONNX to PyTorch data type mapping
- ONNX model analysis
"""

from .dtype import DTYPE_MAP, onnx_dtype_to_torch
from .attributes import get_attribute, get_attributes
from .analyze import analyze_model, AnalysisResult

__all__ = [
    "DTYPE_MAP",
    "onnx_dtype_to_torch",
    "get_attribute",
    "get_attributes",
    "analyze_model",
    "AnalysisResult",
]
