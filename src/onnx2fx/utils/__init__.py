# SPDX-License-Identifier: Apache-2.0
"""Utility modules for ONNX model handling.

This package provides utilities for:
- ONNX attribute parsing
- ONNX to PyTorch data type mapping
- ONNX model analysis
- Name sanitization for valid Python identifiers
"""

from .dtype import DTYPE_MAP, onnx_dtype_to_torch
from .attributes import get_attribute, get_attributes
from .analyze import analyze_model, AnalysisResult
from .training import make_trainable
from .names import sanitize_name

__all__ = [
    "DTYPE_MAP",
    "onnx_dtype_to_torch",
    "get_attribute",
    "get_attributes",
    "analyze_model",
    "AnalysisResult",
    "make_trainable",
    "sanitize_name",
]
