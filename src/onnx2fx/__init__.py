# SPDX-License-Identifier: Apache-2.0
"""onnx2fx: Convert ONNX models to PyTorch FX GraphModules.

This library provides tools for converting ONNX models into PyTorch FX
GraphModules, enabling seamless integration with PyTorch's ecosystem for
optimization, analysis, and deployment.

Core Functions
--------------
convert : Convert an ONNX model to a PyTorch FX GraphModule.
make_trainable : Convert buffers to trainable parameters for training.

Model Analysis
--------------
analyze_model : Analyze an ONNX model for operator support.
AnalysisResult : Dataclass containing analysis results.

Operator Registration
---------------------
register_op : Register a custom operator handler.
unregister_op : Unregister an operator handler.
is_supported : Check if an operator is supported.
get_supported_ops : List supported operators for a domain.
get_all_supported_ops : Get all supported operators across all domains.
get_registered_domains : Get list of registered domains.

Exceptions
----------
Onnx2FxError : Base exception for all onnx2fx errors.
UnsupportedOpError : Raised when an operator is not supported.
ConversionError : Raised when conversion fails.
ValueNotFoundError : Raised when a value is not found in environment.

Example
-------
>>> import onnx
>>> from onnx2fx import convert
>>> model = onnx.load("model.onnx")
>>> fx_module = convert(model)
>>> # Use fx_module like any PyTorch module
>>> output = fx_module(input_tensor)
"""

from .converter import convert
from .exceptions import (
    Onnx2FxError,
    UnsupportedOpError,
    ConversionError,
    ValueNotFoundError,
    UnsupportedDTypeError,
    ExternalDataError,
    InferenceOnlyError,
)
from .op_registry import (
    register_op,
    unregister_op,
    get_supported_ops,
    get_all_supported_ops,
    get_registered_domains,
    is_supported,
)
from .utils.analyze import analyze_model, AnalysisResult
from .utils.training import make_trainable

__all__ = [
    # Core API
    "convert",
    # Training utilities
    "make_trainable",
    # Model analysis
    "analyze_model",
    "AnalysisResult",
    # Operator registration
    "register_op",
    "unregister_op",
    "get_supported_ops",
    "get_all_supported_ops",
    "get_registered_domains",
    "is_supported",
    # Exceptions
    "Onnx2FxError",
    "UnsupportedOpError",
    "ConversionError",
    "ValueNotFoundError",
    "UnsupportedDTypeError",
    "ExternalDataError",
    "InferenceOnlyError",
]
