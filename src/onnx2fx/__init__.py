# SPDX-License-Identifier: Apache-2.0
from .converter import convert
from .exceptions import (
    Onnx2FxError,
    UnsupportedOpError,
    ConversionError,
    ValueNotFoundError,
)
from .op_registry import (
    register_custom_op,
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
    "register_custom_op",
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
]
