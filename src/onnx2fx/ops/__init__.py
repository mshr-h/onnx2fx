# SPDX-License-Identifier: Apache-2.0
"""ONNX operator implementations."""

# Import all operator modules to register handlers
from . import activation
from . import advanced
from . import arithmetic
from . import attention
from . import control
from . import loss
from . import nn
from . import quantization
from . import reduction
from . import sequence
from . import tensor

__all__ = [
    "activation",
    "advanced",
    "arithmetic",
    "attention",
    "control",
    "loss",
    "nn",
    "quantization",
    "reduction",
    "sequence",
    "tensor",
]
