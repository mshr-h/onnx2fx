# SPDX-License-Identifier: Apache-2.0
"""ONNX operator implementations."""

# Import all operator modules to register handlers
from . import activation
from . import advanced
from . import arithmetic
from . import attention
from . import control_flow
from . import image
from . import loss
from . import misc
from . import nn
from . import quantization
from . import random
from . import reduction
from . import sequence
from . import tensor

__all__ = [
    "activation",
    "advanced",
    "arithmetic",
    "attention",
    "control_flow",
    "image",
    "loss",
    "misc",
    "nn",
    "quantization",
    "random",
    "reduction",
    "sequence",
    "tensor",
]
