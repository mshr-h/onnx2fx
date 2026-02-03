# SPDX-License-Identifier: Apache-2.0
"""Training utilities for converted FX modules.

This module provides utilities to make converted ONNX models trainable
by converting buffers to trainable parameters.
"""

import torch
import torch.fx

from ..exceptions import InferenceOnlyError


def make_trainable(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Convert all buffers to trainable parameters.

    By default, ONNX initializers (weights) are registered as buffers in the
    converted FX module, making them non-trainable. This function converts
    all buffers to trainable parameters, enabling gradient computation and
    optimizer updates.

    Args:
        module: A converted FX GraphModule from onnx2fx.convert().

    Returns:
        The same module with buffers converted to parameters (modified in-place).

    Example:
        >>> import onnx
        >>> from onnx2fx import convert, make_trainable
        >>> onnx_model = onnx.load("model.onnx")
        >>> fx_module = convert(onnx_model)
        >>> fx_module = make_trainable(fx_module)
        >>> optimizer = torch.optim.SGD(fx_module.parameters(), lr=0.01)
    """
    if getattr(module, "_onnx2fx_inference_only", False):
        raise InferenceOnlyError(
            "make_trainable is not supported for memmap-based inference-only models"
        )

    # Collect all buffer names and tensors first to avoid modifying dict during iteration
    buffers_to_convert = list(module.named_buffers())

    for name, buf in buffers_to_convert:
        # Delete the buffer
        delattr(module, name)
        # Only floating point tensors can require gradients
        if buf.is_floating_point() or buf.is_complex():
            module.register_parameter(name, torch.nn.Parameter(buf.clone()))
        else:
            # Re-register non-floating point tensors as buffers (e.g., int64 indices)
            module.register_buffer(name, buf.clone())

    return module
