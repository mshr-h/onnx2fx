# SPDX-License-Identifier: Apache-2.0
"""Main entry point for converting ONNX models to PyTorch FX.

This module provides the primary `convert` function for transforming
ONNX models into equivalent PyTorch FX GraphModules.
"""

from typing import Union

import onnx
import torch

from .graph_builder import GraphBuilder


def convert(
    model: Union[onnx.ModelProto, str],
) -> torch.fx.GraphModule:
    """Convert an ONNX model into a ``torch.fx.GraphModule``.

    Parameters
    ----------
    model : Union[onnx.ModelProto, str]
        Either an in-memory ``onnx.ModelProto`` or a file path to an ONNX model.

    Returns
    -------
    torch.fx.GraphModule
        A PyTorch FX Graph module.
    """

    if isinstance(model, str):
        model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        model = model
    else:
        raise TypeError("model must be a path or onnx.ModelProto")

    return GraphBuilder(model).build()
