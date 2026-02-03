# SPDX-License-Identifier: Apache-2.0
"""Main entry point for converting ONNX models to PyTorch FX.

This module provides the primary `convert` function for transforming
ONNX models into equivalent PyTorch FX GraphModules.
"""

import os
from typing import Optional, Union

import onnx
import torch

from .graph_builder import GraphBuilder
from .utils.external_data import validate_external_data_model


def convert(
    model: Union[onnx.ModelProto, str],
    *,
    base_dir: Optional[str] = None,
    memmap_external_data: bool = False,
) -> torch.fx.GraphModule:
    """Convert an ONNX model into a ``torch.fx.GraphModule``.

    Parameters
    ----------
    model : Union[onnx.ModelProto, str]
        Either an in-memory ``onnx.ModelProto`` or a file path to an ONNX model.
    base_dir : Optional[str], optional
        Base directory for resolving external data tensors. Required when
        ``memmap_external_data=True`` and a relative external data path is used.
    memmap_external_data : bool, optional
        If True, do not load external data into memory. Instead, keep external
        data references for memmap-based loading during conversion.

    Returns
    -------
    torch.fx.GraphModule
        A PyTorch FX Graph module.
    """

    if isinstance(model, str):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(model))
        if memmap_external_data:
            model = onnx.load_model(model, load_external_data=False)
        else:
            model = onnx.load_model(model)
    elif isinstance(model, onnx.ModelProto):
        model = model
    else:
        raise TypeError("model must be a path or onnx.ModelProto")

    if memmap_external_data:
        validate_external_data_model(model, base_dir=base_dir, strict=True)

    return GraphBuilder(
        model,
        base_dir=base_dir,
        memmap_external_data=memmap_external_data,
    ).build()
