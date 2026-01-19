# SPDX-License-Identifier: Apache-2.0
"""Linear algebra operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Linear algebra operators
# =============================================================================


@register("Einsum")
def einsum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Einstein summation."""
    equation = get_attribute(node, "equation")
    inputs = [builder.get_value(name) for name in node.input]
    return builder.call_function(torch.einsum, args=(equation, *inputs))


@register("Det")
def det(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Matrix determinant."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.linalg.det, args=(x,))
