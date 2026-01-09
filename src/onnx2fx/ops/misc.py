# SPDX-License-Identifier: Apache-2.0
"""Miscellaneous operators.

This module implements various ONNX operators that don't fit into
other categories, including Optional, Select, Compress, BitShift,
and ConstantOfShape operators.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Optional operators
# =============================================================================


@register("Optional")
def optional_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an optional value."""
    if len(node.input) > 0 and node.input[0]:
        # Has a value
        return builder.get_value(node.input[0])
    else:
        # Empty optional - return None wrapped
        return builder.call_function(lambda: None, args=())


@register("OptionalHasElement")
def optional_has_element(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Check if optional has a value."""
    opt = builder.get_value(node.input[0])

    def _has_element(x) -> torch.Tensor:
        return torch.tensor(x is not None, dtype=torch.bool)

    return builder.call_function(_has_element, args=(opt,))


@register("OptionalGetElement")
def optional_get_element(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Get the value from an optional."""
    opt = builder.get_value(node.input[0])

    def _get_element(x):
        if x is None:
            raise RuntimeError("Optional has no element")
        return x

    return builder.call_function(_get_element, args=(opt,))


# =============================================================================
# Select operator
# =============================================================================


@register("Select")
def select_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Select elements based on indices (like advanced indexing)."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    return builder.call_function(torch.index_select, args=(data, 0, indices))


# =============================================================================
# Compress operator
# =============================================================================


@register("Compress")
def compress_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Select elements based on a boolean condition tensor."""
    data = builder.get_value(node.input[0])
    condition = builder.get_value(node.input[1])

    axis = get_attribute(node, "axis", None)

    if axis is not None:

        def _compress_axis(d: torch.Tensor, c: torch.Tensor, ax: int) -> torch.Tensor:
            # Get indices where condition is True
            indices = torch.nonzero(c, as_tuple=True)[0]
            return torch.index_select(d, ax, indices)

        return builder.call_function(_compress_axis, args=(data, condition, axis))
    else:
        # Flatten and compress
        def _compress_flat(d: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return d.flatten()[c.flatten().bool()]

        return builder.call_function(_compress_flat, args=(data, condition))


# =============================================================================
# Bitwise shifts
# =============================================================================


@register("BitShift")
def bit_shift(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise shift operation."""
    x = builder.get_value(node.input[0])
    y = builder.get_value(node.input[1])

    direction = get_attribute(node, "direction", "LEFT")

    if direction == "LEFT":
        return builder.call_function(torch.bitwise_left_shift, args=(x, y))
    else:
        return builder.call_function(torch.bitwise_right_shift, args=(x, y))


# =============================================================================
# String operators (limited support)
# =============================================================================


@register("StringNormalizer")
def string_normalizer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """String normalization - passthrough for tensors."""
    x = builder.get_value(node.input[0])
    # String operations are not well supported in PyTorch tensors
    # Return as-is
    return x
