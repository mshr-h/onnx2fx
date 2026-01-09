# SPDX-License-Identifier: Apache-2.0
"""Miscellaneous operators.

This module implements ONNX operators that don't fit into other categories.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


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
