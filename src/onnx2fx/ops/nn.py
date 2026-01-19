# SPDX-License-Identifier: Apache-2.0
"""Neural network layer operators.

This module contains core neural network operators like MatMul, Gemm, and Dropout.
Other neural network operators are organized in specialized modules:
- convolution.py: Conv, ConvTranspose, DeformConv
- pooling.py: MaxPool, AveragePool, GlobalAveragePool, etc.
- normalization.py: BatchNormalization, LayerNormalization, etc.
- recurrent.py: LSTM, GRU, RNN
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Matrix multiplication operators
# =============================================================================


@register("MatMul")
def matmul(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Matrix multiplication."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.matmul, args=(a, b))


@register("Gemm")
def gemm(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """General Matrix Multiplication: Y = alpha * A' * B' + beta * C."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])

    alpha = get_attribute(node, "alpha", 1.0)
    beta = get_attribute(node, "beta", 1.0)
    trans_a = get_attribute(node, "transA", 0)
    trans_b = get_attribute(node, "transB", 0)

    def _gemm(a, b, c, alpha, beta, trans_a, trans_b):
        if trans_a:
            a = a.T
        if trans_b:
            b = b.T
        result = alpha * torch.matmul(a, b)
        if c is not None:
            result = result + beta * c
        return result

    c = None
    if len(node.input) > 2 and node.input[2]:
        c = builder.get_value(node.input[2])

    return builder.call_function(_gemm, args=(a, b, c, alpha, beta, trans_a, trans_b))


# =============================================================================
# Dropout and regularization
# =============================================================================


@register("Dropout")
def dropout(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Dropout (inference mode - identity).

    ONNX Dropout can have 2 outputs:
    - output: The result after dropout (same as input in inference mode)
    - mask (optional): Boolean mask indicating which elements were kept (all True in inference mode)
    """
    x = builder.get_value(node.input[0])

    # Check if mask output is requested (second output)
    return_mask = len(node.output) > 1 and node.output[1] != ""

    # In inference mode, dropout is identity
    # ratio = get_attribute(node, "ratio", 0.5)
    # training_mode from input or default to False

    def _dropout_with_mask(x):
        # In inference mode, output is identity and mask is all True
        output = x
        mask = torch.ones_like(x, dtype=torch.bool)
        return output, mask

    if return_mask:
        return builder.call_function(_dropout_with_mask, args=(x,))
    else:
        # For inference without mask, just return input
        return builder.call_function(lambda t: t, args=(x,))
