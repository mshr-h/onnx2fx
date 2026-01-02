# SPDX-License-Identifier: Apache-2.0
"""Arithmetic and element-wise operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Binary arithmetic operators
# =============================================================================


@register("Add")
def add(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise addition."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.add, args=(lhs, rhs))


@register("Sub")
def sub(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise subtraction."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.sub, args=(lhs, rhs))


@register("Mul")
def mul(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise multiplication."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.mul, args=(lhs, rhs))


@register("Div")
def div(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise division."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.div, args=(lhs, rhs))


@register("Pow")
def pow_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise power."""
    base = builder.get_value(node.input[0])
    exponent = builder.get_value(node.input[1])
    return builder.call_function(torch.pow, args=(base, exponent))


@register("Mod")
def mod(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise modulo."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    fmod = get_attribute(node, "fmod", 0)
    if fmod:
        return builder.call_function(torch.fmod, args=(lhs, rhs))
    return builder.call_function(torch.remainder, args=(lhs, rhs))


@register("Min")
def min_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise minimum of inputs."""
    result = builder.get_value(node.input[0])
    for i in range(1, len(node.input)):
        other = builder.get_value(node.input[i])
        result = builder.call_function(torch.minimum, args=(result, other))
    return result


@register("Max")
def max_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise maximum of inputs."""
    result = builder.get_value(node.input[0])
    for i in range(1, len(node.input)):
        other = builder.get_value(node.input[i])
        result = builder.call_function(torch.maximum, args=(result, other))
    return result


@register("Mean")
def mean(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise mean of inputs."""
    inputs = [builder.get_value(name) for name in node.input]
    # Stack and compute mean along first dimension
    stacked = builder.call_function(torch.stack, args=(inputs,))
    return builder.call_function(torch.mean, args=(stacked,), kwargs={"dim": 0})


@register("Sum")
def sum_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise sum of inputs."""
    result = builder.get_value(node.input[0])
    for i in range(1, len(node.input)):
        other = builder.get_value(node.input[i])
        result = builder.call_function(torch.add, args=(result, other))
    return result


# =============================================================================
# Unary arithmetic operators
# =============================================================================


@register("Neg")
def neg(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise negation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.neg, args=(x,))


@register("Abs")
def abs_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise absolute value."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.abs, args=(x,))


@register("Sign")
def sign(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise sign."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sign, args=(x,))


@register("Ceil")
def ceil(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise ceiling."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.ceil, args=(x,))


@register("Floor")
def floor(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise floor."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.floor, args=(x,))


@register("Round")
def round_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise rounding."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.round, args=(x,))


@register("Reciprocal")
def reciprocal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise reciprocal."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.reciprocal, args=(x,))


@register("Sqrt")
def sqrt(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise square root."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sqrt, args=(x,))


@register("Exp")
def exp(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise exponential."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.exp, args=(x,))


@register("Log")
def log(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise natural logarithm."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.log, args=(x,))


# =============================================================================
# Comparison operators
# =============================================================================


@register("Equal")
def equal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise equality comparison."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.eq, args=(lhs, rhs))


@register("Greater")
def greater(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise greater-than comparison."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.gt, args=(lhs, rhs))


@register("GreaterOrEqual")
def greater_or_equal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise greater-than-or-equal comparison."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.ge, args=(lhs, rhs))


@register("Less")
def less(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise less-than comparison."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.lt, args=(lhs, rhs))


@register("LessOrEqual")
def less_or_equal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise less-than-or-equal comparison."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.le, args=(lhs, rhs))


@register("And")
def and_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise logical AND."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.logical_and, args=(lhs, rhs))


@register("Or")
def or_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise logical OR."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.logical_or, args=(lhs, rhs))


@register("Xor")
def xor(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise logical XOR."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.logical_xor, args=(lhs, rhs))


@register("Not")
def not_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise logical NOT."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.logical_not, args=(x,))


@register("Where")
def where(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise conditional selection."""
    condition = builder.get_value(node.input[0])
    x = builder.get_value(node.input[1])
    y = builder.get_value(node.input[2])
    return builder.call_function(torch.where, args=(condition, x, y))


# =============================================================================
# Clip operator
# =============================================================================


@register("Clip", since_version=1)
def clip_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Clip tensor values to a range for opset 1-10.

    In opset < 11, min and max are required attributes.
    """
    x = builder.get_value(node.input[0])

    min_val = get_attribute(node, "min", float("-inf"))
    max_val = get_attribute(node, "max", float("inf"))

    return builder.call_function(
        torch.clamp, args=(x,), kwargs={"min": min_val, "max": max_val}
    )


@register("Clip", since_version=11)
def clip_v11(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Clip tensor values to a range for opset 11+.

    In opset 11+, min and max are optional inputs.
    """
    x = builder.get_value(node.input[0])

    min_val = None
    max_val = None

    if len(node.input) > 1 and node.input[1]:
        min_val = builder.get_value(node.input[1])
    if len(node.input) > 2 and node.input[2]:
        max_val = builder.get_value(node.input[2])

    return builder.call_function(
        torch.clamp, args=(x,), kwargs={"min": min_val, "max": max_val}
    )
