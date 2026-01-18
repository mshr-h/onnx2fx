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


# =============================================================================
# Trigonometric functions
# =============================================================================


@register("Sin")
def sin(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sin, args=(x,))


@register("Cos")
def cos(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.cos, args=(x,))


@register("Tan")
def tan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tan, args=(x,))


@register("Asin")
def asin(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.asin, args=(x,))


@register("Acos")
def acos(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.acos, args=(x,))


@register("Atan")
def atan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.atan, args=(x,))


# =============================================================================
# Hyperbolic functions
# =============================================================================


@register("Sinh")
def sinh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hyperbolic sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sinh, args=(x,))


@register("Cosh")
def cosh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hyperbolic cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.cosh, args=(x,))


@register("Asinh")
def asinh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.asinh, args=(x,))


@register("Acosh")
def acosh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.acosh, args=(x,))


@register("Atanh")
def atanh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.atanh, args=(x,))


# =============================================================================
# Additional math functions
# =============================================================================


@register("Erf")
def erf(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Error function."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.erf, args=(x,))


@register("IsNaN")
def isnan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Check for NaN."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.isnan, args=(x,))


@register("IsInf")
def isinf(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Check for Inf."""
    x = builder.get_value(node.input[0])
    detect_negative = get_attribute(node, "detect_negative", 1)
    detect_positive = get_attribute(node, "detect_positive", 1)

    def _isinf(x, detect_neg, detect_pos):
        if detect_neg and detect_pos:
            return torch.isinf(x)
        elif detect_pos:
            return torch.isposinf(x)
        elif detect_neg:
            return torch.isneginf(x)
        else:
            return torch.zeros_like(x, dtype=torch.bool)

    return builder.call_function(_isinf, args=(x, detect_negative, detect_positive))


# =============================================================================
# Bitwise operations
# =============================================================================


@register("BitwiseAnd")
def bitwise_and(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise AND."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_and, args=(a, b))


@register("BitwiseOr")
def bitwise_or(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise OR."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_or, args=(a, b))


@register("BitwiseXor")
def bitwise_xor(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise XOR."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_xor, args=(a, b))


@register("BitwiseNot")
def bitwise_not(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise NOT."""
    x = builder.get_value(node.input[0])

    # PyTorch bitwise_not doesn't support some unsigned types (e.g., uint16) on CPU.
    # We handle this by casting to a signed type with the same bit width,
    # performing the operation, and casting back.
    def _bitwise_not(x):
        original_dtype = x.dtype
        # Map unsigned types to signed equivalents with same bit width
        dtype_map = {
            torch.uint16: torch.int16,
            torch.uint32: torch.int32,
        }
        if original_dtype in dtype_map:
            x = x.to(dtype_map[original_dtype])
            result = torch.bitwise_not(x)
            return result.to(original_dtype)
        return torch.bitwise_not(x)

    return builder.call_function(_bitwise_not, args=(x,))


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
