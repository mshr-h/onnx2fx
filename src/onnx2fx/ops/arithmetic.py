# SPDX-License-Identifier: Apache-2.0
"""Arithmetic and element-wise operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import binary_op, get_optional_input, unary_op

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Binary arithmetic operators
# =============================================================================


register("Add")(binary_op(torch.add, "Element-wise addition."))
register("Sub")(binary_op(torch.sub, "Element-wise subtraction."))
register("Mul")(binary_op(torch.mul, "Element-wise multiplication."))
register("Pow")(binary_op(torch.pow, "Element-wise power."))


def _onnx_div(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """ONNX-compatible division.

    For integer types, ONNX Div truncates toward zero (like C integer division),
    and the result must have the same integer dtype as the inputs.
    For floating-point types, it performs standard division.
    """
    if not lhs.dtype.is_floating_point and not lhs.dtype.is_complex:
        # Integer types: use truncation toward zero
        return torch.div(lhs, rhs, rounding_mode="trunc")
    return torch.div(lhs, rhs)


@register("Div")
def div(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Element-wise division."""
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(_onnx_div, args=(lhs, rhs))


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


register("Neg")(unary_op(torch.neg, "Element-wise negation."))
register("Abs")(unary_op(torch.abs, "Element-wise absolute value."))
register("Sign")(unary_op(torch.sign, "Element-wise sign."))
register("Ceil")(unary_op(torch.ceil, "Element-wise ceiling."))
register("Floor")(unary_op(torch.floor, "Element-wise floor."))
register("Round")(unary_op(torch.round, "Element-wise rounding."))
register("Reciprocal")(unary_op(torch.reciprocal, "Element-wise reciprocal."))
register("Sqrt")(unary_op(torch.sqrt, "Element-wise square root."))
register("Exp")(unary_op(torch.exp, "Element-wise exponential."))
register("Log")(unary_op(torch.log, "Element-wise natural logarithm."))


# =============================================================================
# Comparison operators
# =============================================================================


register("Equal")(binary_op(torch.eq, "Element-wise equality comparison."))
register("Greater")(binary_op(torch.gt, "Element-wise greater-than comparison."))
register("GreaterOrEqual")(
    binary_op(torch.ge, "Element-wise greater-than-or-equal comparison.")
)
register("Less")(binary_op(torch.lt, "Element-wise less-than comparison."))
register("LessOrEqual")(
    binary_op(torch.le, "Element-wise less-than-or-equal comparison.")
)
register("And")(binary_op(torch.logical_and, "Element-wise logical AND."))
register("Or")(binary_op(torch.logical_or, "Element-wise logical OR."))
register("Xor")(binary_op(torch.logical_xor, "Element-wise logical XOR."))
register("Not")(unary_op(torch.logical_not, "Element-wise logical NOT."))


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

    min_val = get_optional_input(builder, node, 1)
    max_val = get_optional_input(builder, node, 2)

    return builder.call_function(
        torch.clamp, args=(x,), kwargs={"min": min_val, "max": max_val}
    )


# =============================================================================
# Trigonometric functions
# =============================================================================


register("Sin")(unary_op(torch.sin, "Sine."))
register("Cos")(unary_op(torch.cos, "Cosine."))
register("Tan")(unary_op(torch.tan, "Tangent."))
register("Asin")(unary_op(torch.asin, "Arc sine."))
register("Acos")(unary_op(torch.acos, "Arc cosine."))
register("Atan")(unary_op(torch.atan, "Arc tangent."))


# =============================================================================
# Hyperbolic functions
# =============================================================================


register("Sinh")(unary_op(torch.sinh, "Hyperbolic sine."))
register("Cosh")(unary_op(torch.cosh, "Hyperbolic cosine."))
register("Asinh")(unary_op(torch.asinh, "Inverse hyperbolic sine."))
register("Acosh")(unary_op(torch.acosh, "Inverse hyperbolic cosine."))
register("Atanh")(unary_op(torch.atanh, "Inverse hyperbolic tangent."))


# =============================================================================
# Additional math functions
# =============================================================================


register("Erf")(unary_op(torch.erf, "Error function."))
register("IsNaN")(unary_op(torch.isnan, "Check for NaN."))


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


register("BitwiseAnd")(binary_op(torch.bitwise_and, "Bitwise AND."))
register("BitwiseOr")(binary_op(torch.bitwise_or, "Bitwise OR."))
register("BitwiseXor")(binary_op(torch.bitwise_xor, "Bitwise XOR."))


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
