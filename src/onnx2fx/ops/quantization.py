# SPDX-License-Identifier: Apache-2.0
"""Quantization operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Basic quantization operators
# =============================================================================


@register("QuantizeLinear")
def quantize_linear(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantize input tensor using scale and zero_point."""
    x = builder.get_value(node.input[0])
    y_scale = builder.get_value(node.input[1])
    y_zero_point = builder.get_value(node.input[2]) if len(node.input) > 2 else None

    if y_zero_point is not None:

        def _quantize_uint8(
            inp: torch.Tensor, s: torch.Tensor, zp: torch.Tensor
        ) -> torch.Tensor:
            return torch.clamp(torch.round(inp / s) + zp.float(), 0, 255).to(
                torch.uint8
            )

        return builder.call_function(_quantize_uint8, args=(x, y_scale, y_zero_point))
    else:

        def _quantize_int8(inp: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.clamp(torch.round(inp / s), -128, 127).to(torch.int8)

        return builder.call_function(_quantize_int8, args=(x, y_scale))


@register("DequantizeLinear")
def dequantize_linear(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Dequantize input tensor using scale and zero_point."""
    x = builder.get_value(node.input[0])
    x_scale = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2]) if len(node.input) > 2 else None

    if x_zero_point is not None:

        def _dequantize(
            inp: torch.Tensor, s: torch.Tensor, zp: torch.Tensor
        ) -> torch.Tensor:
            return (inp.float() - zp.float()) * s

        return builder.call_function(_dequantize, args=(x, x_scale, x_zero_point))
    else:

        def _dequantize_no_zp(inp: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return inp.float() * s

        return builder.call_function(_dequantize_no_zp, args=(x, x_scale))


@register("DynamicQuantizeLinear")
def dynamic_quantize_linear(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Dynamic quantization of input tensor to uint8.

    Returns tuple of (y, y_scale, y_zero_point).
    """
    x = builder.get_value(node.input[0])

    def _dynamic_quantize(
        inp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min = torch.min(inp)
        x_max = torch.max(inp)
        scale = (x_max - x_min) / 255.0
        zero_point = torch.clamp(torch.round(-x_min / scale), 0, 255).to(torch.uint8)
        y = torch.clamp(torch.round(inp / scale) + zero_point.float(), 0, 255).to(
            torch.uint8
        )
        return y, scale, zero_point

    return builder.call_function(_dynamic_quantize, args=(x,))


# =============================================================================
# QLinear operators
# =============================================================================


@register("QLinearMatMul")
def qlinear_matmul(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized MatMul with scales and zero points."""
    a = builder.get_value(node.input[0])
    a_scale = builder.get_value(node.input[1])
    a_zero_point = builder.get_value(node.input[2])
    b = builder.get_value(node.input[3])
    b_scale = builder.get_value(node.input[4])
    b_zero_point = builder.get_value(node.input[5])
    y_scale = builder.get_value(node.input[6])
    y_zero_point = builder.get_value(node.input[7])

    def _qlinear_matmul(
        a: torch.Tensor,
        a_s: torch.Tensor,
        a_zp: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        b_zp: torch.Tensor,
        y_s: torch.Tensor,
        y_zp: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize
        a_dq = (a.float() - a_zp.float()) * a_s
        b_dq = (b.float() - b_zp.float()) * b_s
        # MatMul
        result = torch.matmul(a_dq, b_dq)
        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_matmul,
        args=(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
        ),
    )


@register("QLinearConv")
def qlinear_conv(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized 2D convolution with scales and zero points.

    Inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, [B]
    """
    x = builder.get_value(node.input[0])
    x_scale = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2])
    w = builder.get_value(node.input[3])
    w_scale = builder.get_value(node.input[4])
    w_zero_point = builder.get_value(node.input[5])
    y_scale = builder.get_value(node.input[6])
    y_zero_point = builder.get_value(node.input[7])
    bias = builder.get_value(node.input[8]) if len(node.input) > 8 else None

    # Get convolution attributes
    # Note: kernel_shape is inferred from weight tensor, not from attribute
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
    dilations = get_attribute(node, "dilations", [1, 1])
    group = get_attribute(node, "group", 1)
    pads = get_attribute(node, "pads", [0, 0, 0, 0])
    strides = get_attribute(node, "strides", [1, 1])

    if auto_pad != "NOTSET":
        # Handle auto_pad - for simplicity, assume SAME_UPPER
        pass

    # Convert pads from ONNX format [H_begin, W_begin, H_end, W_end] to PyTorch format
    if len(pads) == 4:
        padding = (pads[0], pads[1])  # Symmetric padding
    else:
        padding = tuple(pads)

    def _qlinear_conv(
        x: torch.Tensor,
        x_s: torch.Tensor,
        x_zp: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        w_zp: torch.Tensor,
        y_s: torch.Tensor,
        y_zp: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
    ) -> torch.Tensor:
        # Dequantize input and weight
        x_dq = (x.float() - x_zp.float()) * x_s
        w_dq = (w.float() - w_zp.float()) * w_s

        # Perform convolution
        result = torch.nn.functional.conv2d(
            x_dq,
            w_dq,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_conv,
        args=(
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            y_scale,
            y_zero_point,
            bias,
            tuple(strides),
            padding,
            tuple(dilations),
            group,
        ),
    )


@register("QLinearAdd")
def qlinear_add(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized addition with scales and zero points (com.microsoft domain).

    Inputs: A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point
    """
    a = builder.get_value(node.input[0])
    a_scale = builder.get_value(node.input[1])
    a_zero_point = builder.get_value(node.input[2])
    b = builder.get_value(node.input[3])
    b_scale = builder.get_value(node.input[4])
    b_zero_point = builder.get_value(node.input[5])
    c_scale = builder.get_value(node.input[6])
    c_zero_point = builder.get_value(node.input[7])

    def _qlinear_add(
        a: torch.Tensor,
        a_s: torch.Tensor,
        a_zp: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        b_zp: torch.Tensor,
        c_s: torch.Tensor,
        c_zp: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize
        a_dq = (a.float() - a_zp.float()) * a_s
        b_dq = (b.float() - b_zp.float()) * b_s
        # Add
        result = a_dq + b_dq
        # Quantize output
        return torch.clamp(torch.round(result / c_s) + c_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_add,
        args=(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            c_scale,
            c_zero_point,
        ),
    )


@register("QLinearMul")
def qlinear_mul(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized multiplication with scales and zero points."""
    a = builder.get_value(node.input[0])
    a_scale = builder.get_value(node.input[1])
    a_zero_point = builder.get_value(node.input[2])
    b = builder.get_value(node.input[3])
    b_scale = builder.get_value(node.input[4])
    b_zero_point = builder.get_value(node.input[5])
    c_scale = builder.get_value(node.input[6])
    c_zero_point = builder.get_value(node.input[7])

    def _qlinear_mul(
        a: torch.Tensor,
        a_s: torch.Tensor,
        a_zp: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        b_zp: torch.Tensor,
        c_s: torch.Tensor,
        c_zp: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize
        a_dq = (a.float() - a_zp.float()) * a_s
        b_dq = (b.float() - b_zp.float()) * b_s
        # Multiply
        result = a_dq * b_dq
        # Quantize output
        return torch.clamp(torch.round(result / c_s) + c_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_mul,
        args=(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            c_scale,
            c_zero_point,
        ),
    )


@register("QLinearSigmoid")
def qlinear_sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized sigmoid."""
    x = builder.get_value(node.input[0])
    x_scale = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2])
    y_scale = builder.get_value(node.input[3])
    y_zero_point = builder.get_value(node.input[4])

    def _qlinear_sigmoid(
        x: torch.Tensor,
        x_s: torch.Tensor,
        x_zp: torch.Tensor,
        y_s: torch.Tensor,
        y_zp: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize
        x_dq = (x.float() - x_zp.float()) * x_s
        # Sigmoid
        result = torch.sigmoid(x_dq)
        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_sigmoid,
        args=(x, x_scale, x_zero_point, y_scale, y_zero_point),
    )


@register("QLinearLeakyRelu")
def qlinear_leaky_relu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Quantized Leaky ReLU."""
    x = builder.get_value(node.input[0])
    x_scale = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2])
    y_scale = builder.get_value(node.input[3])
    y_zero_point = builder.get_value(node.input[4])
    alpha = get_attribute(node, "alpha", 0.01)

    def _qlinear_leaky_relu(
        x: torch.Tensor,
        x_s: torch.Tensor,
        x_zp: torch.Tensor,
        y_s: torch.Tensor,
        y_zp: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        # Dequantize
        x_dq = (x.float() - x_zp.float()) * x_s
        # LeakyReLU
        result = torch.nn.functional.leaky_relu(x_dq, negative_slope=alpha)
        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_leaky_relu,
        args=(x, x_scale, x_zero_point, y_scale, y_zero_point, alpha),
    )


@register("QLinearGlobalAveragePool")
def qlinear_global_avg_pool(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Quantized Global Average Pooling."""
    x = builder.get_value(node.input[0])
    x_scale = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2])
    y_scale = builder.get_value(node.input[3])
    y_zero_point = builder.get_value(node.input[4])

    def _qlinear_global_avg_pool(
        x: torch.Tensor,
        x_s: torch.Tensor,
        x_zp: torch.Tensor,
        y_s: torch.Tensor,
        y_zp: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize
        x_dq = (x.float() - x_zp.float()) * x_s
        # Global Average Pool
        result = torch.nn.functional.adaptive_avg_pool2d(x_dq, (1, 1))
        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_global_avg_pool,
        args=(x, x_scale, x_zero_point, y_scale, y_zero_point),
    )


# =============================================================================
# Integer arithmetic operators
# =============================================================================


@register("ConvInteger")
def conv_integer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Integer convolution (returns int32)."""
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2]) if len(node.input) > 2 else None
    w_zero_point = builder.get_value(node.input[3]) if len(node.input) > 3 else None

    # Get convolution attributes
    # Note: auto_pad is not implemented; use explicit pads instead
    dilations = get_attribute(node, "dilations", [1, 1])
    group = get_attribute(node, "group", 1)
    pads = get_attribute(node, "pads", [0, 0, 0, 0])
    strides = get_attribute(node, "strides", [1, 1])

    if len(pads) == 4:
        padding = (pads[0], pads[1])
    else:
        padding = tuple(pads)

    def _conv_integer(
        x: torch.Tensor,
        w: torch.Tensor,
        x_zp: torch.Tensor | None,
        w_zp: torch.Tensor | None,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
    ) -> torch.Tensor:
        # Subtract zero points
        x_int = x.int()
        w_int = w.int()
        if x_zp is not None:
            x_int = x_int - x_zp.int()
        if w_zp is not None:
            w_int = w_int - w_zp.int()

        # Perform convolution in float (PyTorch doesn't support int conv)
        result = torch.nn.functional.conv2d(
            x_int.float(),
            w_int.float(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        return result.int()

    return builder.call_function(
        _conv_integer,
        args=(
            x,
            w,
            x_zero_point,
            w_zero_point,
            tuple(strides),
            padding,
            tuple(dilations),
            group,
        ),
    )


@register("MatMulInteger")
def matmul_integer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Integer matrix multiplication (returns int32)."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    a_zero_point = builder.get_value(node.input[2]) if len(node.input) > 2 else None
    b_zero_point = builder.get_value(node.input[3]) if len(node.input) > 3 else None

    def _matmul_integer(
        a: torch.Tensor,
        b: torch.Tensor,
        a_zp: torch.Tensor | None,
        b_zp: torch.Tensor | None,
    ) -> torch.Tensor:
        a_int = a.int()
        b_int = b.int()
        if a_zp is not None:
            a_int = a_int - a_zp.int()
        if b_zp is not None:
            b_int = b_int - b_zp.int()
        return torch.matmul(a_int.float(), b_int.float()).int()

    return builder.call_function(
        _matmul_integer,
        args=(a, b, a_zero_point, b_zero_point),
    )
