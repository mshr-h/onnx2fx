# SPDX-License-Identifier: Apache-2.0
"""Attention and Transformer related operators."""

import math
from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Softmax variants
# =============================================================================


@register("LogSoftmax")
def log_softmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log of softmax."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)
    return builder.call_function(
        torch.nn.functional.log_softmax, args=(x,), kwargs={"dim": axis}
    )


@register("Hardmax")
def hardmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hardmax - one-hot encoding of argmax."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)

    def _hardmax(t: torch.Tensor, ax: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.argmax(t, dim=ax), num_classes=t.shape[ax]
        ).to(t.dtype)

    return builder.call_function(_hardmax, args=(x, axis))


# =============================================================================
# Quantization operators
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
        args=(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point),
    )


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
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
    dilations = get_attribute(node, "dilations", [1, 1])
    group = get_attribute(node, "group", 1)
    kernel_shape = get_attribute(node, "kernel_shape")
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
            x_dq, w_dq, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

        # Quantize output
        return torch.clamp(torch.round(result / y_s) + y_zp.float(), 0, 255).to(
            torch.uint8
        )

    return builder.call_function(
        _qlinear_conv,
        args=(
            x, x_scale, x_zero_point, w, w_scale, w_zero_point,
            y_scale, y_zero_point, bias,
            tuple(strides), padding, tuple(dilations), group
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
        args=(a, a_scale, a_zero_point, b, b_scale, b_zero_point, c_scale, c_zero_point),
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
        args=(a, a_scale, a_zero_point, b, b_scale, b_zero_point, c_scale, c_zero_point),
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
def qlinear_global_avg_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
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


@register("ConvInteger")
def conv_integer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Integer convolution (returns int32)."""
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    x_zero_point = builder.get_value(node.input[2]) if len(node.input) > 2 else None
    w_zero_point = builder.get_value(node.input[3]) if len(node.input) > 3 else None

    # Get convolution attributes
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
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
            x_int.float(), w_int.float(),
            stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        return result.int()

    return builder.call_function(
        _conv_integer,
        args=(x, w, x_zero_point, w_zero_point, tuple(strides), padding, tuple(dilations), group),
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


# =============================================================================
# Loss operators
# =============================================================================


@register("SoftmaxCrossEntropyLoss")
def softmax_cross_entropy_loss(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Softmax cross entropy loss."""
    scores = builder.get_value(node.input[0])
    labels = builder.get_value(node.input[1])
    weights = (
        builder.get_value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    )

    ignore_index = get_attribute(node, "ignore_index", -100)
    reduction = get_attribute(node, "reduction", "mean")

    kwargs = {"ignore_index": ignore_index, "reduction": reduction}
    if weights is not None:
        kwargs["weight"] = weights

    return builder.call_function(
        torch.nn.functional.cross_entropy, args=(scores, labels), kwargs=kwargs
    )


@register("NegativeLogLikelihoodLoss")
def negative_log_likelihood_loss(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Negative log likelihood loss."""
    input_node = builder.get_value(node.input[0])
    target = builder.get_value(node.input[1])
    weight = (
        builder.get_value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    )

    ignore_index = get_attribute(node, "ignore_index", -100)
    reduction = get_attribute(node, "reduction", "mean")

    kwargs = {"ignore_index": ignore_index, "reduction": reduction}
    if weight is not None:
        kwargs["weight"] = weight

    return builder.call_function(
        torch.nn.functional.nll_loss, args=(input_node, target), kwargs=kwargs
    )


# =============================================================================
# Sequence operators
# =============================================================================


@register("SequenceConstruct")
def sequence_construct(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Construct a sequence (list) from input tensors."""
    inputs = [builder.get_value(name) for name in node.input]
    return builder.call_function(lambda *args: list(args), args=tuple(inputs))


@register("SequenceAt")
def sequence_at(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get element at position from sequence."""
    seq = builder.get_value(node.input[0])
    position = builder.get_value(node.input[1])

    def _seq_at(s: list, p: torch.Tensor) -> torch.Tensor:
        idx = int(p.item()) if hasattr(p, "item") else int(p)
        return s[idx]

    return builder.call_function(_seq_at, args=(seq, position))


@register("SequenceLength")
def sequence_length(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get length of sequence."""
    seq = builder.get_value(node.input[0])

    def _seq_len(s: list) -> torch.Tensor:
        return torch.tensor(len(s), dtype=torch.int64)

    return builder.call_function(_seq_len, args=(seq,))


@register("SequenceEmpty")
def sequence_empty(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an empty sequence."""
    return builder.call_function(list, args=())


@register("SequenceInsert")
def sequence_insert(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Insert tensor into sequence at position."""
    seq = builder.get_value(node.input[0])
    tensor = builder.get_value(node.input[1])
    position = builder.get_value(node.input[2]) if len(node.input) > 2 else None

    if position is not None:

        def _seq_insert(s: list, t: torch.Tensor, p: torch.Tensor) -> list:
            idx = int(p.item()) if hasattr(p, "item") else int(p)
            return s[:idx] + [t] + s[idx:]

        return builder.call_function(_seq_insert, args=(seq, tensor, position))
    else:

        def _seq_append(s: list, t: torch.Tensor) -> list:
            return s + [t]

        return builder.call_function(_seq_append, args=(seq, tensor))


@register("SequenceErase")
def sequence_erase(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Remove element from sequence at position."""
    seq = builder.get_value(node.input[0])
    position = builder.get_value(node.input[1]) if len(node.input) > 1 else None

    if position is not None:

        def _seq_erase(s: list, p: torch.Tensor) -> list:
            idx = int(p.item()) if hasattr(p, "item") else int(p)
            return s[:idx] + s[idx + 1 :]

        return builder.call_function(_seq_erase, args=(seq, position))
    else:

        def _seq_pop(s: list) -> list:
            return s[:-1]

        return builder.call_function(_seq_pop, args=(seq,))


@register("ConcatFromSequence")
def concat_from_sequence(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Concatenate tensors from sequence."""
    seq = builder.get_value(node.input[0])

    axis = get_attribute(node, "axis", 0)
    new_axis = get_attribute(node, "new_axis", 0)

    if new_axis:

        def _stack_seq(s: list, ax: int) -> torch.Tensor:
            return torch.stack(s, dim=ax)

        return builder.call_function(_stack_seq, args=(seq, axis))
    else:

        def _concat_seq(s: list, ax: int) -> torch.Tensor:
            return torch.cat(s, dim=ax)

        return builder.call_function(_concat_seq, args=(seq, axis))


@register("SplitToSequence")
def split_to_sequence(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Split tensor into sequence of tensors."""
    x = builder.get_value(node.input[0])
    split = builder.get_value(node.input[1]) if len(node.input) > 1 else None

    axis = get_attribute(node, "axis", 0)

    if split is not None:

        def _split_seq(t: torch.Tensor, s: torch.Tensor, ax: int) -> list:
            sizes = s.tolist() if hasattr(s, "tolist") else [s]
            return list(torch.split(t, sizes, dim=ax))

        return builder.call_function(_split_seq, args=(x, split, axis))
    else:

        def _split_ones(t: torch.Tensor, ax: int) -> list:
            return list(torch.split(t, 1, dim=ax))

        return builder.call_function(_split_ones, args=(x, axis))


# =============================================================================
# Gather/Scatter ND operators
# =============================================================================


@register("GatherND")
def gather_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gather elements from data using indices."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    batch_dims = get_attribute(node, "batch_dims", 0)

    def _gather_nd(d: torch.Tensor, idx: torch.Tensor, bd: int) -> torch.Tensor:
        idx = idx.long()
        idx_shape = idx.shape[:-1]
        last_dim = idx.shape[-1]
        data_shape = d.shape[last_dim:]

        result_shape = idx_shape + data_shape
        result = torch.zeros(result_shape, dtype=d.dtype, device=d.device)

        # Flatten for iteration
        flat_idx = idx.reshape(-1, last_dim)
        flat_result = result.reshape(-1, *data_shape) if data_shape else result.reshape(-1)

        for i in range(flat_idx.shape[0]):
            data_idx = tuple(flat_idx[i].tolist())
            flat_result[i] = d[data_idx]

        return result

    return builder.call_function(_gather_nd, args=(data, indices, batch_dims))


@register("ScatterND")
def scatter_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scatter updates into data at indices."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    updates = builder.get_value(node.input[2])

    reduction = get_attribute(node, "reduction", "none")

    def _scatter_nd(
        d: torch.Tensor, idx: torch.Tensor, upd: torch.Tensor
    ) -> torch.Tensor:
        output = d.clone()
        idx = idx.long()

        idx_shape = idx.shape[:-1]
        last_dim = idx.shape[-1]

        flat_idx = idx.reshape(-1, last_dim)
        flat_upd = upd.reshape(-1, *upd.shape[len(idx_shape) :])

        for i in range(flat_idx.shape[0]):
            data_idx = tuple(flat_idx[i].tolist())
            output[data_idx] = flat_upd[i]

        return output

    return builder.call_function(_scatter_nd, args=(data, indices, updates))


# =============================================================================
# Embedding and LayerNorm variants
# =============================================================================


@register("SkipLayerNormalization")
def skip_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Skip connection + LayerNorm (common in transformers)."""
    x = builder.get_value(node.input[0])
    skip = builder.get_value(node.input[1])
    gamma = builder.get_value(node.input[2])
    beta = builder.get_value(node.input[3]) if len(node.input) > 3 and node.input[3] else None
    bias = builder.get_value(node.input[4]) if len(node.input) > 4 and node.input[4] else None

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _skip_layer_norm(
        inp: torch.Tensor,
        sk: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor | None,
        bi: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        hidden = inp + sk
        if bi is not None:
            hidden = hidden + bi
        return torch.nn.functional.layer_norm(hidden, hidden.shape[-1:], weight=g, bias=b, eps=eps)

    return builder.call_function(
        _skip_layer_norm, args=(x, skip, gamma, beta, bias, epsilon)
    )


@register("EmbedLayerNormalization")
def embed_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Embedding + LayerNorm (common in BERT-like models)."""
    input_ids = builder.get_value(node.input[0])
    segment_ids = (
        builder.get_value(node.input[1])
        if len(node.input) > 1 and node.input[1]
        else None
    )
    word_embedding = builder.get_value(node.input[2])
    position_embedding = builder.get_value(node.input[3])
    segment_embedding = (
        builder.get_value(node.input[4])
        if len(node.input) > 4 and node.input[4]
        else None
    )
    gamma = builder.get_value(node.input[5]) if len(node.input) > 5 else None
    beta = builder.get_value(node.input[6]) if len(node.input) > 6 else None

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _embed_layer_norm(
        ids: torch.Tensor,
        seg_ids: torch.Tensor | None,
        word_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        seg_emb: torch.Tensor | None,
        g: torch.Tensor | None,
        b: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        # Word embedding lookup
        word_embed = torch.nn.functional.embedding(ids, word_emb)

        # Position embedding (assume sequential positions)
        seq_len = ids.shape[1]
        pos_embed = pos_emb[:seq_len].unsqueeze(0).expand(ids.shape[0], -1, -1)

        hidden = word_embed + pos_embed

        # Add segment embedding if present
        if seg_emb is not None and seg_ids is not None:
            seg_embed = torch.nn.functional.embedding(seg_ids, seg_emb)
            hidden = hidden + seg_embed

        # Layer normalization
        if g is not None:
            hidden = torch.nn.functional.layer_norm(
                hidden, hidden.shape[-1:], weight=g, bias=b, eps=eps
            )

        return hidden

    return builder.call_function(
        _embed_layer_norm,
        args=(
            input_ids,
            segment_ids,
            word_embedding,
            position_embedding,
            segment_embedding,
            gamma,
            beta,
            epsilon,
        ),
    )


# =============================================================================
# Attention operator (custom Microsoft domain)
# =============================================================================


@register("Attention")
def attention(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Attention operator.

    Supports two patterns:
    1. Decomposed SDPA (ViT, etc.): Inputs are Q, K, V tensors directly
       - 3 inputs: query, key, value
       - Attributes: is_causal, softcap, qk_matmul_output_mode

    2. Fused SDPA (com.microsoft domain): Inputs are input, weight, bias
       - 5+ inputs: input, weight, bias (optional), mask_index (optional), past (optional)
       - Attributes: num_heads, unidirectional
    """
    num_inputs = len(node.input)

    # Check if this is Decomposed SDPA pattern (Q, K, V directly)
    # Heuristic: If we have exactly 3 inputs and is_causal attribute exists
    is_causal_attr = get_attribute(node, "is_causal", None)
    softcap = get_attribute(node, "softcap", 0.0)

    if num_inputs == 3 and is_causal_attr is not None:
        # Decomposed SDPA: Q, K, V are passed directly
        query = builder.get_value(node.input[0])
        key = builder.get_value(node.input[1])
        value = builder.get_value(node.input[2])

        is_causal = is_causal_attr

        def _decomposed_sdpa(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            is_causal: int,
            scale: float,
        ) -> torch.Tensor:
            # Use scaled_dot_product_attention directly with Q, K, V
            if is_causal:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return output

        return builder.call_function(
            _decomposed_sdpa,
            args=(query, key, value, is_causal, softcap),
        )

    # Fused SDPA pattern (Microsoft domain style)
    input_node = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    mask_index = builder.get_value(node.input[3]) if len(node.input) > 3 and node.input[3] else None

    num_heads = get_attribute(node, "num_heads", 1)
    unidirectional = get_attribute(node, "unidirectional", 0)

    def _attention_sdpa(
        inp: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        mask: torch.Tensor | None,
        n_heads: int,
        is_causal: int,
    ) -> torch.Tensor:
        # QKV projection: input @ weight + bias
        qkv = torch.matmul(inp, w)
        if b is not None:
            qkv = qkv + b

        # Split QKV into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Use scaled_dot_product_attention
        # Note: is_causal cannot be used together with attn_mask
        if mask is not None:
            # Convert mask to attention mask format (additive mask)
            # SDPA expects: 0 = attend, -inf = don't attend
            attn_mask = torch.where(mask == 0, float("-inf"), 0.0)
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        elif is_causal:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        return output

    return builder.call_function(
        _attention_sdpa,
        args=(input_node, weight, bias, mask_index, num_heads, unidirectional),
    )
