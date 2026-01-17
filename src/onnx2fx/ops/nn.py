# SPDX-License-Identifier: Apache-2.0
"""Neural network layer operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

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
# Convolution operators
# =============================================================================


def _get_conv_params(node: onnx.NodeProto) -> dict:
    """Extract common convolution parameters from node attributes."""
    return {
        "dilations": get_attribute(node, "dilations"),
        "group": get_attribute(node, "group", 1),
        "kernel_shape": get_attribute(node, "kernel_shape"),
        "pads": get_attribute(node, "pads"),
        "strides": get_attribute(node, "strides"),
        "auto_pad": get_attribute(node, "auto_pad", "NOTSET"),
    }


@register("Conv")
def conv(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """N-dimensional convolution."""
    x = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    params = _get_conv_params(node)
    strides = params["strides"] or [1]
    dilations = params["dilations"] or [1]
    group = params["group"]
    pads = params["pads"]
    auto_pad = params["auto_pad"]
    kernel_shape = params["kernel_shape"]

    def _conv(x, weight, bias, strides, dilations, group, pads, auto_pad, kernel_shape):
        ndim = len(weight.shape) - 2  # Exclude batch and channel dims

        # Handle padding
        padding = 0
        if pads is not None:
            n = len(pads) // 2
            symmetric = all(pads[i] == pads[i + n] for i in range(n))
            if symmetric:
                padding = tuple(pads[:n])
            else:
                # Asymmetric padding
                # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                # F.pad: [xn_begin, xn_end, ..., x1_begin, x1_end] (reverse order)
                pad_list = []
                for i in range(n - 1, -1, -1):
                    pad_list.extend([pads[i], pads[i + n]])
                x = F.pad(x, pad_list)
                padding = 0

        # Handle auto_pad
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Compute padding for SAME
            input_shape = x.shape[2:]
            output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]
            pad_total = [
                max(0, (o - 1) * st + (k - 1) * d + 1 - i)
                for i, o, k, st, d in zip(
                    input_shape,
                    output_shape,
                    kernel_shape or weight.shape[2:],
                    strides,
                    dilations,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p // 2, p - p // 2])
            else:
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p - p // 2, p // 2])
            x = F.pad(x, pad_list)
            padding = 0

        strides_tuple = tuple(strides) if len(strides) > 1 else strides[0]
        dilations_tuple = tuple(dilations) if len(dilations) > 1 else dilations[0]

        if ndim == 1:
            return F.conv1d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        elif ndim == 2:
            return F.conv2d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        elif ndim == 3:
            return F.conv3d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        else:
            raise NotImplementedError(f"Conv{ndim}D not supported")

    return builder.call_function(
        _conv,
        args=(x, weight, bias, strides, dilations, group, pads, auto_pad, kernel_shape),
    )


@register("ConvTranspose")
def conv_transpose(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """N-dimensional transposed convolution."""
    x = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    strides = get_attribute(node, "strides") or [1]
    dilations = get_attribute(node, "dilations") or [1]
    group = get_attribute(node, "group", 1)
    pads = get_attribute(node, "pads")
    output_padding = get_attribute(node, "output_padding") or [0]

    def _conv_transpose(
        x, weight, bias, strides, dilations, group, pads, output_padding
    ):
        ndim = len(weight.shape) - 2

        padding = 0
        if pads is not None:
            n = len(pads) // 2
            padding = tuple(pads[:n])

        strides_tuple = tuple(strides) if len(strides) > 1 else strides[0]
        dilations_tuple = tuple(dilations) if len(dilations) > 1 else dilations[0]
        output_padding_tuple = (
            tuple(output_padding) if len(output_padding) > 1 else output_padding[0]
        )

        if ndim == 1:
            return F.conv_transpose1d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        elif ndim == 2:
            return F.conv_transpose2d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        elif ndim == 3:
            return F.conv_transpose3d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        else:
            raise NotImplementedError(f"ConvTranspose{ndim}D not supported")

    return builder.call_function(
        _conv_transpose,
        args=(x, weight, bias, strides, dilations, group, pads, output_padding),
    )


# =============================================================================
# Pooling operators
# =============================================================================


@register("MaxPool")
def max_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Max pooling."""
    x = builder.get_value(node.input[0])

    kernel_shape = get_attribute(node, "kernel_shape")
    strides = get_attribute(node, "strides") or kernel_shape
    pads = get_attribute(node, "pads")
    dilations = get_attribute(node, "dilations") or [1] * len(kernel_shape)
    ceil_mode = get_attribute(node, "ceil_mode", 0)
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")

    def _max_pool(x, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad):
        ndim = len(kernel_shape)

        padding = 0
        # Handle auto_pad first (before explicit pads)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Compute padding for SAME
            input_shape = x.shape[2:]
            output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]
            pad_total = [
                max(0, (o - 1) * st + (k - 1) * d + 1 - i)
                for i, o, k, st, d in zip(
                    input_shape,
                    output_shape,
                    kernel_shape,
                    strides,
                    dilations,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p // 2, p - p // 2])
            else:
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p - p // 2, p // 2])
            x = F.pad(x, pad_list, value=float("-inf"))
            padding = 0
        elif pads is not None:
            n = len(pads) // 2
            symmetric = all(pads[i] == pads[i + n] for i in range(n))
            if symmetric:
                padding = tuple(pads[:n])
            else:
                pad_list = []
                for i in range(n - 1, -1, -1):
                    pad_list.extend([pads[i], pads[i + n]])
                x = F.pad(x, pad_list, value=float("-inf"))
                padding = 0

        kernel = tuple(kernel_shape)
        stride = tuple(strides)
        dilation = tuple(dilations)

        if ndim == 1:
            return F.max_pool1d(
                x,
                kernel[0],
                stride=stride[0],
                padding=padding,
                dilation=dilation[0],
                ceil_mode=bool(ceil_mode),
            )
        elif ndim == 2:
            return F.max_pool2d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=bool(ceil_mode),
            )
        elif ndim == 3:
            return F.max_pool3d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=bool(ceil_mode),
            )
        else:
            raise NotImplementedError(f"MaxPool{ndim}D not supported")

    return builder.call_function(
        _max_pool, args=(x, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad)
    )


@register("AveragePool")
def average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Average pooling."""
    x = builder.get_value(node.input[0])

    kernel_shape = get_attribute(node, "kernel_shape")
    strides = get_attribute(node, "strides") or kernel_shape
    pads = get_attribute(node, "pads")
    ceil_mode = get_attribute(node, "ceil_mode", 0)
    count_include_pad = get_attribute(node, "count_include_pad", 0)

    def _avg_pool(x, kernel_shape, strides, pads, ceil_mode, count_include_pad):
        ndim = len(kernel_shape)

        padding = 0
        if pads is not None:
            n = len(pads) // 2
            padding = tuple(pads[:n])

        kernel = tuple(kernel_shape)
        stride = tuple(strides)

        if ndim == 1:
            return F.avg_pool1d(
                x,
                kernel[0],
                stride=stride[0],
                padding=padding,
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        elif ndim == 2:
            return F.avg_pool2d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        elif ndim == 3:
            return F.avg_pool3d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        else:
            raise NotImplementedError(f"AveragePool{ndim}D not supported")

    return builder.call_function(
        _avg_pool, args=(x, kernel_shape, strides, pads, ceil_mode, count_include_pad)
    )


@register("GlobalAveragePool")
def global_average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Global average pooling."""
    x = builder.get_value(node.input[0])

    def _global_avg_pool(x):
        # Average over all spatial dimensions (keep batch and channel)
        dims = tuple(range(2, x.dim()))
        return x.mean(dim=dims, keepdim=True)

    return builder.call_function(_global_avg_pool, args=(x,))


@register("GlobalMaxPool")
def global_max_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Global max pooling."""
    x = builder.get_value(node.input[0])

    def _global_max_pool(x):
        # Max over all spatial dimensions (keep batch and channel)
        result = x
        for dim in range(x.dim() - 1, 1, -1):
            result = result.max(dim=dim, keepdim=True).values
        return result

    return builder.call_function(_global_max_pool, args=(x,))


# =============================================================================
# Normalization operators
# =============================================================================


@register("BatchNormalization")
def batch_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Batch normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])
    mean = builder.get_value(node.input[3])
    var = builder.get_value(node.input[4])

    epsilon = get_attribute(node, "epsilon", 1e-5)
    # Note: ONNX momentum attribute is not used in inference mode
    training_mode = get_attribute(node, "training_mode", 0)

    def _batch_norm(x, scale, bias, mean, var, epsilon, training_mode):
        return F.batch_norm(
            x,
            mean,
            var,
            weight=scale,
            bias=bias,
            training=bool(training_mode),
            eps=epsilon,
        )

    return builder.call_function(
        _batch_norm, args=(x, scale, bias, mean, var, epsilon, training_mode)
    )


@register("LayerNormalization")
def layer_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Layer normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])

    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    axis = get_attribute(node, "axis", -1)
    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _layer_norm(x, scale, bias, axis, epsilon):
        # Compute normalized shape from axis
        if axis < 0:
            axis = x.dim() + axis
        normalized_shape = x.shape[axis:]
        return F.layer_norm(x, normalized_shape, weight=scale, bias=bias, eps=epsilon)

    return builder.call_function(_layer_norm, args=(x, scale, bias, axis, epsilon))


@register("InstanceNormalization")
def instance_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Instance normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _instance_norm(x, scale, bias, epsilon):
        return F.instance_norm(x, weight=scale, bias=bias, eps=epsilon)

    return builder.call_function(_instance_norm, args=(x, scale, bias, epsilon))


@register("GroupNormalization")
def group_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Group normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])

    epsilon = get_attribute(node, "epsilon", 1e-5)
    num_groups = get_attribute(node, "num_groups")

    def _group_norm(x, scale, bias, num_groups, epsilon):
        return F.group_norm(x, num_groups, weight=scale, bias=bias, eps=epsilon)

    return builder.call_function(
        _group_norm, args=(x, scale, bias, num_groups, epsilon)
    )


@register("LRN")
def lrn(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Local Response Normalization."""
    x = builder.get_value(node.input[0])

    alpha = get_attribute(node, "alpha", 0.0001)
    beta = get_attribute(node, "beta", 0.75)
    bias = get_attribute(node, "bias", 1.0)
    size = get_attribute(node, "size")

    return builder.call_function(
        F.local_response_norm,
        args=(x, size),
        kwargs={"alpha": alpha, "beta": beta, "k": bias},
    )


# =============================================================================
# Dropout and regularization
# =============================================================================


@register("Dropout")
def dropout(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Dropout (inference mode - identity)."""
    x = builder.get_value(node.input[0])

    # In inference mode, dropout is identity
    # ratio = get_attribute(node, "ratio", 0.5)
    # training_mode from input or default to False

    # For inference, just return input
    # Note: ONNX Dropout can have 2 outputs (output, mask), we handle first
    return builder.call_function(lambda t: t, args=(x,))
