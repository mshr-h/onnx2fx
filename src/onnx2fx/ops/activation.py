# SPDX-License-Identifier: Apache-2.0
"""Activation function operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


@register("Relu")
def relu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ReLU activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.relu, args=(x,))


@register("LeakyRelu")
def leaky_relu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Leaky ReLU activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 0.01)
    return builder.call_function(
        F.leaky_relu, args=(x,), kwargs={"negative_slope": alpha}
    )


@register("PRelu")
def prelu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Parametric ReLU activation."""
    x = builder.get_value(node.input[0])
    slope = builder.get_value(node.input[1])
    return builder.call_function(F.prelu, args=(x, slope))


@register("Elu")
def elu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Exponential Linear Unit activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.elu, args=(x,), kwargs={"alpha": alpha})


@register("Selu")
def selu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scaled Exponential Linear Unit activation."""
    x = builder.get_value(node.input[0])
    # SELU has fixed alpha and gamma values
    return builder.call_function(F.selu, args=(x,))


@register("Celu")
def celu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Continuously Differentiable Exponential Linear Unit activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.celu, args=(x,), kwargs={"alpha": alpha})


@register("Sigmoid")
def sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sigmoid activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sigmoid, args=(x,))


@register("HardSigmoid")
def hard_sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hard Sigmoid activation.

    Note: ONNX allows custom alpha/beta, but PyTorch's hardsigmoid uses fixed
    values (alpha=1/6, beta=0.5). The ONNX default (alpha=0.2, beta=0.5) differs
    slightly, but we use PyTorch's implementation for efficiency.
    """
    x = builder.get_value(node.input[0])
    return builder.call_function(F.hardsigmoid, args=(x,))


@register("Tanh")
def tanh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Tanh activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tanh, args=(x,))


@register("Softmax", since_version=1)
def softmax_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Softmax activation for opset 1-12.

    In opset < 13, the default axis is 1, and softmax is computed on a
    "coerced" 2D tensor (flatten dimensions before/after axis).
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 1)  # Default was 1 in opset < 13

    def _softmax_v1(t: torch.Tensor, axis: int) -> torch.Tensor:
        # Handle negative axis
        if axis < 0:
            axis = t.dim() + axis

        # Coerce to 2D: flatten [0:axis] and [axis:]
        orig_shape = t.shape
        pre_dim = 1
        for i in range(axis):
            pre_dim *= t.shape[i]

        t_2d = t.reshape(pre_dim, -1)
        result_2d = F.softmax(t_2d, dim=1)
        return result_2d.reshape(orig_shape)

    return builder.call_function(_softmax_v1, args=(x, axis))


@register("Softmax", since_version=13)
def softmax_v13(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Softmax activation for opset 13+.

    In opset 13+, the default axis is -1 (last dimension), and softmax is
    applied directly without coercion.
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)  # Default changed to -1 in opset 13
    return builder.call_function(F.softmax, args=(x,), kwargs={"dim": axis})


@register("LogSoftmax", since_version=1)
def log_softmax_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log Softmax activation for opset 1-12.

    In opset < 13, the default axis is 1, and log_softmax is computed on a
    "coerced" 2D tensor.
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 1)  # Default was 1 in opset < 13

    def _log_softmax_v1(t: torch.Tensor, axis: int) -> torch.Tensor:
        if axis < 0:
            axis = t.dim() + axis

        orig_shape = t.shape
        pre_dim = 1
        for i in range(axis):
            pre_dim *= t.shape[i]

        t_2d = t.reshape(pre_dim, -1)
        result_2d = F.log_softmax(t_2d, dim=1)
        return result_2d.reshape(orig_shape)

    return builder.call_function(_log_softmax_v1, args=(x, axis))


@register("LogSoftmax", since_version=13)
def log_softmax_v13(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log Softmax activation for opset 13+.

    In opset 13+, the default axis is -1.
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)  # Default changed to -1 in opset 13
    return builder.call_function(F.log_softmax, args=(x,), kwargs={"dim": axis})


@register("Softplus")
def softplus(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Softplus activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.softplus, args=(x,))


@register("Softsign")
def softsign(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Softsign activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.softsign, args=(x,))


@register("Gelu")
def gelu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gaussian Error Linear Unit activation."""
    x = builder.get_value(node.input[0])
    approximate = get_attribute(node, "approximate", "none")
    if approximate == "tanh":
        return builder.call_function(F.gelu, args=(x,), kwargs={"approximate": "tanh"})
    return builder.call_function(F.gelu, args=(x,))


@register("Silu")
def silu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sigmoid Linear Unit (SiLU/Swish) activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.silu, args=(x,))


@register("Mish")
def mish(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Mish activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.mish, args=(x,))


@register("ThresholdedRelu")
def thresholded_relu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Thresholded ReLU activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.threshold, args=(x, alpha, 0.0))


@register("HardSwish")
def hard_swish(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hard Swish activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(F.hardswish, args=(x,))


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


@register("Shrink")
def shrink(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Shrink activation.

    If x < -lambd: y = x + bias
    If x > lambd: y = x - bias
    Otherwise: y = 0
    """
    x = builder.get_value(node.input[0])
    bias = get_attribute(node, "bias", 0.0)
    lambd = get_attribute(node, "lambd", 0.5)

    def _shrink(t: torch.Tensor, bias: float, lambd: float) -> torch.Tensor:
        result = torch.zeros_like(t)
        mask_neg = t < -lambd
        mask_pos = t > lambd
        result = torch.where(mask_neg, t + bias, result)
        result = torch.where(mask_pos, t - bias, result)
        return result

    return builder.call_function(_shrink, args=(x, bias, lambd))
