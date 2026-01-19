# SPDX-License-Identifier: Apache-2.0
"""Activation function operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import unary_op

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


register("Relu")(unary_op(F.relu, "ReLU activation."))


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
    """Parametric ReLU activation.

    ONNX PRelu allows slope to be broadcastable to the input tensor,
    while PyTorch's F.prelu expects slope to match channel dimension.
    We implement using torch.where for proper broadcasting support.

    When slope has shape [C] and input has shape [N, C, ...], we need to
    reshape slope to [1, C, 1, ...] for proper broadcasting.
    """
    x = builder.get_value(node.input[0])
    slope = builder.get_value(node.input[1])

    def _prelu(x: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        # If slope is 1D with size matching channels and input is ND with N > 1,
        # reshape slope for proper broadcasting along channel dimension (dim=1)
        if slope.ndim == 1 and x.ndim > 1 and slope.numel() == x.shape[1]:
            # Reshape [C] to [1, C, 1, 1, ...] for broadcasting
            shape = [1, slope.numel()] + [1] * (x.ndim - 2)
            slope = slope.view(shape)
        return torch.where(x >= 0, x, x * slope)

    return builder.call_function(_prelu, args=(x, slope))


@register("Elu")
def elu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Exponential Linear Unit activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.elu, args=(x,), kwargs={"alpha": alpha})


@register("Selu")
def selu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scaled Exponential Linear Unit activation.

    ONNX SELU: y = gamma * (alpha * (exp(x) - 1) for x < 0, x for x >= 0)
    PyTorch SELU uses fixed alpha=1.6732... and gamma=1.0507...
    ONNX defaults: alpha=1.67326..., gamma=1.0507... but allows custom values.
    """
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.67326319217681884765625)
    gamma = get_attribute(node, "gamma", 1.05070102214813232421875)

    # PyTorch's fixed SELU values
    pytorch_alpha = 1.6732632423543772848170429916717
    pytorch_gamma = 1.0507009873554804934193349852946

    # If using PyTorch's fixed values (within tolerance), use F.selu for efficiency
    if abs(alpha - pytorch_alpha) < 1e-5 and abs(gamma - pytorch_gamma) < 1e-5:
        return builder.call_function(F.selu, args=(x,))

    # Otherwise implement manually: gamma * (alpha * (exp(x) - 1) for x < 0, x for x >= 0)
    def _custom_selu(x: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
        return gamma * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

    return builder.call_function(_custom_selu, args=(x, alpha, gamma))


@register("Celu")
def celu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Continuously Differentiable Exponential Linear Unit activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.celu, args=(x,), kwargs={"alpha": alpha})


register("Sigmoid")(unary_op(torch.sigmoid, "Sigmoid activation."))


@register("HardSigmoid")
def hard_sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hard Sigmoid activation.

    ONNX HardSigmoid: max(0, min(1, alpha * x + beta))
    ONNX defaults: alpha=0.2, beta=0.5
    PyTorch hardsigmoid uses fixed alpha=1/6, beta=0.5
    """
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 0.2)
    beta = get_attribute(node, "beta", 0.5)

    # PyTorch hardsigmoid uses alpha=1/6 ≈ 0.16667, beta=0.5
    pytorch_alpha = 1.0 / 6.0

    # If using PyTorch's fixed values (within tolerance), use F.hardsigmoid
    if abs(alpha - pytorch_alpha) < 1e-5 and abs(beta - 0.5) < 1e-5:
        return builder.call_function(F.hardsigmoid, args=(x,))

    # Otherwise implement manually: max(0, min(1, alpha * x + beta))
    def _custom_hardsigmoid(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return torch.clamp(alpha * x + beta, 0.0, 1.0)

    return builder.call_function(_custom_hardsigmoid, args=(x, alpha, beta))


register("Tanh")(unary_op(torch.tanh, "Tanh activation."))


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


register("Softplus")(unary_op(F.softplus, "Softplus activation."))
register("Softsign")(unary_op(F.softsign, "Softsign activation."))


@register("Gelu")
def gelu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gaussian Error Linear Unit activation."""
    x = builder.get_value(node.input[0])
    approximate = get_attribute(node, "approximate", "none")
    if approximate == "tanh":
        return builder.call_function(F.gelu, args=(x,), kwargs={"approximate": "tanh"})
    return builder.call_function(F.gelu, args=(x,))


register("Silu")(unary_op(F.silu, "Sigmoid Linear Unit (SiLU/Swish) activation."))
register("Swish")(unary_op(F.silu, "Swish activation (alias for SiLU)."))
register("Mish")(unary_op(F.mish, "Mish activation."))


@register("ThresholdedRelu")
def thresholded_relu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Thresholded ReLU activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 1.0)
    return builder.call_function(F.threshold, args=(x, alpha, 0.0))


register("HardSwish")(unary_op(F.hardswish, "Hard Swish activation."))


@register("Hardmax")
def hardmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hardmax - one-hot encoding of argmax."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)

    def _hardmax(t: torch.Tensor, ax: int) -> torch.Tensor:
        # Normalize axis to positive
        if ax < 0:
            ax = t.dim() + ax
        # one_hot appends the class dimension at the end
        one_hot = torch.nn.functional.one_hot(
            torch.argmax(t, dim=ax), num_classes=t.shape[ax]
        ).to(t.dtype)
        # Move the one-hot dimension from the end back to the original axis position
        # one_hot has shape: [...dims before ax..., ...dims after ax..., num_classes]
        # We need to move the last dim to position ax
        return torch.movedim(one_hot, -1, ax)

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
