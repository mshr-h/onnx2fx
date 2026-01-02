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
    """Hard Sigmoid activation."""
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", 0.2)
    beta = get_attribute(node, "beta", 0.5)
    # HardSigmoid(x) = max(0, min(1, alpha * x + beta))
    return builder.call_function(F.hardsigmoid, args=(x,))


@register("Tanh")
def tanh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Tanh activation."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tanh, args=(x,))


@register("Softmax")
def softmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Softmax activation."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)
    return builder.call_function(F.softmax, args=(x,), kwargs={"dim": axis})


@register("LogSoftmax")
def log_softmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log Softmax activation."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)
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
