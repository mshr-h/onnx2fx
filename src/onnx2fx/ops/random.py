# SPDX-License-Identifier: Apache-2.0
"""Random number generation operators.

This module implements ONNX operators for generating random tensors,
including normal and uniform distributions.

Note: Window functions (HannWindow, HammingWindow, BlackmanWindow) have been
moved to signal.py as they are used for signal processing.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Random number generation operators
# =============================================================================


@register("RandomNormal")
def random_normal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate random values from normal distribution.

    Note: The seed attribute is not supported; use torch.manual_seed() instead.
    """
    mean = get_attribute(node, "mean", 0.0)
    scale = get_attribute(node, "scale", 1.0)
    shape = get_attribute(node, "shape")

    def _random_normal(m: float, s: float, sh: list) -> torch.Tensor:
        return torch.randn(sh) * s + m

    return builder.call_function(_random_normal, args=(mean, scale, list(shape)))


@register("RandomNormalLike")
def random_normal_like(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate random values like input tensor."""
    x = builder.get_value(node.input[0])

    mean = get_attribute(node, "mean", 0.0)
    scale = get_attribute(node, "scale", 1.0)

    def _random_normal_like(t: torch.Tensor, m: float, s: float) -> torch.Tensor:
        return torch.randn_like(t) * s + m

    return builder.call_function(_random_normal_like, args=(x, mean, scale))


@register("RandomUniform")
def random_uniform(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate random values from uniform distribution."""
    low = get_attribute(node, "low", 0.0)
    high = get_attribute(node, "high", 1.0)
    shape = get_attribute(node, "shape")

    def _random_uniform(lo: float, hi: float, sh: list) -> torch.Tensor:
        return torch.rand(sh) * (hi - lo) + lo

    return builder.call_function(_random_uniform, args=(low, high, list(shape)))


@register("RandomUniformLike")
def random_uniform_like(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate random values like input tensor."""
    x = builder.get_value(node.input[0])

    low = get_attribute(node, "low", 0.0)
    high = get_attribute(node, "high", 1.0)

    def _random_uniform_like(t: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return torch.rand_like(t) * (hi - lo) + lo

    return builder.call_function(_random_uniform_like, args=(x, low, high))


@register("Multinomial")
def multinomial(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sample from multinomial distribution."""
    x = builder.get_value(node.input[0])

    sample_size = get_attribute(node, "sample_size", 1)

    return builder.call_function(
        torch.multinomial, args=(x, sample_size), kwargs={"replacement": True}
    )


@register("Bernoulli")
def bernoulli(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sample from Bernoulli distribution."""
    x = builder.get_value(node.input[0])

    return builder.call_function(torch.bernoulli, args=(x,))
