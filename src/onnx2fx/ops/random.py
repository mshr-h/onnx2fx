# SPDX-License-Identifier: Apache-2.0
"""Random number generation operators.

This module implements ONNX operators for generating random tensors,
including normal and uniform distributions.
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


# =============================================================================
# Window function operators
# =============================================================================


@register("HannWindow", since_version=17)
def hann_window(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate a Hann window.

    Attributes:
        periodic: If 1, returns periodic window. If 0, returns symmetric window.
        output_datatype: ONNX TensorProto data type for output (default: FLOAT).
    """
    from ..utils.dtype import onnx_dtype_to_torch

    size = builder.get_value(node.input[0])
    periodic = get_attribute(node, "periodic", 1)
    output_datatype = get_attribute(node, "output_datatype", 1)  # Default: FLOAT

    dtype = onnx_dtype_to_torch(output_datatype)

    def _hann_window(
        window_length: torch.Tensor, periodic: bool, dtype: torch.dtype
    ) -> torch.Tensor:
        length = int(window_length.item())
        return torch.hann_window(length, periodic=periodic, dtype=dtype)

    return builder.call_function(_hann_window, args=(size, bool(periodic), dtype))


@register("BlackmanWindow", since_version=17)
def blackman_window(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate a Blackman window.

    Attributes:
        periodic: If 1, returns periodic window. If 0, returns symmetric window.
        output_datatype: ONNX TensorProto data type for output (default: FLOAT).
    """
    from ..utils.dtype import onnx_dtype_to_torch

    size = builder.get_value(node.input[0])
    periodic = get_attribute(node, "periodic", 1)
    output_datatype = get_attribute(node, "output_datatype", 1)  # Default: FLOAT

    dtype = onnx_dtype_to_torch(output_datatype)

    def _blackman_window(
        window_length: torch.Tensor, periodic: bool, dtype: torch.dtype
    ) -> torch.Tensor:
        length = int(window_length.item())
        return torch.blackman_window(length, periodic=periodic, dtype=dtype)

    return builder.call_function(_blackman_window, args=(size, bool(periodic), dtype))
