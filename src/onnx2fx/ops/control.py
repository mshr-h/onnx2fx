# SPDX-License-Identifier: Apache-2.0
"""Control flow and optional operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Control Flow operators
# =============================================================================


@register("If")
def if_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Conditional execution based on condition tensor.

    This is a simplified implementation that evaluates both branches
    and selects the result based on the condition.
    """
    condition = builder.get_value(node.input[0])

    # Get the then and else branches from attributes
    then_branch = None
    else_branch = None
    for attr in node.attribute:
        if attr.name == "then_branch":
            then_branch = attr.g
        elif attr.name == "else_branch":
            else_branch = attr.g

    if then_branch is None or else_branch is None:
        raise ValueError("If operator requires both then_branch and else_branch")

    # For FX graph, we create a function that evaluates condition at runtime
    def _if_then_else(
        cond: torch.Tensor,
        then_fn,
        else_fn,
        inputs: dict,
    ) -> torch.Tensor:
        if cond.item():
            return then_fn(inputs)
        else:
            return else_fn(inputs)

    # Since FX doesn't support dynamic control flow well,
    # we'll use torch.where for simple cases or cond for complex ones
    def _simple_if(cond: torch.Tensor) -> bool:
        return bool(cond.item()) if cond.numel() == 1 else True

    return builder.call_function(_simple_if, args=(condition,))


@register("Loop")
def loop_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Loop operator.

    This is a placeholder implementation. Full Loop support requires
    dynamic graph construction which is complex in FX.
    """
    max_trip_count = builder.get_value(node.input[0]) if node.input[0] else None
    keep_going = builder.get_value(node.input[1]) if len(node.input) > 1 and node.input[1] else None

    # Get loop body
    body = None
    for attr in node.attribute:
        if attr.name == "body":
            body = attr.g

    # Collect initial loop-carried values
    initial_values = []
    for i in range(2, len(node.input)):
        if node.input[i]:
            initial_values.append(builder.get_value(node.input[i]))

    # Simplified: return initial values (no iteration)
    # Full implementation would require torch.jit.script or similar
    if len(initial_values) == 1:
        return initial_values[0]
    else:
        return builder.call_function(lambda *args: args, args=tuple(initial_values))


@register("Scan")
def scan_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scan operator - applies a function repeatedly over sequences.

    Simplified implementation that processes sequences.
    """
    num_scan_inputs = get_attribute(node, "num_scan_inputs", 1)

    # Get initial state and scan inputs
    initial_states = []
    scan_inputs = []

    state_count = len(node.input) - num_scan_inputs
    for i in range(state_count):
        if node.input[i]:
            initial_states.append(builder.get_value(node.input[i]))

    for i in range(state_count, len(node.input)):
        if node.input[i]:
            scan_inputs.append(builder.get_value(node.input[i]))

    # Simplified: return concatenated scan inputs
    if len(scan_inputs) == 1:
        return scan_inputs[0]
    else:
        return builder.call_function(torch.cat, args=(scan_inputs,), kwargs={"dim": 0})


# =============================================================================
# Optional operators
# =============================================================================


@register("Optional")
def optional_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an optional value."""
    if len(node.input) > 0 and node.input[0]:
        # Has a value
        return builder.get_value(node.input[0])
    else:
        # Empty optional - return None wrapped
        return builder.call_function(lambda: None, args=())


@register("OptionalHasElement")
def optional_has_element(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Check if optional has a value."""
    opt = builder.get_value(node.input[0])

    def _has_element(x) -> torch.Tensor:
        return torch.tensor(x is not None, dtype=torch.bool)

    return builder.call_function(_has_element, args=(opt,))


@register("OptionalGetElement")
def optional_get_element(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get the value from an optional."""
    opt = builder.get_value(node.input[0])

    def _get_element(x):
        if x is None:
            raise RuntimeError("Optional has no element")
        return x

    return builder.call_function(_get_element, args=(opt,))


# =============================================================================
# Where operator (already in arithmetic, but useful for control flow)
# =============================================================================


@register("Select")
def select_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Select elements based on indices (like advanced indexing)."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    return builder.call_function(torch.index_select, args=(data, 0, indices))


# =============================================================================
# Compress operator
# =============================================================================


@register("Compress")
def compress_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Select elements based on a boolean condition tensor."""
    data = builder.get_value(node.input[0])
    condition = builder.get_value(node.input[1])

    axis = get_attribute(node, "axis", None)

    if axis is not None:

        def _compress_axis(d: torch.Tensor, c: torch.Tensor, ax: int) -> torch.Tensor:
            # Get indices where condition is True
            indices = torch.nonzero(c, as_tuple=True)[0]
            return torch.index_select(d, ax, indices)

        return builder.call_function(_compress_axis, args=(data, condition, axis))
    else:
        # Flatten and compress
        def _compress_flat(d: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return d.flatten()[c.flatten().bool()]

        return builder.call_function(_compress_flat, args=(data, condition))


# =============================================================================
# Bitwise shifts
# =============================================================================


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


# =============================================================================
# String operators (limited support)
# =============================================================================


@register("StringNormalizer")
def string_normalizer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """String normalization - passthrough for tensors."""
    x = builder.get_value(node.input[0])
    # String operations are not well supported in PyTorch tensors
    # Return as-is
    return x


# =============================================================================
# Constant of Shape
# =============================================================================


@register("ConstantOfShape")
def constant_of_shape(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create a constant tensor of a given shape."""
    shape = builder.get_value(node.input[0])

    # Get the value to fill with
    value = get_attribute(node, "value")
    if value is not None:
        # value is already a torch.Tensor from attribute parsing
        fill_value = float(value.item())
    else:
        fill_value = 0.0

    def _constant_of_shape(s: torch.Tensor, val: float) -> torch.Tensor:
        shape_list = s.tolist()
        return torch.full(shape_list, val, dtype=torch.float32)

    return builder.call_function(_constant_of_shape, args=(shape, fill_value))


# =============================================================================
# Eye-like and Random operators
# =============================================================================


@register("EyeLike")
def eye_like(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an identity matrix with the same shape as input."""
    x = builder.get_value(node.input[0])

    k = get_attribute(node, "k", 0)
    dtype = get_attribute(node, "dtype", None)

    def _eye_like(t: torch.Tensor, diag: int) -> torch.Tensor:
        n, m = t.shape[-2], t.shape[-1]
        eye = torch.eye(n, m, dtype=t.dtype, device=t.device)
        if diag != 0:
            eye = torch.diagonal(eye, offset=diag)
        return eye

    return builder.call_function(_eye_like, args=(x, k))


@register("RandomNormal")
def random_normal(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate random values from normal distribution."""
    mean = get_attribute(node, "mean", 0.0)
    scale = get_attribute(node, "scale", 1.0)
    shape = get_attribute(node, "shape")
    seed = get_attribute(node, "seed", None)

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


# =============================================================================
# Bernoulli
# =============================================================================


@register("Bernoulli")
def bernoulli(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sample from Bernoulli distribution."""
    x = builder.get_value(node.input[0])

    return builder.call_function(torch.bernoulli, args=(x,))
