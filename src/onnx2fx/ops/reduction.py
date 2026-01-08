# SPDX-License-Identifier: Apache-2.0
"""Reduction operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


def _get_reduction_axes(node: onnx.NodeProto, builder: "GraphBuilder"):
    """Get axes for reduction, handling both attribute and input formats.

    In opset < 13, axes is an attribute.
    In opset 13-17, axes can be an attribute or an optional input.
    In opset 18+, axes is an optional input only.
    """
    # First, always check if axes is provided as an attribute
    # This works for opset < 18 where axes can be an attribute
    axes_attr = get_attribute(node, "axes")
    if axes_attr is not None:
        return axes_attr

    # If no attribute, check if axes is an input (opset 13+)
    if builder.opset_version >= 13:
        if len(node.input) > 1 and node.input[1]:
            axes_name = node.input[1]
            # Check if axes is an initializer (constant)
            if axes_name in builder.initializer_map:
                axes_tensor = builder.initializer_map[axes_name]
                axes = axes_tensor.tolist()
                if isinstance(axes, int):
                    axes = [axes]
                return axes
            # Otherwise get the FX node
            axes = builder.get_value(axes_name)
            if isinstance(axes, torch.Tensor):
                axes = axes.tolist()
            return axes

    # No axes specified means reduce over all dimensions
    return None


@register("ReduceSum")
def reduce_sum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sum reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_sum(t, axes, keepdims):
        if axes is None:
            return torch.sum(t)
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.sum(t, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_sum, args=(x, axes, bool(keepdims)))


@register("ReduceMean")
def reduce_mean(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Mean reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_mean(t, axes, keepdims):
        if axes is None:
            return torch.mean(t)
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.mean(t, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_mean, args=(x, axes, bool(keepdims)))


@register("ReduceMax")
def reduce_max(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Max reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_max(t, axes, keepdims):
        if axes is None:
            return t.max()
        if isinstance(axes, list) and len(axes) == 1:
            axes = axes[0]
        if isinstance(axes, int):
            return t.max(dim=axes, keepdim=keepdims).values
        # Multiple axes: reduce sequentially
        result = t
        for axis in sorted(axes, reverse=True):
            result = result.max(dim=axis, keepdim=keepdims).values
        return result

    return builder.call_function(_reduce_max, args=(x, axes, bool(keepdims)))


@register("ReduceMin")
def reduce_min(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Min reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_min(t, axes, keepdims):
        if axes is None:
            return t.min()
        if isinstance(axes, list) and len(axes) == 1:
            axes = axes[0]
        if isinstance(axes, int):
            return t.min(dim=axes, keepdim=keepdims).values
        # Multiple axes: reduce sequentially
        result = t
        for axis in sorted(axes, reverse=True):
            result = result.min(dim=axis, keepdim=keepdims).values
        return result

    return builder.call_function(_reduce_min, args=(x, axes, bool(keepdims)))


@register("ReduceProd")
def reduce_prod(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Product reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    if axes is None:
        return builder.call_function(torch.prod, args=(x,))

    def _reduce_prod(t, axes, keepdims):
        if isinstance(axes, list) and len(axes) == 1:
            axes = axes[0]
        if isinstance(axes, int):
            return torch.prod(t, dim=axes, keepdim=keepdims)
        # Multiple axes: reduce sequentially
        result = t
        for axis in sorted(axes, reverse=True):
            result = torch.prod(result, dim=axis, keepdim=keepdims)
        return result

    return builder.call_function(_reduce_prod, args=(x, axes, bool(keepdims)))


@register("ReduceL1")
def reduce_l1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """L1 norm reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_l1(t, axes, keepdims):
        abs_t = torch.abs(t)
        if axes is None:
            return torch.sum(abs_t)
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.sum(abs_t, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_l1, args=(x, axes, bool(keepdims)))


@register("ReduceL2")
def reduce_l2(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """L2 norm reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_l2(t, axes, keepdims):
        if axes is None:
            return torch.norm(t)
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.norm(t, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_l2, args=(x, axes, bool(keepdims)))


@register("ReduceLogSum")
def reduce_log_sum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log of sum reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_log_sum(t, axes, keepdims):
        if axes is None:
            return torch.log(torch.sum(t))
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.log(torch.sum(t, dim=axes, keepdim=keepdims))

    return builder.call_function(_reduce_log_sum, args=(x, axes, bool(keepdims)))


@register("ReduceLogSumExp")
def reduce_log_sum_exp(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """LogSumExp reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_log_sum_exp(t, axes, keepdims):
        if axes is None:
            return torch.logsumexp(t, dim=tuple(range(t.dim())))
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.logsumexp(t, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_log_sum_exp, args=(x, axes, bool(keepdims)))


@register("ReduceSumSquare")
def reduce_sum_square(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sum of squares reduction."""
    x = builder.get_value(node.input[0])
    axes = _get_reduction_axes(node, builder)
    keepdims = get_attribute(node, "keepdims", 1)

    def _reduce_sum_square(t, axes, keepdims):
        sq = torch.square(t)
        if axes is None:
            return torch.sum(sq)
        if isinstance(axes, torch.Tensor):
            axes = tuple(axes.tolist())
        elif isinstance(axes, (list, tuple)):
            axes = tuple(axes)
        return torch.sum(sq, dim=axes, keepdim=keepdims)

    return builder.call_function(_reduce_sum_square, args=(x, axes, bool(keepdims)))


# =============================================================================
# ArgMax/ArgMin operators
# =============================================================================


@register("ArgMax")
def argmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Index of maximum value."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 0)
    keepdims = get_attribute(node, "keepdims", 1)
    select_last_index = get_attribute(node, "select_last_index", 0)

    def _argmax(t, axis, keepdims, select_last_index):
        if select_last_index:
            # Flip, argmax, then adjust index
            flipped = torch.flip(t, [axis])
            idx = torch.argmax(flipped, dim=axis, keepdim=keepdims)
            return t.size(axis) - 1 - idx
        return torch.argmax(t, dim=axis, keepdim=keepdims)

    return builder.call_function(
        _argmax, args=(x, axis, bool(keepdims), bool(select_last_index))
    )


@register("ArgMin")
def argmin(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Index of minimum value."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 0)
    keepdims = get_attribute(node, "keepdims", 1)
    select_last_index = get_attribute(node, "select_last_index", 0)

    def _argmin(t, axis, keepdims, select_last_index):
        if select_last_index:
            flipped = torch.flip(t, [axis])
            idx = torch.argmin(flipped, dim=axis, keepdim=keepdims)
            return t.size(axis) - 1 - idx
        return torch.argmin(t, dim=axis, keepdim=keepdims)

    return builder.call_function(
        _argmin, args=(x, axis, bool(keepdims), bool(select_last_index))
    )
