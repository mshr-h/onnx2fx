# SPDX-License-Identifier: Apache-2.0
"""Tensor manipulation operators."""

from typing import TYPE_CHECKING, List

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Constant and Identity operators
# =============================================================================


@register("Constant")
def constant(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create a constant tensor."""
    value = get_attribute(node, "value")
    if value is None:
        value_float = get_attribute(node, "value_float")
        if value_float is not None:
            value = torch.tensor(value_float, dtype=torch.float32)
        value_int = get_attribute(node, "value_int")
        if value_int is not None:
            value = torch.tensor(value_int, dtype=torch.int64)
        value_floats = get_attribute(node, "value_floats")
        if value_floats is not None:
            value = torch.tensor(value_floats, dtype=torch.float32)
        value_ints = get_attribute(node, "value_ints")
        if value_ints is not None:
            value = torch.tensor(value_ints, dtype=torch.int64)

    if value is None:
        raise ValueError(f"Constant node {node.name} has no value attribute")

    output_name = node.output[0]
    safe_name = output_name.replace(".", "_").replace("/", "_")
    builder._constants[safe_name] = value

    fx_node = builder.graph.get_attr(safe_name)
    fx_node.meta["onnx_op_type"] = "Constant"
    fx_node.meta["onnx_name"] = output_name
    fx_node.meta["onnx_shape"] = list(value.shape) if hasattr(value, "shape") else []
    fx_node.meta["onnx_dtype"] = value.dtype if hasattr(value, "dtype") else None
    return fx_node


@register("Identity")
def identity(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Identity operator - returns input unchanged."""
    return builder.get_value(node.input[0])


@register("Cast")
def cast(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Cast tensor to a different data type."""
    from ..utils.dtype import onnx_dtype_to_torch

    x = builder.get_value(node.input[0])
    to_dtype = get_attribute(node, "to")
    torch_dtype = onnx_dtype_to_torch(to_dtype)

    if torch_dtype is None:
        raise ValueError(f"Unsupported cast target dtype: {to_dtype}")

    return builder.call_function(lambda t, dtype: t.to(dtype), args=(x, torch_dtype))


@register("CastLike")
def cast_like(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Cast tensor to the same data type as the target tensor."""
    x = builder.get_value(node.input[0])
    target = builder.get_value(node.input[1])

    def _cast_like(t, target):
        return t.to(target.dtype)

    return builder.call_function(_cast_like, args=(x, target))


# =============================================================================
# Shape manipulation operators
# =============================================================================


@register("Reshape")
def reshape(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Reshape tensor to a new shape."""
    x = builder.get_value(node.input[0])
    shape = builder.get_value(node.input[1])

    def _reshape(t, shape):
        if isinstance(shape, torch.Tensor):
            shape = tuple(shape.tolist())
        return torch.reshape(t, shape)

    return builder.call_function(_reshape, args=(x, shape))


@register("Transpose")
def transpose(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Transpose tensor dimensions."""
    x = builder.get_value(node.input[0])
    perm = get_attribute(node, "perm")
    if perm is None:
        # Default: reverse all dimensions
        return builder.call_function(lambda t: t.T, args=(x,))
    return builder.call_function(torch.permute, args=(x, perm))


@register("Squeeze")
def squeeze(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Remove dimensions of size 1."""
    x = builder.get_value(node.input[0])

    # axes can be input (opset 13+) or attribute
    if len(node.input) > 1 and node.input[1]:
        axes = builder.get_value(node.input[1])
        # If axes is a tensor node, use dynamic squeeze
        def _squeeze_dynamic(t, axes):
            if isinstance(axes, torch.Tensor):
                axes = axes.tolist()
            if isinstance(axes, list):
                if len(axes) == 1:
                    return torch.squeeze(t, dim=axes[0])
                # Multiple axes - squeeze in reverse order
                result = t
                for axis in sorted(axes, reverse=True):
                    result = torch.squeeze(result, dim=int(axis))
                return result
            return torch.squeeze(t, dim=int(axes))

        return builder.call_function(_squeeze_dynamic, args=(x, axes))

    axes = get_attribute(node, "axes")
    if axes is not None:
        # Squeeze specific dimensions
        result = x
        # Sort in reverse to maintain correct indices after each squeeze
        for axis in sorted(axes, reverse=True):
            result = builder.call_function(torch.squeeze, args=(result,), kwargs={"dim": axis})
        return result
    return builder.call_function(torch.squeeze, args=(x,))


@register("Unsqueeze")
def unsqueeze(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Insert dimensions of size 1."""
    x = builder.get_value(node.input[0])

    # axes is input (opset 13+) or attribute
    if len(node.input) > 1 and node.input[1]:
        axes = builder.get_value(node.input[1])
        # If axes is a tensor node, use dynamic unsqueeze
        def _unsqueeze_dynamic(t, axes):
            if isinstance(axes, torch.Tensor):
                axes = axes.tolist()
            if isinstance(axes, int):
                return torch.unsqueeze(t, axes)
            # Handle multiple axes - unsqueeze in sorted order
            result = t
            for axis in sorted(axes):
                result = torch.unsqueeze(result, int(axis))
            return result

        return builder.call_function(_unsqueeze_dynamic, args=(x, axes))

    axes = get_attribute(node, "axes")
    if axes is None:
        raise ValueError("Unsqueeze requires axes")

    # Handle single axis
    if isinstance(axes, int):
        return builder.call_function(torch.unsqueeze, args=(x, axes))

    # Handle multiple axes - unsqueeze in sorted order
    result = x
    for axis in sorted(axes):
        result = builder.call_function(torch.unsqueeze, args=(result, axis))
    return result


@register("Flatten")
def flatten(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Flatten tensor to 2D."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 1)
    return builder.call_function(torch.flatten, args=(x,), kwargs={"start_dim": axis})


@register("Expand")
def expand(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Broadcast tensor to a new shape."""
    x = builder.get_value(node.input[0])
    shape = builder.get_value(node.input[1])
    return builder.call_function(torch.Tensor.expand, args=(x, shape))


# =============================================================================
# Concatenation and splitting operators
# =============================================================================


@register("Concat")
def concat(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Concatenate tensors along an axis."""
    inputs = [builder.get_value(name) for name in node.input]
    axis = get_attribute(node, "axis", 0)
    return builder.call_function(torch.cat, args=(inputs,), kwargs={"dim": axis})


@register("Split")
def split(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Split tensor into chunks."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 0)
    num_outputs = get_attribute(node, "num_outputs")

    # split sizes can be input (opset 13+) or attribute
    if len(node.input) > 1 and node.input[1]:
        split_sizes = builder.get_value(node.input[1])
        result = builder.call_function(torch.split, args=(x, split_sizes, axis))
    elif num_outputs is not None:
        # Split into equal parts
        result = builder.call_function(torch.chunk, args=(x, num_outputs, axis))
    else:
        split_attr = get_attribute(node, "split")
        if split_attr is not None:
            result = builder.call_function(torch.split, args=(x, split_attr, axis))
        else:
            # Default: split into single-element chunks
            result = builder.call_function(torch.chunk, args=(x, len(node.output), axis))

    # Handle multiple outputs
    for i, output_name in enumerate(node.output):
        if output_name:
            idx_node = builder.call_function(lambda t, idx: t[idx], args=(result, i))
            builder.env[output_name] = idx_node

    return result


# =============================================================================
# Slicing and indexing operators
# =============================================================================


@register("Slice")
def slice_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Slice tensor along axes."""
    x = builder.get_value(node.input[0])
    starts = builder.get_value(node.input[1])
    ends = builder.get_value(node.input[2])

    axes = None
    steps = None

    if len(node.input) > 3 and node.input[3]:
        axes = builder.get_value(node.input[3])
    if len(node.input) > 4 and node.input[4]:
        steps = builder.get_value(node.input[4])

    # Use torch.narrow for simple cases, or dynamic slicing
    return builder.call_function(
        _dynamic_slice,
        args=(x, starts, ends, axes, steps),
    )


def _dynamic_slice(x, starts, ends, axes=None, steps=None):
    """Helper function for dynamic slicing."""
    import torch

    # Convert to lists if tensors
    if isinstance(starts, torch.Tensor):
        starts = starts.tolist()
    if isinstance(ends, torch.Tensor):
        ends = ends.tolist()
    if axes is not None and isinstance(axes, torch.Tensor):
        axes = axes.tolist()
    if steps is not None and isinstance(steps, torch.Tensor):
        steps = steps.tolist()

    ndim = x.dim()
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)

    # Build slice objects for each dimension
    slices = [slice(None)] * ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        # Handle negative indices
        dim_size = x.size(axis)
        if start < 0:
            start = max(0, dim_size + start)
        if end < 0:
            end = max(0, dim_size + end)
        # Clamp to valid range
        end = min(end, dim_size)
        slices[axis] = slice(int(start), int(end), int(step))

    return x[tuple(slices)]


@register("Gather")
def gather(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gather elements along an axis.

    ONNX Gather behavior:
    - If indices is a scalar, the axis dimension is removed from the output
    - If indices is a 1D tensor, the axis dimension is replaced by the indices dimension
    """
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    axis = get_attribute(node, "axis", 0)

    def _gather(data, indices, axis):
        # Handle scalar indices - need to squeeze the dimension after gather
        if indices.ndim == 0:
            # Scalar index: select single element along axis, removing that dimension
            return torch.index_select(data, axis, indices.unsqueeze(0)).squeeze(axis)
        else:
            return torch.index_select(data, axis, indices.flatten())

    return builder.call_function(_gather, args=(x, indices, axis))


@register("GatherElements")
def gather_elements(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gather elements using indices with same rank as input."""
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    axis = get_attribute(node, "axis", 0)
    return builder.call_function(torch.gather, args=(x, axis, indices))


@register("GatherND")
def gather_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gather slices using n-dimensional indices."""
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    batch_dims = get_attribute(node, "batch_dims", 0)

    def _gather_nd(data, indices, batch_dims=0):
        # Simplified GatherND implementation
        indices = indices.long()
        if batch_dims == 0:
            # Flatten indices to list of coordinate tuples
            idx_shape = indices.shape
            indices_flat = indices.reshape(-1, idx_shape[-1])
            result = torch.stack([data[tuple(idx)] for idx in indices_flat])
            return result.reshape(idx_shape[:-1] + data.shape[indices.shape[-1]:])
        else:
            raise NotImplementedError("batch_dims > 0 not yet supported for GatherND")

    return builder.call_function(_gather_nd, args=(x, indices, batch_dims))


@register("ScatterElements")
def scatter_elements(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scatter elements using indices."""
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    updates = builder.get_value(node.input[2])
    axis = get_attribute(node, "axis", 0)
    return builder.call_function(torch.scatter, args=(x, axis, indices, updates))


# =============================================================================
# Tiling and padding operators
# =============================================================================


@register("Tile")
def tile(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Tile tensor by repeating."""
    x = builder.get_value(node.input[0])
    repeats = builder.get_value(node.input[1])
    return builder.call_function(torch.tile, args=(x, repeats))


@register("Pad")
def pad(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Pad tensor."""
    import torch.nn.functional as F

    x = builder.get_value(node.input[0])
    pads = builder.get_value(node.input[1])
    mode = get_attribute(node, "mode", "constant")
    
    constant_value = 0.0
    if len(node.input) > 2 and node.input[2]:
        constant_value = builder.get_value(node.input[2])

    # Convert ONNX pad format to PyTorch format
    # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    # PyTorch: [xn_begin, xn_end, ..., x1_begin, x1_end]
    def _convert_pads(x, pads, mode, constant_value):
        import torch
        import torch.nn.functional as F

        if isinstance(pads, torch.Tensor):
            pads = pads.tolist()
        
        n = len(pads) // 2
        # Reverse and interleave
        torch_pads = []
        for i in range(n - 1, -1, -1):
            torch_pads.extend([int(pads[i]), int(pads[i + n])])

        mode_map = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}
        torch_mode = mode_map.get(mode, "constant")

        if torch_mode == "constant":
            return F.pad(x, torch_pads, mode=torch_mode, value=float(constant_value))
        return F.pad(x, torch_pads, mode=torch_mode)

    return builder.call_function(_convert_pads, args=(x, pads, mode, constant_value))


# =============================================================================
# Shape operators
# =============================================================================


@register("Shape")
def shape(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get tensor shape."""
    x = builder.get_value(node.input[0])
    start = get_attribute(node, "start", 0)
    end = get_attribute(node, "end")

    def _get_shape(t, start, end):
        shape = torch.tensor(t.shape, dtype=torch.int64)
        if end is None:
            return shape[start:]
        return shape[start:end]

    return builder.call_function(_get_shape, args=(x, start, end))


@register("Size")
def size(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get total number of elements."""
    x = builder.get_value(node.input[0])
    return builder.call_function(lambda t: torch.tensor(t.numel(), dtype=torch.int64), args=(x,))


@register("ConstantOfShape")
def constant_of_shape(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create tensor filled with constant value."""
    shape = builder.get_value(node.input[0])
    value = get_attribute(node, "value")

    if value is not None:
        fill_value = value.item() if hasattr(value, "item") else float(value.flatten()[0])
        dtype = value.dtype
    else:
        fill_value = 0.0
        dtype = torch.float32

    def _constant_of_shape(shape, fill_value, dtype):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        return torch.full(shape, fill_value, dtype=dtype)

    return builder.call_function(_constant_of_shape, args=(shape, fill_value, dtype))
