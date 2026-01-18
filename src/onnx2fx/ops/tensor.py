# SPDX-License-Identifier: Apache-2.0
"""Tensor manipulation operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..exceptions import ConversionError
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
        raise ConversionError(
            "Constant node has no value attribute",
            node_name=node.name,
            op_type="Constant",
        )

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
        raise ConversionError(
            f"Unsupported cast target dtype: {to_dtype}",
            node_name=node.name,
            op_type="Cast",
        )

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
    """Reshape tensor to a new shape.

    ONNX Reshape semantics:
    - A value of 0 means the dimension is unchanged from the input shape
    - A value of -1 means the dimension is inferred from the remaining elements
    """
    x = builder.get_value(node.input[0])
    shape = builder.get_value(node.input[1])

    # Check allowzero attribute (default is 0, meaning 0 copies from input)
    allowzero = get_attribute(node, "allowzero", 0)

    def _reshape(t, shape, allowzero):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        else:
            shape = list(shape)

        # ONNX: if allowzero=0, a value of 0 in shape means copy from input
        if not allowzero:
            for i, dim in enumerate(shape):
                if dim == 0:
                    if i < t.dim():
                        shape[i] = t.shape[i]

        return torch.reshape(t, tuple(shape))

    return builder.call_function(_reshape, args=(x, shape, allowzero))


@register("Transpose")
def transpose(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Transpose tensor dimensions."""
    x = builder.get_value(node.input[0])
    perm = get_attribute(node, "perm")
    if perm is None:
        # Default: reverse all dimensions
        return builder.call_function(lambda t: t.T, args=(x,))
    return builder.call_function(torch.permute, args=(x, perm))


@register("Squeeze", since_version=1)
def squeeze_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Remove dimensions of size 1 for opset 1-12.

    In opset < 13, axes is an attribute.
    """
    x = builder.get_value(node.input[0])

    axes = get_attribute(node, "axes")
    if axes is not None:
        # Squeeze specific dimensions
        result = x
        # Sort in reverse to maintain correct indices after each squeeze
        for axis in sorted(axes, reverse=True):
            result = builder.call_function(
                torch.squeeze, args=(result,), kwargs={"dim": axis}
            )
        return result
    return builder.call_function(torch.squeeze, args=(x,))


@register("Squeeze", since_version=13)
def squeeze_v13(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Remove dimensions of size 1 for opset 13+.

    In opset 13+, axes is an optional input (not attribute).
    """
    x = builder.get_value(node.input[0])

    # axes is an optional input in opset 13+
    if len(node.input) > 1 and node.input[1]:
        axes = builder.get_value(node.input[1])

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

    # No axes input - squeeze all dimensions of size 1
    return builder.call_function(torch.squeeze, args=(x,))


@register("Unsqueeze", since_version=1)
def unsqueeze_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Insert dimensions of size 1 for opset 1-12.

    In opset < 13, axes is a required attribute.
    """
    x = builder.get_value(node.input[0])

    axes = get_attribute(node, "axes")
    if axes is None:
        raise ConversionError(
            "Unsqueeze requires axes attribute in opset < 13",
            node_name=node.name,
            op_type="Unsqueeze",
        )

    # Handle single axis
    if isinstance(axes, int):
        return builder.call_function(torch.unsqueeze, args=(x, axes))

    # Handle multiple axes - unsqueeze in sorted order
    result = x
    for axis in sorted(axes):
        result = builder.call_function(torch.unsqueeze, args=(result, axis))
    return result


@register("Unsqueeze", since_version=13)
def unsqueeze_v13(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Insert dimensions of size 1 for opset 13+.

    In opset 13+, axes is a required input (not attribute).
    """
    x = builder.get_value(node.input[0])

    if len(node.input) < 2 or not node.input[1]:
        raise ConversionError(
            "Unsqueeze requires axes input in opset 13+",
            node_name=node.name,
            op_type="Unsqueeze",
        )

    axes = builder.get_value(node.input[1])

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


@register("Flatten")
def flatten(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Flatten tensor to 2D.

    ONNX Flatten reshapes the input tensor to a 2D tensor:
    - First dimension = product of dimensions from 0 to axis-1
    - Second dimension = product of dimensions from axis to end
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 1)

    def _flatten_to_2d(t, axis):
        shape = t.shape
        # Handle negative axis
        if axis < 0:
            axis = len(shape) + axis
        # Compute dimensions
        dim0 = 1
        for i in range(axis):
            dim0 *= shape[i]
        dim1 = 1
        for i in range(axis, len(shape)):
            dim1 *= shape[i]
        return t.reshape(dim0, dim1)

    return builder.call_function(_flatten_to_2d, args=(x, axis))


@register("Expand")
def expand(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Broadcast tensor to a new shape.

    ONNX Expand uses bidirectional broadcasting, which means:
    - If target dim is 1, keep the original dimension
    - The output shape is max(input_dim, target_dim) for each dimension
    """
    x = builder.get_value(node.input[0])
    shape = builder.get_value(node.input[1])

    def _expand(t, shape):
        if isinstance(shape, torch.Tensor):
            shape = tuple(int(s) for s in shape.tolist())
        # Use broadcast_shapes to compute the actual broadcast shape
        # This handles cases where target_dim=1 should preserve input_dim
        broadcast_shape = torch.broadcast_shapes(t.shape, shape)
        return t.expand(broadcast_shape)

    return builder.call_function(_expand, args=(x, shape))


# =============================================================================
# Concatenation and splitting operators
# =============================================================================


@register("Concat")
def concat(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Concatenate tensors along an axis."""
    inputs = [builder.get_value(name) for name in node.input]
    axis = get_attribute(node, "axis", 0)
    return builder.call_function(torch.cat, args=(inputs,), kwargs={"dim": axis})


@register("Split", since_version=1)
def split_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Split tensor into chunks for opset 1-12.

    In opset < 13, split sizes is an optional attribute.
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 0)

    split_attr = get_attribute(node, "split")
    if split_attr is not None:
        result = builder.call_function(torch.split, args=(x, list(split_attr), axis))
    else:
        # Default: split into equal parts based on number of outputs
        result = builder.call_function(torch.chunk, args=(x, len(node.output), axis))

    # Handle multiple outputs
    for i, output_name in enumerate(node.output):
        if output_name:
            idx_node = builder.call_function(lambda t, idx: t[idx], args=(result, i))
            builder.env[output_name] = idx_node

    return result


@register("Split", since_version=13)
def split_v13(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Split tensor into chunks for opset 13+.

    In opset 13+, split sizes is an optional input.
    In opset 18+, num_outputs attribute was added.
    """
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", 0)
    num_outputs = get_attribute(node, "num_outputs")  # Added in opset 18

    # split sizes is an optional input in opset 13+
    if len(node.input) > 1 and node.input[1]:
        split_sizes = builder.get_value(node.input[1])

        def _split_with_sizes(t, sizes, dim):
            if hasattr(sizes, "tolist"):
                sizes = sizes.tolist()
            return torch.split(t, sizes, dim)

        result = builder.call_function(_split_with_sizes, args=(x, split_sizes, axis))
    elif num_outputs is not None:
        # Split into equal parts using num_outputs (opset 18+)
        result = builder.call_function(torch.chunk, args=(x, num_outputs, axis))
    else:
        # Default: split into equal parts based on number of outputs
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


@register("Slice", since_version=1)
def slice_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Slice tensor along axes (opset 1-9).

    In opset < 10, starts, ends, and axes are attributes.
    """
    x = builder.get_value(node.input[0])
    starts = get_attribute(node, "starts")
    ends = get_attribute(node, "ends")
    axes = get_attribute(node, "axes")
    # Note: steps attribute doesn't exist in opset < 10

    return builder.call_function(
        _dynamic_slice,
        args=(x, list(starts), list(ends), list(axes) if axes else None, None),
    )


@register("Slice", since_version=10)
def slice_v10(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Slice tensor along axes (opset 10+).

    In opset 10+, starts, ends, axes, and steps are inputs.
    """
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
    - output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
    - If indices is a scalar, the axis dimension is removed from the output
    - If indices is a multi-dimensional tensor, indices.shape replaces the axis dimension
    """
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    axis = get_attribute(node, "axis", 0)

    def _gather(data, indices, axis):
        indices = indices.long()

        if axis < 0:
            axis = data.dim() + axis

        # Handle scalar indices - need to squeeze the dimension after gather
        if indices.ndim == 0:
            # Scalar index: select single element along axis, removing that dimension
            return torch.index_select(data, axis, indices.unsqueeze(0)).squeeze(axis)

        # For multi-dimensional indices, we need proper ONNX Gather semantics
        # Move the gather axis to position 0
        if axis != 0:
            data = data.movedim(axis, 0)

        # Flatten indices for indexing
        indices_flat = indices.flatten()
        gathered = data[indices_flat]  # [num_indices, ...]

        # Reshape to restore indices dimensions
        new_shape = list(indices.shape) + list(data.shape[1:])
        gathered = gathered.view(new_shape)

        # Move the original leading dimensions back
        if axis != 0:
            # Permute dimensions to restore original order
            # Current: [idx..., prefix..., suffix...]
            # Target:  [prefix..., idx..., suffix...]
            num_idx_dims = indices.ndim
            num_prefix_dims = axis

            perm = (
                list(range(num_idx_dims, num_idx_dims + num_prefix_dims))
                + list(range(num_idx_dims))
                + list(range(num_idx_dims + num_prefix_dims, gathered.ndim))
            )
            gathered = gathered.permute(perm)

        return gathered

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
            return result.reshape(idx_shape[:-1] + data.shape[indices.shape[-1] :])
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

    def _tile(t, reps):
        if isinstance(reps, torch.Tensor):
            reps = tuple(int(r) for r in reps.tolist())
        return torch.tile(t, reps)

    return builder.call_function(_tile, args=(x, repeats))


def _pad_impl(x, pads, mode, constant_value):
    """Helper function for Pad operator.

    Converts ONNX pad format to PyTorch format and applies padding.
    ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    PyTorch: [xn_begin, xn_end, ..., x1_begin, x1_end]
    """
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

    # For non-constant modes (reflect, replicate), PyTorch only supports padding
    # the last N dimensions. Trim leading zero-padding pairs.
    # torch_pads is ordered as [last_dim_begin, last_dim_end, ..., first_dim_begin, first_dim_end]
    # We need to trim trailing zero pairs (which correspond to first dimensions).
    while len(torch_pads) > 2 and torch_pads[-1] == 0 and torch_pads[-2] == 0:
        torch_pads = torch_pads[:-2]

    return F.pad(x, torch_pads, mode=torch_mode)


@register("Pad", since_version=1)
def pad_v1(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Pad tensor (opset 1-10).

    In opset < 11, pads and value are attributes.
    """
    x = builder.get_value(node.input[0])
    pads = list(get_attribute(node, "pads"))
    mode = get_attribute(node, "mode", "constant")
    constant_value = get_attribute(node, "value", 0.0)

    return builder.call_function(_pad_impl, args=(x, pads, mode, constant_value))


@register("Pad", since_version=11)
def pad_v11(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Pad tensor (opset 11+).

    In opset 11+, pads, constant_value, and axes are inputs.
    """
    x = builder.get_value(node.input[0])
    pads = builder.get_value(node.input[1])
    mode = get_attribute(node, "mode", "constant")

    constant_value = 0.0
    if len(node.input) > 2 and node.input[2]:
        constant_value = builder.get_value(node.input[2])

    # Note: axes input (opset 18+) is not yet supported
    # If needed, would require reordering pads based on axes

    return builder.call_function(_pad_impl, args=(x, pads, mode, constant_value))


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
    return builder.call_function(
        lambda t: torch.tensor(t.numel(), dtype=torch.int64), args=(x,)
    )


@register("ConstantOfShape")
def constant_of_shape(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create tensor filled with constant value."""
    shape = builder.get_value(node.input[0])
    value = get_attribute(node, "value")

    if value is not None:
        fill_value = (
            value.item() if hasattr(value, "item") else float(value.flatten()[0])
        )
        dtype = value.dtype
    else:
        fill_value = 0.0
        dtype = torch.float32

    def _constant_of_shape(shape, fill_value, dtype):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        return torch.full(shape, fill_value, dtype=dtype)

    return builder.call_function(_constant_of_shape, args=(shape, fill_value, dtype))


# =============================================================================
# Tensor generation operators
# =============================================================================


@register("Range")
def range_(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate a range of values."""
    start = builder.get_value(node.input[0])
    limit = builder.get_value(node.input[1])
    delta = builder.get_value(node.input[2])

    def _range(start, limit, delta):
        # Extract scalar values
        st = start.item() if isinstance(start, torch.Tensor) else start
        lim = limit.item() if isinstance(limit, torch.Tensor) else limit
        dlt = delta.item() if isinstance(delta, torch.Tensor) else delta
        dtype = start.dtype if isinstance(start, torch.Tensor) else torch.float32
        return torch.arange(st, lim, dlt, dtype=dtype)

    return builder.call_function(_range, args=(start, limit, delta))


@register("OneHot")
def one_hot(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """One-hot encoding."""
    indices = builder.get_value(node.input[0])
    depth = builder.get_value(node.input[1])
    values = builder.get_value(node.input[2])

    axis = get_attribute(node, "axis", -1)

    def _one_hot(indices, depth, values, axis):
        d = depth.item() if isinstance(depth, torch.Tensor) else depth
        off_value = values[0]
        on_value = values[1]

        # Create one-hot tensor
        result = torch.nn.functional.one_hot(indices.long(), int(d))
        result = result.to(values.dtype)

        # Apply on/off values
        result = result * (on_value - off_value) + off_value

        # Move axis if needed
        if axis != -1 and axis != indices.dim():
            # Permute to move the one-hot dimension to the correct axis
            ndim = result.dim()
            if axis < 0:
                axis = ndim + axis
            perm = list(range(ndim - 1))
            perm.insert(axis, ndim - 1)
            result = result.permute(perm)

        return result

    return builder.call_function(_one_hot, args=(indices, depth, values, axis))


@register("NonZero")
def non_zero(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Find indices of non-zero elements."""
    x = builder.get_value(node.input[0])

    def _non_zero(x):
        # ONNX returns shape (rank, num_nonzero), PyTorch returns tuple
        result = torch.nonzero(x, as_tuple=False).T
        return result.to(torch.int64)

    return builder.call_function(_non_zero, args=(x,))


@register("Unique")
def unique(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Find unique elements."""
    x = builder.get_value(node.input[0])

    axis = get_attribute(node, "axis")
    sorted_ = get_attribute(node, "sorted", 1)

    def _unique(x, axis, sorted_):
        if axis is not None:
            return torch.unique(
                x,
                sorted=bool(sorted_),
                return_inverse=True,
                return_counts=True,
                dim=axis,
            )
        return torch.unique(
            x, sorted=bool(sorted_), return_inverse=True, return_counts=True
        )

    return builder.call_function(_unique, args=(x, axis, sorted_))


@register("Trilu")
def trilu(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Triangular part of matrix."""
    x = builder.get_value(node.input[0])

    k = 0
    if len(node.input) > 1 and node.input[1]:
        k = builder.get_value(node.input[1])

    upper = get_attribute(node, "upper", 1)

    def _trilu(x, k, upper):
        k_val = k.item() if isinstance(k, torch.Tensor) else k
        if upper:
            return torch.triu(x, diagonal=int(k_val))
        return torch.tril(x, diagonal=int(k_val))

    return builder.call_function(_trilu, args=(x, k, upper))


@register("EyeLike")
def eye_like(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an identity matrix with the same shape as input.

    Note: The dtype attribute is ignored; output uses input tensor's dtype.
    """
    x = builder.get_value(node.input[0])
    k = get_attribute(node, "k", 0)

    def _eye_like(t: torch.Tensor, diag: int) -> torch.Tensor:
        n, m = t.shape[-2], t.shape[-1]
        eye = torch.eye(n, m, dtype=t.dtype, device=t.device)
        if diag != 0:
            eye = torch.diagonal(eye, offset=diag)
        return eye

    return builder.call_function(_eye_like, args=(x, k))


# =============================================================================
# Scatter ND operators
# =============================================================================


@register("ScatterND")
def scatter_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scatter updates into data at indices."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    updates = builder.get_value(node.input[2])

    _reduction = get_attribute(node, "reduction", "none")

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
# Select and Compress operators
# =============================================================================


@register("Select")
def select_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Select elements based on indices (like advanced indexing)."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    return builder.call_function(torch.index_select, args=(data, 0, indices))


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
