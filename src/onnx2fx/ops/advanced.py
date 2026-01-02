# SPDX-License-Identifier: Apache-2.0
"""Advanced tensor and sequence operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Einsum and advanced matrix operations
# =============================================================================


@register("Einsum")
def einsum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Einstein summation."""
    equation = get_attribute(node, "equation")
    inputs = [builder.get_value(name) for name in node.input]
    return builder.call_function(torch.einsum, args=(equation, *inputs))


@register("MatMulInteger")
def matmul_integer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Integer matrix multiplication."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])

    a_zero_point = None
    b_zero_point = None
    if len(node.input) > 2 and node.input[2]:
        a_zero_point = builder.get_value(node.input[2])
    if len(node.input) > 3 and node.input[3]:
        b_zero_point = builder.get_value(node.input[3])

    def _matmul_integer(a, b, a_zp, b_zp):
        a_adj = a.to(torch.int32)
        b_adj = b.to(torch.int32)
        if a_zp is not None:
            a_adj = a_adj - a_zp.to(torch.int32)
        if b_zp is not None:
            b_adj = b_adj - b_zp.to(torch.int32)
        return torch.matmul(a_adj, b_adj)

    return builder.call_function(
        _matmul_integer, args=(a, b, a_zero_point, b_zero_point)
    )


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


@register("TopK")
def topk(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Find top K values and indices."""
    x = builder.get_value(node.input[0])
    k = builder.get_value(node.input[1])

    axis = get_attribute(node, "axis", -1)
    largest = get_attribute(node, "largest", 1)
    sorted_ = get_attribute(node, "sorted", 1)

    def _topk(x, k, axis, largest, sorted_):
        k_val = k.item() if isinstance(k, torch.Tensor) else k
        values, indices = torch.topk(
            x, int(k_val), dim=axis, largest=bool(largest), sorted=bool(sorted_)
        )
        return values, indices

    result = builder.call_function(_topk, args=(x, k, axis, largest, sorted_))

    # Handle multiple outputs
    for i, output_name in enumerate(node.output):
        if output_name:
            idx_node = builder.call_function(lambda t, idx: t[idx], args=(result, i))
            builder.env[output_name] = idx_node

    return result


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


@register("NonMaxSuppression")
def nms(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Non-maximum suppression for object detection."""
    boxes = builder.get_value(node.input[0])
    scores = builder.get_value(node.input[1])

    max_output = None
    iou_threshold = 0.0
    score_threshold = float("-inf")

    if len(node.input) > 2 and node.input[2]:
        max_output = builder.get_value(node.input[2])
    if len(node.input) > 3 and node.input[3]:
        iou_threshold = builder.get_value(node.input[3])
    if len(node.input) > 4 and node.input[4]:
        score_threshold = builder.get_value(node.input[4])

    center_point_box = get_attribute(node, "center_point_box", 0)

    def _nms(
        boxes, scores, max_output, iou_threshold, score_threshold, center_point_box
    ):
        from torchvision.ops import nms as tv_nms

        batch_size = boxes.shape[0]
        num_classes = scores.shape[1]

        iou_th = (
            iou_threshold.item()
            if isinstance(iou_threshold, torch.Tensor)
            else iou_threshold
        )
        score_th = (
            score_threshold.item()
            if isinstance(score_threshold, torch.Tensor)
            else score_threshold
        )
        max_out = (
            max_output.item()
            if isinstance(max_output, torch.Tensor) and max_output is not None
            else max_output
        )

        results = []
        for batch_idx in range(batch_size):
            batch_boxes = boxes[batch_idx]  # [num_boxes, 4]

            # Convert center format to corner format if needed
            if center_point_box:
                cx, cy, w, h = (
                    batch_boxes[:, 0],
                    batch_boxes[:, 1],
                    batch_boxes[:, 2],
                    batch_boxes[:, 3],
                )
                batch_boxes = torch.stack(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
                )

            for class_idx in range(num_classes):
                class_scores = scores[batch_idx, class_idx]  # [num_boxes]

                # Filter by score threshold
                mask = class_scores > score_th
                if not mask.any():
                    continue

                filtered_boxes = batch_boxes[mask]
                filtered_scores = class_scores[mask]

                # Apply NMS
                keep = tv_nms(filtered_boxes, filtered_scores, iou_th)

                if max_out is not None:
                    keep = keep[: int(max_out)]

                # Get original indices
                original_indices = torch.where(mask)[0][keep]

                for idx in original_indices:
                    results.append([batch_idx, class_idx, idx.item()])

        if len(results) == 0:
            return torch.zeros((0, 3), dtype=torch.int64)
        return torch.tensor(results, dtype=torch.int64)

    return builder.call_function(
        _nms,
        args=(
            boxes,
            scores,
            max_output,
            iou_threshold,
            score_threshold,
            center_point_box,
        ),
    )


# =============================================================================
# Bitwise operations
# =============================================================================


@register("BitwiseAnd")
def bitwise_and(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise AND."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_and, args=(a, b))


@register("BitwiseOr")
def bitwise_or(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise OR."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_or, args=(a, b))


@register("BitwiseXor")
def bitwise_xor(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise XOR."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.bitwise_xor, args=(a, b))


@register("BitwiseNot")
def bitwise_not(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Bitwise NOT."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.bitwise_not, args=(x,))


# =============================================================================
# Trigonometric functions
# =============================================================================


@register("Sin")
def sin(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sin, args=(x,))


@register("Cos")
def cos(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.cos, args=(x,))


@register("Tan")
def tan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tan, args=(x,))


@register("Asin")
def asin(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.asin, args=(x,))


@register("Acos")
def acos(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.acos, args=(x,))


@register("Atan")
def atan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Arc tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.atan, args=(x,))


@register("Sinh")
def sinh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hyperbolic sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sinh, args=(x,))


@register("Cosh")
def cosh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hyperbolic cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.cosh, args=(x,))


@register("Asinh")
def asinh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic sine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.asinh, args=(x,))


@register("Acosh")
def acosh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic cosine."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.acosh, args=(x,))


@register("Atanh")
def atanh(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Inverse hyperbolic tangent."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.atanh, args=(x,))


# =============================================================================
# Additional math functions
# =============================================================================


@register("Erf")
def erf(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Error function."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.erf, args=(x,))


@register("IsNaN")
def isnan(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Check for NaN."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.isnan, args=(x,))


@register("IsInf")
def isinf(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Check for Inf."""
    x = builder.get_value(node.input[0])
    detect_negative = get_attribute(node, "detect_negative", 1)
    detect_positive = get_attribute(node, "detect_positive", 1)

    def _isinf(x, detect_neg, detect_pos):
        if detect_neg and detect_pos:
            return torch.isinf(x)
        elif detect_pos:
            return torch.isposinf(x)
        elif detect_neg:
            return torch.isneginf(x)
        else:
            return torch.zeros_like(x, dtype=torch.bool)

    return builder.call_function(_isinf, args=(x, detect_negative, detect_positive))


@register("Det")
def det(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Matrix determinant."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.linalg.det, args=(x,))


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


@register("CumSum")
def cumsum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Cumulative sum."""
    x = builder.get_value(node.input[0])
    axis = builder.get_value(node.input[1])

    exclusive = get_attribute(node, "exclusive", 0)
    reverse = get_attribute(node, "reverse", 0)

    def _cumsum(x, axis, exclusive, reverse):
        ax = axis.item() if isinstance(axis, torch.Tensor) else axis

        if reverse:
            x = torch.flip(x, [int(ax)])

        result = torch.cumsum(x, dim=int(ax))

        if exclusive:
            # Shift by one and pad with zero
            pad_shape = list(x.shape)
            pad_shape[int(ax)] = 1
            zero_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            result = torch.cat(
                [zero_pad, result.narrow(int(ax), 0, x.shape[int(ax)] - 1)], dim=int(ax)
            )

        if reverse:
            result = torch.flip(result, [int(ax)])

        return result

    return builder.call_function(_cumsum, args=(x, axis, exclusive, reverse))


@register("ReverseSequence")
def reverse_sequence(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Reverse sequences in a tensor."""
    x = builder.get_value(node.input[0])
    sequence_lens = builder.get_value(node.input[1])

    batch_axis = get_attribute(node, "batch_axis", 1)
    time_axis = get_attribute(node, "time_axis", 0)

    def _reverse_sequence(x, sequence_lens, batch_axis, time_axis):
        result = x.clone()
        for i, seq_len in enumerate(sequence_lens):
            seq_len_val = (
                seq_len.item() if isinstance(seq_len, torch.Tensor) else seq_len
            )
            # Create indices for this batch
            idx = [slice(None)] * x.dim()
            idx[batch_axis] = i
            idx[time_axis] = slice(None, int(seq_len_val))

            reversed_idx = list(idx)
            reversed_idx[time_axis] = slice(int(seq_len_val) - 1, None, -1)

            result[tuple(idx)] = x[tuple(reversed_idx)]

        return result

    return builder.call_function(
        _reverse_sequence, args=(x, sequence_lens, batch_axis, time_axis)
    )


# =============================================================================
# Image operations
# =============================================================================


@register("Resize")
def resize(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Resize tensor using interpolation."""
    x = builder.get_value(node.input[0])

    # roi, scales, sizes are optional inputs
    roi = None
    scales = None
    sizes = None

    if len(node.input) > 1 and node.input[1]:
        roi = builder.get_value(node.input[1])
    if len(node.input) > 2 and node.input[2]:
        scales = builder.get_value(node.input[2])
    if len(node.input) > 3 and node.input[3]:
        sizes = builder.get_value(node.input[3])

    mode = get_attribute(node, "mode", "nearest")
    coordinate_transformation_mode = get_attribute(
        node, "coordinate_transformation_mode", "half_pixel"
    )

    def _resize(x, roi, scales, sizes, mode, coord_mode):
        import torch.nn.functional as F

        # Map ONNX mode to PyTorch mode
        mode_map = {
            "nearest": "nearest",
            "linear": "bilinear" if x.dim() == 4 else "linear",
            "cubic": "bicubic",
        }
        torch_mode = mode_map.get(mode, "nearest")

        # Determine align_corners based on coordinate transformation mode
        align_corners = coord_mode == "align_corners"
        if torch_mode == "nearest":
            align_corners = None

        if sizes is not None:
            # Use explicit sizes
            size_list = sizes.tolist() if isinstance(sizes, torch.Tensor) else sizes
            # Skip batch and channel dimensions
            output_size = [int(s) for s in size_list[2:]]
        elif scales is not None:
            # Use scales
            scale_list = scales.tolist() if isinstance(scales, torch.Tensor) else scales
            input_shape = x.shape[2:]
            output_size = [int(s * sc) for s, sc in zip(input_shape, scale_list[2:])]
        else:
            return x

        kwargs = {"size": output_size, "mode": torch_mode}
        if align_corners is not None and torch_mode not in ["nearest", "area"]:
            kwargs["align_corners"] = align_corners

        return F.interpolate(x, **kwargs)

    return builder.call_function(
        _resize, args=(x, roi, scales, sizes, mode, coordinate_transformation_mode)
    )


@register("DepthToSpace")
def depth_to_space(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Rearrange depth to spatial dimensions."""
    x = builder.get_value(node.input[0])
    blocksize = get_attribute(node, "blocksize")
    mode = get_attribute(node, "mode", "DCR")

    def _depth_to_space(x, blocksize, mode):
        b, c, h, w = x.shape
        if mode == "DCR":
            # Depth-Column-Row
            x = x.reshape(b, blocksize, blocksize, c // (blocksize**2), h, w)
            x = x.permute(0, 3, 4, 1, 5, 2)
            x = x.reshape(b, c // (blocksize**2), h * blocksize, w * blocksize)
        else:
            # CRD mode
            x = x.reshape(b, c // (blocksize**2), blocksize, blocksize, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(b, c // (blocksize**2), h * blocksize, w * blocksize)
        return x

    return builder.call_function(_depth_to_space, args=(x, blocksize, mode))


@register("SpaceToDepth")
def space_to_depth(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Rearrange spatial dimensions to depth."""
    x = builder.get_value(node.input[0])
    blocksize = get_attribute(node, "blocksize")

    def _space_to_depth(x, blocksize):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // blocksize, blocksize, w // blocksize, blocksize)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(b, c * blocksize * blocksize, h // blocksize, w // blocksize)
        return x

    return builder.call_function(_space_to_depth, args=(x, blocksize))
