# SPDX-License-Identifier: Apache-2.0
"""Neural network layer operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Matrix multiplication operators
# =============================================================================


@register("MatMul")
def matmul(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Matrix multiplication."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])
    return builder.call_function(torch.matmul, args=(a, b))


@register("Gemm")
def gemm(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """General Matrix Multiplication: Y = alpha * A' * B' + beta * C."""
    a = builder.get_value(node.input[0])
    b = builder.get_value(node.input[1])

    alpha = get_attribute(node, "alpha", 1.0)
    beta = get_attribute(node, "beta", 1.0)
    trans_a = get_attribute(node, "transA", 0)
    trans_b = get_attribute(node, "transB", 0)

    def _gemm(a, b, c, alpha, beta, trans_a, trans_b):
        if trans_a:
            a = a.T
        if trans_b:
            b = b.T
        result = alpha * torch.matmul(a, b)
        if c is not None:
            result = result + beta * c
        return result

    c = None
    if len(node.input) > 2 and node.input[2]:
        c = builder.get_value(node.input[2])

    return builder.call_function(_gemm, args=(a, b, c, alpha, beta, trans_a, trans_b))


# =============================================================================
# Convolution operators
# =============================================================================


def _get_conv_params(node: onnx.NodeProto) -> dict:
    """Extract common convolution parameters from node attributes."""
    return {
        "dilations": get_attribute(node, "dilations"),
        "group": get_attribute(node, "group", 1),
        "kernel_shape": get_attribute(node, "kernel_shape"),
        "pads": get_attribute(node, "pads"),
        "strides": get_attribute(node, "strides"),
        "auto_pad": get_attribute(node, "auto_pad", "NOTSET"),
    }


@register("Conv")
def conv(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """N-dimensional convolution."""
    x = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    params = _get_conv_params(node)
    strides = params["strides"] or [1]
    dilations = params["dilations"] or [1]
    group = params["group"]
    pads = params["pads"]
    auto_pad = params["auto_pad"]
    kernel_shape = params["kernel_shape"]

    def _conv(x, weight, bias, strides, dilations, group, pads, auto_pad, kernel_shape):
        ndim = len(weight.shape) - 2  # Exclude batch and channel dims

        # Expand strides and dilations to match ndim
        if len(strides) == 1:
            strides = strides * ndim
        if len(dilations) == 1:
            dilations = dilations * ndim

        # Handle padding
        padding = 0
        if pads is not None:
            n = len(pads) // 2
            symmetric = all(pads[i] == pads[i + n] for i in range(n))
            if symmetric:
                padding = tuple(pads[:n])
            else:
                # Asymmetric padding
                # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                # F.pad: [xn_begin, xn_end, ..., x1_begin, x1_end] (reverse order)
                pad_list = []
                for i in range(n - 1, -1, -1):
                    pad_list.extend([pads[i], pads[i + n]])
                x = F.pad(x, pad_list)
                padding = 0

        # Handle auto_pad
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Compute padding for SAME
            input_shape = x.shape[2:]
            output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]
            pad_total = [
                max(0, (o - 1) * st + (k - 1) * d + 1 - i)
                for i, o, k, st, d in zip(
                    input_shape,
                    output_shape,
                    kernel_shape or weight.shape[2:],
                    strides,
                    dilations,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p // 2, p - p // 2])
            else:
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p - p // 2, p // 2])
            x = F.pad(x, pad_list)
            padding = 0

        strides_tuple = tuple(strides) if len(strides) > 1 else strides[0]
        dilations_tuple = tuple(dilations) if len(dilations) > 1 else dilations[0]

        if ndim == 1:
            return F.conv1d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        elif ndim == 2:
            return F.conv2d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        elif ndim == 3:
            return F.conv3d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding,
                dilation=dilations_tuple,
                groups=group,
            )
        else:
            raise NotImplementedError(f"Conv{ndim}D not supported")

    return builder.call_function(
        _conv,
        args=(x, weight, bias, strides, dilations, group, pads, auto_pad, kernel_shape),
    )


@register("ConvTranspose")
def conv_transpose(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """N-dimensional transposed convolution."""
    x = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    strides = get_attribute(node, "strides") or [1]
    dilations = get_attribute(node, "dilations") or [1]
    group = get_attribute(node, "group", 1)
    pads = get_attribute(node, "pads")
    output_padding = get_attribute(node, "output_padding") or [0]
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
    output_shape = get_attribute(node, "output_shape")
    kernel_shape = get_attribute(node, "kernel_shape")

    def _conv_transpose(
        x,
        weight,
        bias,
        strides,
        dilations,
        group,
        pads,
        output_padding,
        auto_pad,
        output_shape,
        kernel_shape,
    ):
        ndim = len(weight.shape) - 2

        # Expand strides, dilations, and output_padding to match ndim
        if len(strides) == 1:
            strides = strides * ndim
        if len(dilations) == 1:
            dilations = dilations * ndim
        if len(output_padding) == 1:
            output_padding = output_padding * ndim

        # Get kernel shape from weight if not provided
        k_shape = kernel_shape if kernel_shape else list(weight.shape[2:])

        # Handle auto_pad and output_shape
        # For ConvTranspose, the output shape formula is:
        # output_shape = (input_shape - 1) * stride + (kernel_shape - 1) * dilation + 1 - pad_begin - pad_end + output_padding
        padding = [0] * ndim
        adj_output_padding = list(output_padding)

        if output_shape is not None:
            # Compute pads from output_shape
            # output_shape[i] = (input_shape[i] - 1) * stride[i] + (k - 1) * dilation[i] + 1 - total_pad[i] + output_pad[i]
            # total_pad[i] = (input_shape[i] - 1) * stride[i] + (k - 1) * dilation[i] + 1 - output_shape[i] + output_pad[i]
            input_shape = x.shape[2:]
            for i in range(ndim):
                default_output = (
                    (input_shape[i] - 1) * strides[i]
                    + (k_shape[i] - 1) * dilations[i]
                    + 1
                )
                total_pad = default_output - output_shape[i]
                if total_pad >= 0:
                    padding[i] = total_pad // 2
                    # Adjust output_padding to match the exact output_shape
                    adj_output_padding[i] = total_pad - 2 * padding[i]
                else:
                    # Need additional output_padding
                    padding[i] = 0
                    adj_output_padding[i] = -total_pad
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # For SAME auto_pad in ConvTranspose:
            # target output_shape = input_shape * stride
            # We do full conv_transpose without padding and then slice the output
            input_shape = x.shape[2:]
            target_shape = [input_shape[i] * strides[i] for i in range(ndim)]

            # Default output without padding
            default_output = [
                (input_shape[i] - 1) * strides[i] + (k_shape[i] - 1) * dilations[i] + 1
                for i in range(ndim)
            ]

            # Calculate how much to trim from each dimension
            trim_total = [default_output[i] - target_shape[i] for i in range(ndim)]

            # For SAME_UPPER: extra pad at end means trim from end
            # For SAME_LOWER: extra pad at begin means trim from begin
            if auto_pad == "SAME_UPPER":
                trim_begin = [t // 2 for t in trim_total]
                trim_end = [t - t // 2 for t in trim_total]
            else:  # SAME_LOWER
                trim_end = [t // 2 for t in trim_total]
                trim_begin = [t - t // 2 for t in trim_total]

            # Do full conv_transpose without padding
            strides_tuple = tuple(strides) if len(strides) > 1 else strides[0]
            dilations_tuple = tuple(dilations) if len(dilations) > 1 else dilations[0]

            if ndim == 1:
                result = F.conv_transpose1d(
                    x,
                    weight,
                    bias,
                    stride=strides_tuple,
                    padding=0,
                    output_padding=0,
                    groups=group,
                    dilation=dilations_tuple,
                )
                # Slice to get target shape
                end0 = result.shape[2] - trim_end[0] if trim_end[0] > 0 else None
                return result[:, :, trim_begin[0] : end0]
            elif ndim == 2:
                result = F.conv_transpose2d(
                    x,
                    weight,
                    bias,
                    stride=strides_tuple,
                    padding=0,
                    output_padding=0,
                    groups=group,
                    dilation=dilations_tuple,
                )
                # Slice to get target shape
                end0 = result.shape[2] - trim_end[0] if trim_end[0] > 0 else None
                end1 = result.shape[3] - trim_end[1] if trim_end[1] > 0 else None
                return result[:, :, trim_begin[0] : end0, trim_begin[1] : end1]
            elif ndim == 3:
                result = F.conv_transpose3d(
                    x,
                    weight,
                    bias,
                    stride=strides_tuple,
                    padding=0,
                    output_padding=0,
                    groups=group,
                    dilation=dilations_tuple,
                )
                # Slice to get target shape
                end0 = result.shape[2] - trim_end[0] if trim_end[0] > 0 else None
                end1 = result.shape[3] - trim_end[1] if trim_end[1] > 0 else None
                end2 = result.shape[4] - trim_end[2] if trim_end[2] > 0 else None
                return result[
                    :,
                    :,
                    trim_begin[0] : end0,
                    trim_begin[1] : end1,
                    trim_begin[2] : end2,
                ]
            else:
                raise NotImplementedError(f"ConvTranspose{ndim}D not supported")
        elif pads is not None:
            n = len(pads) // 2
            padding = list(pads[:n])
            # Handle asymmetric pads via output_padding
            for i in range(n):
                if pads[i] != pads[i + n]:
                    adj_output_padding[i] = pads[i + n] - pads[i]

        strides_tuple = tuple(strides) if len(strides) > 1 else strides[0]
        dilations_tuple = tuple(dilations) if len(dilations) > 1 else dilations[0]
        padding_tuple = tuple(padding) if len(padding) > 1 else padding[0]
        output_padding_tuple = (
            tuple(adj_output_padding)
            if len(adj_output_padding) > 1
            else adj_output_padding[0]
        )

        if ndim == 1:
            return F.conv_transpose1d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding_tuple,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        elif ndim == 2:
            return F.conv_transpose2d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding_tuple,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        elif ndim == 3:
            return F.conv_transpose3d(
                x,
                weight,
                bias,
                stride=strides_tuple,
                padding=padding_tuple,
                output_padding=output_padding_tuple,
                groups=group,
                dilation=dilations_tuple,
            )
        else:
            raise NotImplementedError(f"ConvTranspose{ndim}D not supported")

    return builder.call_function(
        _conv_transpose,
        args=(
            x,
            weight,
            bias,
            strides,
            dilations,
            group,
            pads,
            output_padding,
            auto_pad,
            output_shape,
            kernel_shape,
        ),
    )


@register("DeformConv")
def deform_conv(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Deformable convolution.

    Performs deformable convolution as described in:
    - Deformable Convolutional Networks (https://arxiv.org/abs/1703.06211)
    - Deformable ConvNets v2 (https://arxiv.org/abs/1811.11168) when mask is provided

    Note: Only 2D deformable convolution is supported as torchvision.ops.deform_conv2d
    only supports 2D inputs.
    """
    import torchvision.ops

    x = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    offset = builder.get_value(node.input[2])

    bias = None
    if len(node.input) > 3 and node.input[3]:
        bias = builder.get_value(node.input[3])

    mask = None
    if len(node.input) > 4 and node.input[4]:
        mask = builder.get_value(node.input[4])

    strides = get_attribute(node, "strides") or [1, 1]
    dilations = get_attribute(node, "dilations") or [1, 1]
    pads = get_attribute(node, "pads") or [0, 0, 0, 0]
    # Note: group and offset_group are inferred from tensor shapes by torchvision
    # ONNX attributes are parsed but not explicitly passed to the function

    def _deform_conv(x, weight, offset, bias, mask, strides, dilations, pads):
        # Handle padding - ONNX uses [begin0, begin1, end0, end1] format
        # torchvision.ops.deform_conv2d expects (pad_H, pad_W)
        # For simplicity, assume symmetric padding (ONNX pads should be symmetric)
        n = len(pads) // 2
        padding = tuple(pads[:n])

        stride = tuple(strides) if len(strides) > 1 else (strides[0], strides[0])
        dilation = (
            tuple(dilations) if len(dilations) > 1 else (dilations[0], dilations[0])
        )

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask,
        )

    return builder.call_function(
        _deform_conv,
        args=(x, weight, offset, bias, mask, strides, dilations, pads),
    )


# =============================================================================
# Pooling operators
# =============================================================================


@register("MaxPool")
def max_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Max pooling."""
    x = builder.get_value(node.input[0])

    kernel_shape = get_attribute(node, "kernel_shape")
    # ONNX spec: strides defaults to 1 along each spatial axis (not kernel_shape)
    strides = get_attribute(node, "strides") or [1] * len(kernel_shape)
    pads = get_attribute(node, "pads")
    dilations = get_attribute(node, "dilations") or [1] * len(kernel_shape)
    ceil_mode = get_attribute(node, "ceil_mode", 0)
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
    storage_order = get_attribute(node, "storage_order", 0)

    # Check if we need to return indices (second output requested)
    return_indices = len(node.output) > 1 and node.output[1] != ""

    def _max_pool(
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        auto_pad,
        return_indices,
        storage_order,
    ):
        ndim = len(kernel_shape)
        input_dtype = x.dtype
        input_shape = x.shape  # (N, C, D1, D2, ...)

        # PyTorch max_pool doesn't support int8/uint8, need to convert
        needs_cast = input_dtype in (torch.int8, torch.uint8)
        if needs_cast:
            x = x.float()

        padding = 0
        # Handle auto_pad first (before explicit pads)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # ONNX spec for SAME padding (ceil_mode disabled):
            # output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides[i]) + 1
            spatial_shape = x.shape[2:]
            output_shape = [(s - 1) // st + 1 for s, st in zip(spatial_shape, strides)]
            # pad_shape[i] = (output_shape[i] - 1) * strides[i]
            #                + ((kernel[i] - 1) * dilations[i] + 1) - input_shape[i]
            pad_total = [
                max(0, (o - 1) * st + (k - 1) * d + 1 - i)
                for i, o, k, st, d in zip(
                    spatial_shape,
                    output_shape,
                    kernel_shape,
                    strides,
                    dilations,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p // 2, p - p // 2])
            else:
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p - p // 2, p // 2])
            x = F.pad(x, pad_list, value=float("-inf"))
            padding = 0
        elif pads is not None:
            n = len(pads) // 2
            symmetric = all(pads[i] == pads[i + n] for i in range(n))

            # Check if padding exceeds PyTorch's limit
            # PyTorch: pad should be at most half of effective kernel size
            # effective_kernel = (kernel_size - 1) * dilation + 1
            # max_pad = effective_kernel // 2
            max_allowed_pad = [
                ((k - 1) * d + 1) // 2 for k, d in zip(kernel_shape, dilations)
            ]
            exceeds_limit = any(
                pads[i] > max_allowed_pad[i] or pads[i + n] > max_allowed_pad[i]
                for i in range(n)
            )

            if symmetric and not exceeds_limit:
                padding = tuple(pads[:n])
            else:
                # Use explicit F.pad for asymmetric or large padding
                pad_list = []
                for i in range(n - 1, -1, -1):
                    pad_list.extend([pads[i], pads[i + n]])
                x = F.pad(x, pad_list, value=float("-inf"))
                padding = 0

        kernel = tuple(kernel_shape)
        stride = tuple(strides)
        dilation = tuple(dilations)

        if ndim == 1:
            result = F.max_pool1d(
                x,
                kernel[0],
                stride=stride[0],
                padding=padding if isinstance(padding, int) else padding[0],
                dilation=dilation[0],
                ceil_mode=bool(ceil_mode),
                return_indices=return_indices,
            )
        elif ndim == 2:
            result = F.max_pool2d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=bool(ceil_mode),
                return_indices=return_indices,
            )
        elif ndim == 3:
            result = F.max_pool3d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=bool(ceil_mode),
                return_indices=return_indices,
            )
        else:
            raise NotImplementedError(f"MaxPool{ndim}D not supported")

        if return_indices:
            values, indices = result
            if needs_cast:
                values = values.to(input_dtype)

            # Handle storage_order for indices
            # PyTorch returns row-major indices (last dim varies fastest)
            # ONNX storage_order=0 means row-major (default)
            # ONNX storage_order=1 means column-major (first spatial dim varies fastest)
            if storage_order == 1:
                # Convert row-major indices to column-major
                # For input shape (N, C, D1, D2, ...), we need to convert indices
                # Row-major: idx = n*C*D1*D2*... + c*D1*D2*... + d1*D2*... + d2*... + ...
                # Column-major: idx = n + c*N + d1*N*C + d2*N*C*D1 + ...
                # Compute the multi-index from row-major flat index
                flat_indices = indices
                # Spatial dims of original input (before any padding)
                spatial_dims = list(input_shape[2:])
                n_batch = input_shape[0]
                n_channel = input_shape[1]

                # Decompose row-major index to (n, c, d1, d2, ...)
                remaining = flat_indices
                coords = []
                # First extract spatial coords in reverse order (last dim first)
                for dim_size in reversed(spatial_dims):
                    coords.append(remaining % dim_size)
                    remaining = remaining // dim_size
                # Now remaining = n * C + c
                c_coord = remaining % n_channel
                n_coord = remaining // n_channel

                # Reverse coords to get (d1, d2, ...) order
                spatial_coords = list(reversed(coords))

                # Compute column-major index
                # col_idx = n + c*N + d1*N*C + d2*N*C*D1 + ...
                col_idx = n_coord
                stride_factor = n_batch
                col_idx = col_idx + c_coord * stride_factor
                stride_factor = stride_factor * n_channel
                for i, d_coord in enumerate(spatial_coords):
                    col_idx = col_idx + d_coord * stride_factor
                    stride_factor = stride_factor * spatial_dims[i]

                indices = col_idx

            return values, indices
        else:
            if needs_cast:
                result = result.to(input_dtype)
            return result

    return builder.call_function(
        _max_pool,
        args=(
            x,
            kernel_shape,
            strides,
            pads,
            dilations,
            ceil_mode,
            auto_pad,
            return_indices,
            storage_order,
        ),
    )


@register("MaxUnpool")
def max_unpool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """MaxUnpool - partial inverse of MaxPool.

    Unpools the input tensor using indices from MaxPool.
    """
    x = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    # Optional output_shape input
    output_shape = None
    if len(node.input) > 2 and node.input[2]:
        output_shape = builder.get_value(node.input[2])

    kernel_shape = get_attribute(node, "kernel_shape")
    strides = get_attribute(node, "strides") or [1] * len(kernel_shape)
    pads = get_attribute(node, "pads") or [0] * (2 * len(kernel_shape))

    def _max_unpool(x, indices, kernel_shape, strides, pads, output_shape):
        ndim = len(kernel_shape)

        # Convert ONNX pads format to PyTorch padding
        # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        # PyTorch: symmetric padding per dimension
        n = len(pads) // 2
        padding = tuple(pads[:n])

        kernel = tuple(kernel_shape)
        stride = tuple(strides)

        # Calculate default output size (without explicit output_shape)
        # Default output: out_i = (in_i - 1) * stride_i + kernel_i - 2 * pad_i
        input_spatial_shape = x.shape[2:]
        default_spatial = []
        for i in range(ndim):
            out_dim = (
                (input_spatial_shape[i] - 1) * stride[i]
                + kernel[i]
                - pads[i]
                - pads[i + n]
            )
            default_spatial.append(out_dim)

        # Determine output size
        out_size = None
        if output_shape is not None:
            # output_shape is the full shape including batch and channel dims
            if isinstance(output_shape, torch.Tensor):
                out_size = tuple(int(s) for s in output_shape.tolist())
            else:
                out_size = tuple(int(s) for s in output_shape)

            # Get spatial dimensions from output_shape
            target_spatial = out_size[2:]

            # Check if we need to convert indices
            # ONNX indices are computed for the original (default) tensor size
            # PyTorch expects indices relative to the output_size
            if list(target_spatial) != list(default_spatial):
                # Convert indices from default spatial shape to target spatial shape
                # Indices are flattened over (N, C, D1, D2, ...) dimensions
                # We need to extract (d1, d2, ...) coords from default shape
                # and recompute indices for target shape

                # For efficiency, work with the spatial dimensions only
                # The batch and channel dimensions affect the flat index calculation
                channels = x.shape[1]

                # Compute the total size for default spatial dimensions
                default_spatial_size = 1
                for d in default_spatial:
                    default_spatial_size *= d

                # Decompose flat indices to (n, c, spatial_coords) in default shape
                remaining = indices

                # Extract spatial coordinates in reverse order (last spatial dim first)
                spatial_coords = []
                for dim_size in reversed(default_spatial):
                    spatial_coords.append(remaining % dim_size)
                    remaining = remaining // dim_size
                spatial_coords = list(reversed(spatial_coords))

                # remaining now contains (n * channels + c)
                c_coord = remaining % channels
                n_coord = remaining // channels

                # Recompute flat indices for target spatial shape
                # new_idx = n * (C * prod(target_spatial)) + c * prod(target_spatial) + spatial_flat
                target_spatial_size = 1
                for d in target_spatial:
                    target_spatial_size *= d

                # Compute spatial flat index for target shape
                spatial_flat = spatial_coords[0]
                for i in range(1, ndim):
                    spatial_flat = spatial_flat * target_spatial[i] + spatial_coords[i]

                # Compute full flat index
                indices = (
                    n_coord * (channels * target_spatial_size)
                    + c_coord * target_spatial_size
                    + spatial_flat
                )

        if ndim == 1:
            return F.max_unpool1d(
                x,
                indices,
                kernel[0],
                stride=stride[0],
                padding=padding[0],
                output_size=out_size,
            )
        elif ndim == 2:
            return F.max_unpool2d(
                x,
                indices,
                kernel,
                stride=stride,
                padding=padding,
                output_size=out_size,
            )
        elif ndim == 3:
            return F.max_unpool3d(
                x,
                indices,
                kernel,
                stride=stride,
                padding=padding,
                output_size=out_size,
            )
        else:
            raise NotImplementedError(f"MaxUnpool{ndim}D not supported")

    return builder.call_function(
        _max_unpool,
        args=(x, indices, kernel_shape, strides, pads, output_shape),
    )


@register("AveragePool")
def average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Average pooling."""
    x = builder.get_value(node.input[0])

    kernel_shape = get_attribute(node, "kernel_shape")
    strides = get_attribute(node, "strides") or [1] * len(kernel_shape)
    pads = get_attribute(node, "pads")
    dilations = get_attribute(node, "dilations") or [1] * len(kernel_shape)
    ceil_mode = get_attribute(node, "ceil_mode", 0)
    count_include_pad = get_attribute(node, "count_include_pad", 0)
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")

    def _avg_pool_dilated(
        x, kernel_shape, strides, dilations, pads, ceil_mode, count_include_pad
    ):
        """Compute average pooling with dilation support using unfold.

        PyTorch's avg_pool doesn't support dilation, so we implement it manually.
        """
        ndim = len(kernel_shape)
        batch_size = x.shape[0]
        channels = x.shape[1]
        spatial_shape = list(x.shape[2:])

        # Compute effective kernel size with dilation
        # effective_k = (k - 1) * d + 1
        effective_kernel = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]

        # Apply padding if specified
        if pads is not None:
            n = len(pads) // 2
            pads_begin = [pads[i] for i in range(n)]
            pads_end = [pads[i + n] for i in range(n)]
        else:
            n = ndim
            pads_begin = [0] * n
            pads_end = [0] * n

        # Track original pads (before ceil_mode adjustment) for count_include_pad
        orig_pads_end = pads_end.copy()

        # For ceil_mode, add extra end padding if needed to get ceil behavior
        # ceil_mode output: ceil((input + pad_begin + pad_end - ek) / stride) + 1
        # floor_mode output: floor((input + pad_begin + pad_end - ek) / stride) + 1
        # To get ceil behavior with floor, add padding: (stride - 1)
        ceil_extra_pad = [0] * ndim
        if ceil_mode:
            for i in range(ndim):
                padded_size = spatial_shape[i] + pads_begin[i] + pads_end[i]
                # Compute output with floor
                out_floor = (padded_size - effective_kernel[i]) // strides[i] + 1
                # Compute output with ceil
                out_ceil = (
                    padded_size - effective_kernel[i] + strides[i] - 1
                ) // strides[i] + 1
                if out_ceil > out_floor:
                    # Need extra padding to get one more output element
                    ceil_extra_pad[i] = strides[i] - 1
                    pads_end[i] += ceil_extra_pad[i]

        # Build pad_list for F.pad (reversed order: last dim first)
        pad_list = []
        for i in range(ndim - 1, -1, -1):
            pad_list.extend([pads_begin[i], pads_end[i]])

        has_padding = any(p > 0 for p in pad_list)
        has_ceil_extra = any(p > 0 for p in ceil_extra_pad)

        if has_padding:
            x = F.pad(x, pad_list, value=0)
            spatial_shape_padded = list(x.shape[2:])

            # Create a mask for computing the correct count
            # Case 1: count_include_pad=False -> mask marks original (non-padded) area
            # Case 2: count_include_pad=True with ceil_extra_pad -> mask marks area
            #         up to original pads (but not ceil extra pads)
            # Case 3: count_include_pad=True without ceil_extra_pad -> no mask needed
            if not count_include_pad:
                # Original shape before any padding
                orig_shape = [batch_size, channels] + [
                    spatial_shape_padded[i] - pads_begin[i] - pads_end[i]
                    for i in range(ndim)
                ]
                mask = torch.ones(orig_shape, dtype=x.dtype, device=x.device)
                mask = F.pad(mask, pad_list, value=0)
            elif has_ceil_extra:
                # count_include_pad=True but with ceil extra padding
                # Create mask that includes original padding but not ceil extra
                orig_pad_list = []
                for i in range(ndim - 1, -1, -1):
                    orig_pad_list.extend([pads_begin[i], orig_pads_end[i]])
                # Shape after original padding only
                orig_padded_shape = [batch_size, channels] + [
                    spatial_shape[i] + pads_begin[i] + orig_pads_end[i]
                    for i in range(ndim)
                ]
                mask = torch.ones(orig_padded_shape, dtype=x.dtype, device=x.device)
                # Pad with ceil extra padding (but these should be 0 in mask)
                ceil_pad_list = []
                for i in range(ndim - 1, -1, -1):
                    ceil_pad_list.extend([0, ceil_extra_pad[i]])
                mask = F.pad(mask, ceil_pad_list, value=0)
            else:
                mask = None
        else:
            mask = None

        # Use unfold to extract patches with dilation
        # For each spatial dimension, unfold with size=kernel and step=stride
        # We need to account for dilation by selecting every d-th element

        if ndim == 1:
            # Use unfold for 1D
            # unfold(dimension, size, step)
            _, d, s = kernel_shape[0], dilations[0], strides[0]
            ek = effective_kernel[0]

            # Unfold with effective kernel size and stride
            # Then select every d-th element within each patch
            patches = x.unfold(2, ek, s)  # (N, C, out_L, ek)
            # Select dilated elements: indices 0, d, 2d, ..., (k-1)*d
            indices = torch.arange(0, ek, d, device=x.device)
            patches = patches.index_select(-1, indices)  # (N, C, out_L, k)

            if mask is not None:
                mask_patches = mask.unfold(2, ek, s)
                mask_patches = mask_patches.index_select(-1, indices)
                count = mask_patches.sum(dim=-1)
                sum_val = patches.sum(dim=-1)
                return sum_val / count.clamp(min=1)
            else:
                return patches.mean(dim=-1)

        elif ndim == 2:
            k0, k1 = kernel_shape
            d0, d1 = dilations
            s0, s1 = strides
            ek0, ek1 = effective_kernel

            # Unfold along height (dim 2), then width (dim 3)
            patches = x.unfold(2, ek0, s0).unfold(3, ek1, s1)
            # patches shape: (N, C, out_H, out_W, ek0, ek1)

            # Select dilated elements
            indices0 = torch.arange(0, ek0, d0, device=x.device)
            indices1 = torch.arange(0, ek1, d1, device=x.device)
            patches = patches.index_select(-2, indices0).index_select(-1, indices1)
            # patches shape: (N, C, out_H, out_W, k0, k1)

            if mask is not None:
                mask_patches = mask.unfold(2, ek0, s0).unfold(3, ek1, s1)
                mask_patches = mask_patches.index_select(-2, indices0).index_select(
                    -1, indices1
                )
                count = mask_patches.sum(dim=(-2, -1))
                sum_val = patches.sum(dim=(-2, -1))
                return sum_val / count.clamp(min=1)
            else:
                return patches.mean(dim=(-2, -1))

        elif ndim == 3:
            k0, k1, k2 = kernel_shape
            d0, d1, d2 = dilations
            s0, s1, s2 = strides
            ek0, ek1, ek2 = effective_kernel

            # Unfold along each spatial dimension
            patches = x.unfold(2, ek0, s0).unfold(3, ek1, s1).unfold(4, ek2, s2)
            # patches shape: (N, C, out_D, out_H, out_W, ek0, ek1, ek2)

            # Select dilated elements
            indices0 = torch.arange(0, ek0, d0, device=x.device)
            indices1 = torch.arange(0, ek1, d1, device=x.device)
            indices2 = torch.arange(0, ek2, d2, device=x.device)
            patches = (
                patches.index_select(-3, indices0)
                .index_select(-2, indices1)
                .index_select(-1, indices2)
            )
            # patches shape: (N, C, out_D, out_H, out_W, k0, k1, k2)

            if mask is not None:
                mask_patches = (
                    mask.unfold(2, ek0, s0).unfold(3, ek1, s1).unfold(4, ek2, s2)
                )
                mask_patches = (
                    mask_patches.index_select(-3, indices0)
                    .index_select(-2, indices1)
                    .index_select(-1, indices2)
                )
                count = mask_patches.sum(dim=(-3, -2, -1))
                sum_val = patches.sum(dim=(-3, -2, -1))
                return sum_val / count.clamp(min=1)
            else:
                return patches.mean(dim=(-3, -2, -1))

        else:
            raise NotImplementedError(f"AveragePool{ndim}D not supported")

    def _avg_pool(
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        count_include_pad,
        auto_pad,
    ):
        ndim = len(kernel_shape)

        # Check if we have non-trivial dilation
        has_dilation = any(d != 1 for d in dilations)

        # Handle auto_pad first (before explicit pads)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # For SAME padding with count_include_pad=0, we need to compute
            # the average only over valid (non-padded) input positions.
            # We do this by:
            # 1. Sum pooling on padded input (pad with 0s, so they don't affect sum)
            # 2. Count pooling on a mask (to count valid positions per output)
            # 3. Divide sum by count
            input_shape = x.shape[2:]
            output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]
            # Compute effective kernel size with dilation
            effective_kernel = [
                (k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)
            ]
            pad_total = [
                max(0, (o - 1) * st + ek - i)
                for i, o, ek, st in zip(
                    input_shape,
                    output_shape,
                    effective_kernel,
                    strides,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p // 2, p - p // 2])
            else:
                pad_list = []
                for p in reversed(pad_total):
                    pad_list.extend([p - p // 2, p // 2])

            # Convert pad_list to pads format for dilated implementation
            n = ndim
            pads_onnx = [0] * (2 * n)
            for i in range(n):
                pads_onnx[i] = pad_list[2 * (n - 1 - i)]
                pads_onnx[i + n] = pad_list[2 * (n - 1 - i) + 1]

            # Use dilated implementation which handles padding correctly
            return _avg_pool_dilated(
                x, kernel_shape, strides, dilations, pads_onnx, ceil_mode, 0
            )

        # If we have dilation, use the dilated implementation
        if has_dilation:
            return _avg_pool_dilated(
                x, kernel_shape, strides, dilations, pads, ceil_mode, count_include_pad
            )

        # Check if we need to use manual padding (asymmetric or exceeds limit)
        padding = 0
        use_manual_pad = False
        if pads is not None:
            n = len(pads) // 2
            symmetric = all(pads[i] == pads[i + n] for i in range(n))

            # Check if padding exceeds PyTorch's limit
            # PyTorch: pad should be at most half of kernel size
            max_allowed_pad = [k // 2 for k in kernel_shape]
            exceeds_limit = any(
                pads[i] > max_allowed_pad[i] or pads[i + n] > max_allowed_pad[i]
                for i in range(n)
            )

            if symmetric and not exceeds_limit:
                padding = tuple(pads[:n])
            else:
                use_manual_pad = True

        if use_manual_pad:
            # Use dilated implementation which handles asymmetric/large padding
            return _avg_pool_dilated(
                x, kernel_shape, strides, dilations, pads, ceil_mode, count_include_pad
            )

        kernel = tuple(kernel_shape)
        stride = tuple(strides)

        if ndim == 1:
            return F.avg_pool1d(
                x,
                kernel[0],
                stride=stride[0],
                padding=padding if isinstance(padding, int) else padding[0],
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        elif ndim == 2:
            return F.avg_pool2d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        elif ndim == 3:
            return F.avg_pool3d(
                x,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=bool(ceil_mode),
                count_include_pad=bool(count_include_pad),
            )
        else:
            raise NotImplementedError(f"AveragePool{ndim}D not supported")

    return builder.call_function(
        _avg_pool,
        args=(
            x,
            kernel_shape,
            strides,
            pads,
            dilations,
            ceil_mode,
            count_include_pad,
            auto_pad,
        ),
    )


@register("GlobalAveragePool")
def global_average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Global average pooling."""
    x = builder.get_value(node.input[0])

    def _global_avg_pool(x):
        # Average over all spatial dimensions (keep batch and channel)
        dims = tuple(range(2, x.dim()))
        return x.mean(dim=dims, keepdim=True)

    return builder.call_function(_global_avg_pool, args=(x,))


@register("GlobalMaxPool")
def global_max_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Global max pooling."""
    x = builder.get_value(node.input[0])

    def _global_max_pool(x):
        # Max over all spatial dimensions (keep batch and channel)
        result = x
        for dim in range(x.dim() - 1, 1, -1):
            result = result.max(dim=dim, keepdim=True).values
        return result

    return builder.call_function(_global_max_pool, args=(x,))


@register("LpPool")
def lp_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Lp pooling.

    Computes the Lp norm over a sliding window:
    output = (sum(|x|^p))^(1/p)
    """
    x = builder.get_value(node.input[0])

    kernel_shape = get_attribute(node, "kernel_shape")
    strides = get_attribute(node, "strides") or [1] * len(kernel_shape)
    pads = get_attribute(node, "pads")
    dilations = get_attribute(node, "dilations") or [1] * len(kernel_shape)
    ceil_mode = get_attribute(node, "ceil_mode", 0)
    auto_pad = get_attribute(node, "auto_pad", "NOTSET")
    p = get_attribute(node, "p", 2)

    def _lp_pool_dilated(x, kernel_shape, strides, dilations, pads, ceil_mode, p):
        """Compute Lp pooling with dilation support using unfold.

        PyTorch's lp_pool doesn't support dilation or padding, so we implement
        it manually.
        """
        ndim = len(kernel_shape)
        spatial_shape = list(x.shape[2:])

        # Compute effective kernel size with dilation
        # effective_k = (k - 1) * d + 1
        effective_kernel = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]

        # Apply padding if specified
        if pads is not None:
            n = len(pads) // 2
            pads_begin = [pads[i] for i in range(n)]
            pads_end = [pads[i + n] for i in range(n)]
        else:
            n = ndim
            pads_begin = [0] * n
            pads_end = [0] * n

        # For ceil_mode, add extra end padding if needed to get ceil behavior
        if ceil_mode:
            for i in range(ndim):
                padded_size = spatial_shape[i] + pads_begin[i] + pads_end[i]
                # Compute output with floor
                out_floor = (padded_size - effective_kernel[i]) // strides[i] + 1
                # Compute output with ceil
                out_ceil = (
                    padded_size - effective_kernel[i] + strides[i] - 1
                ) // strides[i] + 1
                if out_ceil > out_floor:
                    # Need extra padding to get one more output element
                    pads_end[i] += strides[i] - 1

        # Build pad_list for F.pad (reversed order: last dim first)
        pad_list = []
        for i in range(ndim - 1, -1, -1):
            pad_list.extend([pads_begin[i], pads_end[i]])

        has_padding = any(p_val > 0 for p_val in pad_list)

        if has_padding:
            x = F.pad(x, pad_list, value=0)

        # Use unfold to extract patches with dilation
        if ndim == 1:
            _, d, s = kernel_shape[0], dilations[0], strides[0]
            ek = effective_kernel[0]

            # Unfold with effective kernel size and stride
            patches = x.unfold(2, ek, s)  # (N, C, out_L, ek)
            # Select dilated elements: indices 0, d, 2d, ..., (k-1)*d
            indices = torch.arange(0, ek, d, device=x.device)
            patches = patches.index_select(-1, indices)  # (N, C, out_L, k)

            # Compute Lp norm: (sum(|x|^p))^(1/p)
            return (patches.abs().pow(p).sum(dim=-1)).pow(1.0 / p)

        elif ndim == 2:
            k0, k1 = kernel_shape
            d0, d1 = dilations
            s0, s1 = strides
            ek0, ek1 = effective_kernel

            # Unfold along height (dim 2), then width (dim 3)
            patches = x.unfold(2, ek0, s0).unfold(3, ek1, s1)
            # patches shape: (N, C, out_H, out_W, ek0, ek1)

            # Select dilated elements
            indices0 = torch.arange(0, ek0, d0, device=x.device)
            indices1 = torch.arange(0, ek1, d1, device=x.device)
            patches = patches.index_select(-2, indices0).index_select(-1, indices1)
            # patches shape: (N, C, out_H, out_W, k0, k1)

            # Compute Lp norm: (sum(|x|^p))^(1/p)
            return (patches.abs().pow(p).sum(dim=(-2, -1))).pow(1.0 / p)

        elif ndim == 3:
            k0, k1, k2 = kernel_shape
            d0, d1, d2 = dilations
            s0, s1, s2 = strides
            ek0, ek1, ek2 = effective_kernel

            # Unfold along each spatial dimension
            patches = x.unfold(2, ek0, s0).unfold(3, ek1, s1).unfold(4, ek2, s2)
            # patches shape: (N, C, out_D, out_H, out_W, ek0, ek1, ek2)

            # Select dilated elements
            indices0 = torch.arange(0, ek0, d0, device=x.device)
            indices1 = torch.arange(0, ek1, d1, device=x.device)
            indices2 = torch.arange(0, ek2, d2, device=x.device)
            patches = (
                patches.index_select(-3, indices0)
                .index_select(-2, indices1)
                .index_select(-1, indices2)
            )
            # patches shape: (N, C, out_D, out_H, out_W, k0, k1, k2)

            # Compute Lp norm: (sum(|x|^p))^(1/p)
            return (patches.abs().pow(p).sum(dim=(-3, -2, -1))).pow(1.0 / p)

        else:
            raise NotImplementedError(f"LpPool{ndim}D not supported")

    def _lp_pool(x, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad, p):
        ndim = len(kernel_shape)

        # Check if we have non-trivial dilation
        has_dilation = any(d != 1 for d in dilations)

        # Handle auto_pad first (before explicit pads)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            input_shape = x.shape[2:]
            output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]
            # Compute effective kernel size with dilation
            effective_kernel = [
                (k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)
            ]
            pad_total = [
                max(0, (o - 1) * st + ek - i)
                for i, o, ek, st in zip(
                    input_shape,
                    output_shape,
                    effective_kernel,
                    strides,
                )
            ]
            if auto_pad == "SAME_UPPER":
                pad_list = []
                for p_total in reversed(pad_total):
                    pad_list.extend([p_total // 2, p_total - p_total // 2])
            else:
                pad_list = []
                for p_total in reversed(pad_total):
                    pad_list.extend([p_total - p_total // 2, p_total // 2])

            # Convert pad_list to pads format for dilated implementation
            n = ndim
            pads_onnx = [0] * (2 * n)
            for i in range(n):
                pads_onnx[i] = pad_list[2 * (n - 1 - i)]
                pads_onnx[i + n] = pad_list[2 * (n - 1 - i) + 1]

            # Use dilated implementation which handles padding correctly
            return _lp_pool_dilated(
                x, kernel_shape, strides, dilations, pads_onnx, ceil_mode, p
            )

        # If we have dilation, use the dilated implementation
        if has_dilation:
            return _lp_pool_dilated(
                x, kernel_shape, strides, dilations, pads, ceil_mode, p
            )

        # Check if we need to use manual padding (asymmetric or any padding)
        # PyTorch's lp_pool doesn't support padding at all
        if pads is not None and any(pad_val > 0 for pad_val in pads):
            return _lp_pool_dilated(
                x, kernel_shape, strides, dilations, pads, ceil_mode, p
            )

        # PyTorch's lp_pool functions use sign(f(x)) * |f(x)|^(1/p) where f(x) = sum(x^p),
        # but ONNX's LpPool uses (sum(|x|^p))^(1/p). The difference is that ONNX
        # takes absolute value FIRST before raising to power p. This matters when
        # x contains negative values and p is odd (like p=3), as PyTorch's version
        # can produce NaN while ONNX's version is always well-defined.
        # Therefore, we always use our manual implementation which correctly applies abs() first.
        return _lp_pool_dilated(
            x, kernel_shape, strides, dilations, pads, ceil_mode, p
        )

    return builder.call_function(
        _lp_pool,
        args=(
            x,
            kernel_shape,
            strides,
            pads,
            dilations,
            ceil_mode,
            auto_pad,
            p,
        ),
    )


@register("GlobalLpPool")
def global_lp_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Global Lp pooling.

    Computes the Lp norm over all spatial dimensions.
    """
    x = builder.get_value(node.input[0])
    p = get_attribute(node, "p", 2)

    def _global_lp_pool(x, p):
        # Lp norm over all spatial dimensions (keep batch and channel)
        dims = tuple(range(2, x.dim()))
        return (x.abs().pow(p).sum(dim=dims, keepdim=True)).pow(1.0 / p)

    return builder.call_function(_global_lp_pool, args=(x, p))


# =============================================================================
# Normalization operators
# =============================================================================


@register("LpNormalization")
def lp_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Lp Normalization.

    Normalizes input element-wise by dividing by the Lp norm along the specified axis.

    Attributes:
        axis: The axis on which to apply normalization (default: -1)
        p: The order of the normalization, only 1 or 2 are supported (default: 2)
    """
    x = builder.get_value(node.input[0])

    axis = get_attribute(node, "axis", -1)
    p = get_attribute(node, "p", 2)

    def _lp_normalize(x, axis, p):
        if p == 1:
            # L1 normalization: x / sum(|x|)
            norm = torch.sum(torch.abs(x), dim=axis, keepdim=True)
            # Avoid division by zero
            norm = torch.clamp(norm, min=1e-12)
            return x / norm
        elif p == 2:
            # L2 normalization: x / sqrt(sum(x^2))
            # Note: We don't use F.normalize because it returns 0 for zero vectors,
            # but ONNX expects NaN (0/0 behavior)
            norm = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True))
            return x / norm
        else:
            raise ValueError(f"LpNormalization only supports p=1 or p=2, got p={p}")

    return builder.call_function(_lp_normalize, args=(x, axis, p))


@register("BatchNormalization")
def batch_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Batch normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])
    mean = builder.get_value(node.input[3])
    var = builder.get_value(node.input[4])

    epsilon = get_attribute(node, "epsilon", 1e-5)
    # Note: ONNX momentum attribute is not used in inference mode
    training_mode = get_attribute(node, "training_mode", 0)

    def _batch_norm(x, scale, bias, mean, var, epsilon, training_mode):
        return F.batch_norm(
            x,
            mean,
            var,
            weight=scale,
            bias=bias,
            training=bool(training_mode),
            eps=epsilon,
        )

    return builder.call_function(
        _batch_norm, args=(x, scale, bias, mean, var, epsilon, training_mode)
    )


@register("LayerNormalization")
def layer_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Layer normalization.

    ONNX LayerNormalization returns up to 3 outputs:
    - Y: normalized output (required)
    - Mean: mean values (optional)
    - InvStdDev: inverse standard deviation (optional)
    """
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])

    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = builder.get_value(node.input[2])

    axis = get_attribute(node, "axis", -1)
    epsilon = get_attribute(node, "epsilon", 1e-5)
    stash_type = get_attribute(node, "stash_type", 1)  # 1 = float32

    # Check how many outputs are requested
    num_outputs = len([o for o in node.output if o])

    def _layer_norm_single(x, scale, bias, axis, epsilon):
        # Compute normalized shape from axis
        if axis < 0:
            axis = x.dim() + axis
        normalized_shape = x.shape[axis:]
        return F.layer_norm(x, normalized_shape, weight=scale, bias=bias, eps=epsilon)

    def _layer_norm_with_stats(x, scale, bias, axis, epsilon, stash_type):
        # Compute normalized shape from axis
        if axis < 0:
            axis = x.dim() + axis

        # Determine stash dtype for mean/invstddev computation
        if stash_type == 1:
            stash_dtype = torch.float32
        elif stash_type == 11:
            stash_dtype = torch.float64
        elif stash_type == 10:
            stash_dtype = torch.float16
        elif stash_type == 16:
            stash_dtype = torch.bfloat16
        else:
            stash_dtype = torch.float32

        # Cast input to stash dtype for computing statistics
        original_dtype = x.dtype
        x_stash = x.to(stash_dtype)

        # Compute mean and variance over the normalized dimensions
        dims = list(range(axis, x.dim()))
        mean = x_stash.mean(dim=dims, keepdim=True)
        var = x_stash.var(dim=dims, unbiased=False, keepdim=True)
        inv_std_dev = 1.0 / torch.sqrt(var + epsilon)

        # Normalize
        x_norm = (x_stash - mean) * inv_std_dev

        # Apply scale and bias
        if scale is not None:
            x_norm = x_norm * scale.to(stash_dtype)
        if bias is not None:
            x_norm = x_norm + bias.to(stash_dtype)

        # Cast back to original dtype
        y = x_norm.to(original_dtype)

        return (y, mean, inv_std_dev)

    if num_outputs == 1:
        return builder.call_function(
            _layer_norm_single, args=(x, scale, bias, axis, epsilon)
        )
    else:
        return builder.call_function(
            _layer_norm_with_stats, args=(x, scale, bias, axis, epsilon, stash_type)
        )


@register("RMSNormalization", since_version=23)
def rms_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """RMS Normalization (Root Mean Square Layer Normalization).

    This is LayerNormalization without mean subtraction, also known as RMSNorm.
    Formula: Y = X / sqrt(mean(X^2) + epsilon) * scale

    Inputs:
        X: Input tensor
        scale: Scale tensor (broadcastable to normalized shape)

    Attributes:
        axis: First normalization dimension (default: -1)
        epsilon: Small constant for numerical stability (default: 1e-5)
        stash_type: Floating-point precision for computation (default: 1 = float32)
    """
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])

    axis = get_attribute(node, "axis", -1)
    epsilon = get_attribute(node, "epsilon", 1e-5)
    stash_type = get_attribute(node, "stash_type", 1)

    def _rms_norm(x, scale, axis, epsilon, stash_type):
        # Determine stash dtype for computation
        if stash_type == 1:
            stash_dtype = torch.float32
        elif stash_type == 11:
            stash_dtype = torch.float64
        elif stash_type == 10:
            stash_dtype = torch.float16
        elif stash_type == 16:
            stash_dtype = torch.bfloat16
        else:
            stash_dtype = torch.float32

        # Normalize axis
        if axis < 0:
            axis_pos = x.dim() + axis
        else:
            axis_pos = axis

        # Save original dtype for casting back
        original_dtype = x.dtype

        # Cast to stash dtype for computation
        x_stash = x.to(stash_dtype)

        # Compute dimensions to reduce over (from axis to end)
        dims = list(range(axis_pos, x.dim()))

        # Compute RMS: sqrt(mean(x^2) + epsilon)
        x_squared = x_stash.pow(2)
        mean_squared = x_squared.mean(dim=dims, keepdim=True)
        rms = torch.sqrt(mean_squared + epsilon)

        # Normalize
        x_normalized = x_stash / rms

        # Apply scale
        scale_stash = scale.to(stash_dtype)
        y = x_normalized * scale_stash

        # Cast back to original dtype
        return y.to(original_dtype)

    return builder.call_function(_rms_norm, args=(x, scale, axis, epsilon, stash_type))


@register("InstanceNormalization")
def instance_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Instance normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _instance_norm(x, scale, bias, epsilon):
        return F.instance_norm(x, weight=scale, bias=bias, eps=epsilon)

    return builder.call_function(_instance_norm, args=(x, scale, bias, epsilon))


@register("GroupNormalization")
def group_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Group normalization."""
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])
    bias = builder.get_value(node.input[2])

    epsilon = get_attribute(node, "epsilon", 1e-5)
    num_groups = get_attribute(node, "num_groups")

    def _group_norm(x, scale, bias, num_groups, epsilon):
        return F.group_norm(x, num_groups, weight=scale, bias=bias, eps=epsilon)

    return builder.call_function(
        _group_norm, args=(x, scale, bias, num_groups, epsilon)
    )


@register("LRN")
def lrn(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Local Response Normalization."""
    x = builder.get_value(node.input[0])

    alpha = get_attribute(node, "alpha", 0.0001)
    beta = get_attribute(node, "beta", 0.75)
    bias = get_attribute(node, "bias", 1.0)
    size = get_attribute(node, "size")

    return builder.call_function(
        F.local_response_norm,
        args=(x, size),
        kwargs={"alpha": alpha, "beta": beta, "k": bias},
    )


@register("MeanVarianceNormalization")
def mean_variance_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Mean Variance Normalization.

    Performs normalization using formula: (X - E[X]) / sqrt(E[(X - E[X])^2])
    Default axes are [0, 2, 3] for NCHW format (normalize across N, H, W).
    """
    x = builder.get_value(node.input[0])
    axes = get_attribute(node, "axes", [0, 2, 3])

    def _mvn(x, axes):
        axes = tuple(axes)
        eps = 1e-9
        mean = x.mean(dim=axes, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=axes, keepdim=True)
        std = torch.sqrt(variance + eps)
        return (x - mean) / std

    return builder.call_function(_mvn, args=(x, axes))


# =============================================================================
# Dropout and regularization
# =============================================================================


@register("Dropout")
def dropout(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Dropout (inference mode - identity).

    ONNX Dropout can have 2 outputs:
    - output: The result after dropout (same as input in inference mode)
    - mask (optional): Boolean mask indicating which elements were kept (all True in inference mode)
    """
    x = builder.get_value(node.input[0])

    # Check if mask output is requested (second output)
    return_mask = len(node.output) > 1 and node.output[1] != ""

    # In inference mode, dropout is identity
    # ratio = get_attribute(node, "ratio", 0.5)
    # training_mode from input or default to False

    def _dropout_with_mask(x):
        # In inference mode, output is identity and mask is all True
        output = x
        mask = torch.ones_like(x, dtype=torch.bool)
        return output, mask

    if return_mask:
        return builder.call_function(_dropout_with_mask, args=(x,))
    else:
        # For inference without mask, just return input
        return builder.call_function(lambda t: t, args=(x,))


# =============================================================================
# Recurrent neural network operators
# =============================================================================


@register("LSTM")
def lstm(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """LSTM (Long Short-Term Memory) operator.

    Computes an one-layer LSTM.

    ONNX LSTM Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, 4*hidden_size, input_size]
    - R: recurrence weight [num_directions, 4*hidden_size, hidden_size]
    - B (optional): bias [num_directions, 8*hidden_size]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]
    - initial_c (optional): [num_directions, batch_size, hidden_size]
    - P (optional): peephole weights [num_directions, 3*hidden_size]

    ONNX LSTM Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]
    - Y_c (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    - Ct = ft (.) Ct-1 + it (.) ct
    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    - Ht = ot (.) h(Ct)
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = None
    if len(node.input) > 3 and node.input[3]:
        b = builder.get_value(node.input[3])

    sequence_lens = None
    if len(node.input) > 4 and node.input[4]:
        sequence_lens = builder.get_value(node.input[4])

    initial_h = None
    if len(node.input) > 5 and node.input[5]:
        initial_h = builder.get_value(node.input[5])

    initial_c = None
    if len(node.input) > 6 and node.input[6]:
        initial_c = builder.get_value(node.input[6])

    peepholes = None
    if len(node.input) > 7 and node.input[7]:
        peepholes = builder.get_value(node.input[7])

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    input_forget = get_attribute(node, "input_forget", 0)
    # activations = get_attribute(node, "activations", ["Sigmoid", "Tanh", "Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""
    output_y_c = len(node.output) > 2 and node.output[2] != ""

    def _lstm_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        initial_c,
        peepholes,
        hidden_size,
        direction,
        layout,
        input_forget,
        output_y,
        output_y_h,
        output_y_c,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Initialize cell state if not provided
        if initial_c is None:
            initial_c = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []
        all_y_c = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, 4*hidden_size, input_size]
            # ONNX order: [Wi, Wo, Wf, Wc] concatenated (input, output, forget, cell)
            w_dir = w[dir_idx]  # [4*hidden_size, input_size]
            w_i = w_dir[0:hidden_size, :]  # [hidden_size, input_size]
            w_o = w_dir[hidden_size : 2 * hidden_size, :]
            w_f = w_dir[2 * hidden_size : 3 * hidden_size, :]
            w_c = w_dir[3 * hidden_size : 4 * hidden_size, :]

            # R shape: [num_directions, 4*hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [4*hidden_size, hidden_size]
            r_i = r_dir[0:hidden_size, :]  # [hidden_size, hidden_size]
            r_o = r_dir[hidden_size : 2 * hidden_size, :]
            r_f = r_dir[2 * hidden_size : 3 * hidden_size, :]
            r_c = r_dir[3 * hidden_size : 4 * hidden_size, :]

            # Biases (optional)
            # B shape: [num_directions, 8*hidden_size]
            # = [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
            if b is not None:
                b_dir = b[dir_idx]  # [8*hidden_size]
                wb_i = b_dir[0:hidden_size]
                wb_o = b_dir[hidden_size : 2 * hidden_size]
                wb_f = b_dir[2 * hidden_size : 3 * hidden_size]
                wb_c = b_dir[3 * hidden_size : 4 * hidden_size]
                rb_i = b_dir[4 * hidden_size : 5 * hidden_size]
                rb_o = b_dir[5 * hidden_size : 6 * hidden_size]
                rb_f = b_dir[6 * hidden_size : 7 * hidden_size]
                rb_c = b_dir[7 * hidden_size : 8 * hidden_size]
            else:
                wb_i = wb_o = wb_f = wb_c = rb_i = rb_o = rb_f = rb_c = 0

            # Peepholes (optional)
            # P shape: [num_directions, 3*hidden_size] = [Pi, Po, Pf]
            if peepholes is not None:
                p_dir = peepholes[dir_idx]  # [3*hidden_size]
                p_i = p_dir[0:hidden_size]
                p_o = p_dir[hidden_size : 2 * hidden_size]
                p_f = p_dir[2 * hidden_size : 3 * hidden_size]
            else:
                p_i = p_o = p_f = 0

            # Initial hidden state and cell state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]
            c_t = initial_c[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute gates
                # it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
                i_t = torch.sigmoid(x_t @ w_i.T + h_t @ r_i.T + p_i * c_t + wb_i + rb_i)

                # ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
                f_t = torch.sigmoid(x_t @ w_f.T + h_t @ r_f.T + p_f * c_t + wb_f + rb_f)

                # Handle input_forget (coupled input-forget gate)
                if input_forget:
                    f_t = 1 - i_t

                # ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                c_tilde = torch.tanh(x_t @ w_c.T + h_t @ r_c.T + wb_c + rb_c)

                # Ct = ft (.) Ct-1 + it (.) ct
                c_t = f_t * c_t + i_t * c_tilde

                # ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
                o_t = torch.sigmoid(x_t @ w_o.T + h_t @ r_o.T + p_o * c_t + wb_o + rb_o)

                # Ht = ot (.) tanh(Ct)
                h_t = o_t * torch.tanh(c_t)

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)
            all_y_c.append(c_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Y_c: [num_directions, batch_size, hidden_size]
        y_c = torch.stack(all_y_c, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)
            # Convert Y_c from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_c = y_c.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h and output_y_c:
            return (y, y_h, y_c)
        elif output_y and output_y_h:
            return (y, y_h)
        elif output_y and output_y_c:
            return (y, y_c)
        elif output_y_h and output_y_c:
            return (y_h, y_c)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        elif output_y_c:
            return y_c
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _lstm_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            initial_c,
            peepholes,
            hidden_size,
            direction,
            layout,
            input_forget,
            output_y,
            output_y_h,
            output_y_c,
        ),
    )


@register("GRU")
def gru(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """GRU (Gated Recurrent Unit) operator.

    Computes an one-layer GRU.

    ONNX GRU Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, 3*hidden_size, input_size]
    - R: recurrence weight [num_directions, 3*hidden_size, hidden_size]
    - B (optional): bias [num_directions, 6*hidden_size]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]

    ONNX GRU Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Sigmoid, g=Tanh):
    - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # linear_before_reset=0
    - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)  # linear_before_reset!=0
    - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = None
    if len(node.input) > 3 and node.input[3]:
        b = builder.get_value(node.input[3])

    sequence_lens = None
    if len(node.input) > 4 and node.input[4]:
        sequence_lens = builder.get_value(node.input[4])

    initial_h = None
    if len(node.input) > 5 and node.input[5]:
        initial_h = builder.get_value(node.input[5])

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    linear_before_reset = get_attribute(node, "linear_before_reset", 0)
    # activations = get_attribute(node, "activations", ["Sigmoid", "Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""

    def _gru_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        layout,
        linear_before_reset,
        output_y,
        output_y_h,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, 3*hidden_size, input_size]
            # ONNX order: [Wz, Wr, Wh] concatenated
            w_dir = w[dir_idx]  # [3*hidden_size, input_size]
            w_z = w_dir[0:hidden_size, :]  # [hidden_size, input_size]
            w_r = w_dir[hidden_size : 2 * hidden_size, :]
            w_h = w_dir[2 * hidden_size : 3 * hidden_size, :]

            # R shape: [num_directions, 3*hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [3*hidden_size, hidden_size]
            r_z = r_dir[0:hidden_size, :]  # [hidden_size, hidden_size]
            r_r = r_dir[hidden_size : 2 * hidden_size, :]
            r_h = r_dir[2 * hidden_size : 3 * hidden_size, :]

            # Biases (optional)
            # B shape: [num_directions, 6*hidden_size] = [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
            if b is not None:
                b_dir = b[dir_idx]  # [6*hidden_size]
                wb_z = b_dir[0:hidden_size]
                wb_r = b_dir[hidden_size : 2 * hidden_size]
                wb_h = b_dir[2 * hidden_size : 3 * hidden_size]
                rb_z = b_dir[3 * hidden_size : 4 * hidden_size]
                rb_r = b_dir[4 * hidden_size : 5 * hidden_size]
                rb_h = b_dir[5 * hidden_size : 6 * hidden_size]
            else:
                wb_z = wb_r = wb_h = rb_z = rb_r = rb_h = 0

            # Initial hidden state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute gates
                # zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
                z_t = torch.sigmoid(
                    x_t @ w_z.T + h_t @ r_z.T + wb_z + rb_z
                )  # [batch_size, hidden_size]

                # rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
                r_t = torch.sigmoid(x_t @ w_r.T + h_t @ r_r.T + wb_r + rb_r)

                # ht = tanh(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # linear_before_reset=0
                # ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)  # linear_before_reset!=0
                if linear_before_reset:
                    h_tilde = torch.tanh(
                        x_t @ w_h.T + r_t * (h_t @ r_h.T + rb_h) + wb_h
                    )
                else:
                    h_tilde = torch.tanh(
                        x_t @ w_h.T + (r_t * h_t) @ r_h.T + rb_h + wb_h
                    )

                # Ht = (1 - zt) (.) ht + zt (.) Ht-1
                h_t = (1 - z_t) * h_tilde + z_t * h_t

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h:
            return (y, y_h)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _gru_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            layout,
            linear_before_reset,
            output_y,
            output_y_h,
        ),
    )


@register("RNN")
def rnn(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """RNN (Simple Recurrent Neural Network) operator.

    Computes an one-layer simple RNN.

    ONNX RNN Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, hidden_size, input_size]
    - R: recurrence weight [num_directions, hidden_size, hidden_size]
    - B (optional): bias [num_directions, 2*hidden_size] = [Wbi, Rbi]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]

    ONNX RNN Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Tanh):
    - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = None
    if len(node.input) > 3 and node.input[3]:
        b = builder.get_value(node.input[3])

    sequence_lens = None
    if len(node.input) > 4 and node.input[4]:
        sequence_lens = builder.get_value(node.input[4])

    initial_h = None
    if len(node.input) > 5 and node.input[5]:
        initial_h = builder.get_value(node.input[5])

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    # activations = get_attribute(node, "activations", ["Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""

    def _rnn_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        layout,
        output_y,
        output_y_h,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, hidden_size, input_size]
            w_dir = w[dir_idx]  # [hidden_size, input_size]

            # R shape: [num_directions, hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [hidden_size, hidden_size]

            # Biases (optional)
            # B shape: [num_directions, 2*hidden_size] = [Wbi, Rbi]
            if b is not None:
                b_dir = b[dir_idx]  # [2*hidden_size]
                wb_i = b_dir[0:hidden_size]
                rb_i = b_dir[hidden_size : 2 * hidden_size]
            else:
                wb_i = rb_i = 0

            # Initial hidden state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute: Ht = tanh(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                h_t = torch.tanh(x_t @ w_dir.T + h_t @ r_dir.T + wb_i + rb_i)

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h:
            return (y, y_h)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _rnn_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            layout,
            output_y,
            output_y_h,
        ),
    )
