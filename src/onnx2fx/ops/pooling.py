# SPDX-License-Identifier: Apache-2.0
"""Pooling operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import compute_same_padding, get_optional_input, pad_list_to_onnx_pads

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


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
            spatial_shape = x.shape[2:]
            pad_list = compute_same_padding(
                tuple(spatial_shape),
                tuple(kernel_shape),
                tuple(strides),
                tuple(dilations),
                auto_pad,
            )
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
    output_shape = get_optional_input(builder, node, 2)

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
            pad_list = compute_same_padding(
                tuple(input_shape),
                tuple(kernel_shape),
                tuple(strides),
                tuple(dilations),
                auto_pad,
                use_effective_kernel=True,
            )

            # Convert pad_list to pads format for dilated implementation
            pads_onnx = pad_list_to_onnx_pads(pad_list, ndim)

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
            pad_list = compute_same_padding(
                tuple(input_shape),
                tuple(kernel_shape),
                tuple(strides),
                tuple(dilations),
                auto_pad,
                use_effective_kernel=True,
            )

            # Convert pad_list to pads format for dilated implementation
            pads_onnx = pad_list_to_onnx_pads(pad_list, ndim)

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
        return _lp_pool_dilated(x, kernel_shape, strides, dilations, pads, ceil_mode, p)

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
