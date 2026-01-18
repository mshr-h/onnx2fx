# SPDX-License-Identifier: Apache-2.0
"""Image and spatial transformation operators.

This module implements ONNX operators for image resizing and
spatial dimension rearrangement.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Resize operator
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


# =============================================================================
# Upsample operator (deprecated in opset 10, replaced by Resize)
# =============================================================================


@register("Upsample", since_version=7)
def upsample(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Upsample tensor using interpolation.

    Deprecated: This operator is deprecated since opset 10.
    Use Resize operator instead.

    Opset 7-8: scales is an attribute
    Opset 9: scales is an input
    """
    x = builder.get_value(node.input[0])

    # In opset 9, scales is an input; in opset 7-8, it's an attribute
    opset = builder.opset_version
    if opset >= 9 and len(node.input) > 1 and node.input[1]:
        scales = builder.get_value(node.input[1])
    else:
        scales = get_attribute(node, "scales")

    mode = get_attribute(node, "mode", "nearest")

    def _upsample(x, scales, mode):
        import torch.nn.functional as F

        # Map ONNX mode to PyTorch mode
        mode_map = {
            "nearest": "nearest",
            "linear": "bilinear" if x.dim() == 4 else "linear",
            "cubic": "bicubic",
        }
        torch_mode = mode_map.get(mode, "nearest")

        # Use scales to compute output size
        scale_list = scales.tolist() if isinstance(scales, torch.Tensor) else scales
        input_shape = x.shape[2:]
        output_size = [int(s * sc) for s, sc in zip(input_shape, scale_list[2:])]

        kwargs = {"size": output_size, "mode": torch_mode}
        # align_corners is not used for nearest mode
        if torch_mode not in ["nearest", "area"]:
            kwargs["align_corners"] = False

        return F.interpolate(x, **kwargs)

    return builder.call_function(_upsample, args=(x, scales, mode))


# =============================================================================
# Depth/Space rearrangement operators
# =============================================================================


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


# =============================================================================
# Col2Im operator
# =============================================================================


@register("Col2Im", since_version=18)
def col2im(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Rearrange column blocks back into a multidimensional image.

    ONNX Col2Im is the inverse of Im2Col. It combines sliding local blocks
    (columns) back into a larger image tensor.

    Inputs:
        input: [N, C * prod(block_shape), L] - batched column data
        image_shape: spatial dimensions of the output image
        block_shape: shape of the sliding block

    Attributes:
        strides: stride along each spatial axis (default: 1)
        pads: padding for each spatial axis in ONNX format (default: 0)
        dilations: dilation for each spatial axis (default: 1)

    Output:
        [N, C, *image_shape] - reconstructed image
    """
    x = builder.get_value(node.input[0])
    image_shape = builder.get_value(node.input[1])
    block_shape = builder.get_value(node.input[2])

    strides = get_attribute(node, "strides")
    pads = get_attribute(node, "pads")
    dilations = get_attribute(node, "dilations")

    def _col2im(x, image_shape, block_shape, strides, pads, dilations):
        import torch.nn.functional as F
        from functools import reduce
        from itertools import product
        from operator import mul

        # Convert to lists if tensors
        if isinstance(image_shape, torch.Tensor):
            image_shape = image_shape.tolist()
        if isinstance(block_shape, torch.Tensor):
            block_shape = block_shape.tolist()

        n_dims = len(block_shape)

        # Default values
        if strides is None:
            strides = [1] * n_dims
        if pads is None:
            pads = [0] * (2 * n_dims)
        if dilations is None:
            dilations = [1] * n_dims

        # For 2D, use PyTorch's optimized fold
        if n_dims == 2:
            # PyTorch fold uses symmetric padding per dimension
            padding = (pads[0], pads[1])
            return F.fold(
                x,
                output_size=tuple(image_shape),
                kernel_size=tuple(block_shape),
                stride=tuple(strides),
                padding=padding,
                dilation=tuple(dilations),
            )

        # For N-D, implement manually
        N = x.shape[0]
        L = x.shape[2]
        block_size = reduce(mul, block_shape, 1)
        C = x.shape[1] // block_size

        # Reshape input: [N, C * prod(block_shape), L] -> [N, C, *block_shape, L]
        input_reshaped = x.reshape(N, C, *block_shape, L)

        # Initialize output: [N, C, *image_shape]
        output = torch.zeros(N, C, *image_shape, dtype=x.dtype, device=x.device)

        # Compute effective kernel size after dilation
        effective_block = [(b - 1) * d + 1 for b, d in zip(block_shape, dilations)]

        # Compute number of blocks in each dimension
        n_blocks = []
        for i, (img_dim, eff_block, s) in enumerate(
            zip(image_shape, effective_block, strides)
        ):
            p_begin = pads[i]
            p_end = pads[n_dims + i]
            n_block = (img_dim + p_begin + p_end - eff_block) // s + 1
            n_blocks.append(n_block)

        # Iterate over all block positions
        block_indices = list(product(*[range(nb) for nb in n_blocks]))

        for l_idx, block_idx in enumerate(block_indices):
            # Compute starting position for this block
            starts = [
                bi * s - pads[i] for i, (bi, s) in enumerate(zip(block_idx, strides))
            ]

            # For each position in the block
            block_positions = list(product(*[range(b) for b in block_shape]))
            for block_pos in block_positions:
                # Compute actual output position with dilation
                output_pos = [
                    starts[i] + block_pos[i] * dilations[i] for i in range(n_dims)
                ]

                # Check bounds
                valid = all(0 <= output_pos[i] < image_shape[i] for i in range(n_dims))
                if valid:
                    # Get value from input_reshaped: [N, C, *block_shape, L]
                    idx = (slice(None), slice(None)) + tuple(block_pos) + (l_idx,)
                    value = input_reshaped[idx]

                    # Add to output
                    out_idx = (slice(None), slice(None)) + tuple(output_pos)
                    output[out_idx] += value

        return output

    return builder.call_function(
        _col2im, args=(x, image_shape, block_shape, strides, pads, dilations)
    )


# =============================================================================
# CenterCropPad operator
# =============================================================================


# =============================================================================
# GridSample operator
# =============================================================================


@register("GridSample", since_version=16)
def grid_sample(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Sample input using grid of sampling locations.

    Given an input X and a flow-field grid, computes the output Y using X values
    and pixel locations from the grid. This is equivalent to PyTorch's
    torch.nn.functional.grid_sample.

    Inputs:
        X: Input tensor of shape (N, C, H, W) for 4D or (N, C, D, H, W) for 5D
        grid: Grid tensor of shape (N, H_out, W_out, 2) for 4D or
              (N, D_out, H_out, W_out, 3) for 5D

    Attributes:
        align_corners: If 1, extrema (-1 and 1) refer to center of corner pixels.
                      If 0, they refer to corner points. Default: 0
        mode: Interpolation mode - 'linear'/'bilinear' (default), 'nearest',
              'cubic'/'bicubic'. Opset 16 uses bilinear/bicubic, opset 20+ uses
              linear/cubic.
        padding_mode: Padding mode for outside grid values - 'zeros' (default),
                     'border', 'reflection'

    Output:
        Y: Output tensor of shape (N, C, H_out, W_out) or (N, C, D_out, H_out, W_out)
    """
    x = builder.get_value(node.input[0])
    grid = builder.get_value(node.input[1])

    align_corners = get_attribute(node, "align_corners", 0)
    # Handle different mode names across opset versions
    # Opset 16: bilinear (default), nearest, bicubic
    # Opset 20+: linear (default), nearest, cubic
    mode = get_attribute(node, "mode", "linear")
    padding_mode = get_attribute(node, "padding_mode", "zeros")

    def _grid_sample(x, grid, mode, padding_mode, align_corners):
        import torch.nn.functional as F

        # Map ONNX mode names to PyTorch mode names
        # PyTorch expects: 'bilinear', 'nearest', 'bicubic' for 4D input
        # PyTorch expects: 'bilinear', 'nearest' for 5D input (no bicubic)
        mode_map = {
            "linear": "bilinear",
            "bilinear": "bilinear",
            "nearest": "nearest",
            "cubic": "bicubic",
            "bicubic": "bicubic",
        }
        torch_mode = mode_map.get(mode, "bilinear")

        # Convert align_corners from int to bool
        align_corners_bool = bool(align_corners)

        return F.grid_sample(
            x,
            grid,
            mode=torch_mode,
            padding_mode=padding_mode,
            align_corners=align_corners_bool,
        )

    return builder.call_function(
        _grid_sample, args=(x, grid, mode, padding_mode, align_corners)
    )


# =============================================================================
# AffineGrid operator
# =============================================================================


@register("AffineGrid", since_version=20)
def affine_grid(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Generate 2D or 3D flow field (sampling grid) from affine matrices.

    Given a batch of affine matrices theta, generates a grid of sampling
    locations. This is typically used with GridSample to build Spatial
    Transformer Networks.

    Inputs:
        theta: Input batch of affine matrices with shape (N, 2, 3) for 2D
               or (N, 3, 4) for 3D
        size: Target output image size (N, C, H, W) for 2D or (N, C, D, H, W)
              for 3D, as a 1-D tensor

    Attributes:
        align_corners: If 1, consider -1 and 1 to refer to the centers of the
                      corner pixels. If 0, consider -1 and 1 to refer to the
                      outer edge of corner pixels. Default: 0

    Output:
        grid: Output tensor of shape (N, H, W, 2) for 2D sample coordinates
              or (N, D, H, W, 3) for 3D sample coordinates
    """
    theta = builder.get_value(node.input[0])
    size = builder.get_value(node.input[1])

    align_corners = get_attribute(node, "align_corners", 0)

    def _affine_grid(theta, size, align_corners):
        import torch.nn.functional as F

        # Convert size tensor to a list of integers for torch.Size
        if isinstance(size, torch.Tensor):
            size_list = size.tolist()
        else:
            size_list = list(size)
        size_tuple = torch.Size([int(s) for s in size_list])

        # Convert align_corners from int to bool
        align_corners_bool = bool(align_corners)

        return F.affine_grid(theta, size_tuple, align_corners=align_corners_bool)

    return builder.call_function(
        _affine_grid, args=(theta, size, align_corners)
    )


# =============================================================================
# CenterCropPad operator
# =============================================================================


@register("CenterCropPad", since_version=18)
def center_crop_pad(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Center crop or pad an input to given dimensions.

    The crop/pad dimensions can be specified for a subset of the axes.
    Unspecified dimensions will remain unchanged.

    For cropping (input > target): centered window, start position rounded down.
    For padding (input < target): centered padding, extra pixel on right if odd.

    Inputs:
        input_data: Input tensor to crop/pad
        shape: 1-D tensor of target dimensions for specified axes

    Attributes:
        axes: Subset of axes that shape refers to (default: all axes)

    Output:
        Output tensor with specified dimensions
    """
    x = builder.get_value(node.input[0])
    shape = builder.get_value(node.input[1])

    axes = get_attribute(node, "axes")

    def _center_crop_pad(x, shape, axes):
        # Convert shape to list if tensor
        if isinstance(shape, torch.Tensor):
            target_shape = shape.tolist()
        else:
            target_shape = list(shape)

        ndim = x.dim()

        # If axes is not provided, use all axes
        if axes is None:
            axes_list = list(range(ndim))
        else:
            axes_list = list(axes)

        # Normalize negative axes
        axes_list = [(a + ndim) if a < 0 else a for a in axes_list]

        # Build slices for cropping and padding amounts
        result = x
        for i, axis in enumerate(axes_list):
            current_size = result.shape[axis]
            target_size = int(target_shape[i])

            if current_size == target_size:
                # No change needed for this axis
                continue
            elif current_size > target_size:
                # Crop: extract centered window
                # Start position is rounded down (floor division)
                diff = current_size - target_size
                start = diff // 2
                end = start + target_size

                # Build slice for this axis
                slices = [slice(None)] * result.dim()
                slices[axis] = slice(start, end)
                result = result[tuple(slices)]
            else:
                # Pad: add zeros centered
                # Extra pixel goes to the right side
                diff = target_size - current_size
                pad_before = diff // 2
                pad_after = diff - pad_before

                # torch.nn.functional.pad uses reverse order: last dim first
                # and pairs are (before, after) for each dim from last to first
                # We need to construct padding for just this one axis

                # Number of dimensions from the end
                dims_from_end = result.dim() - 1 - axis

                # Build pad tuple: pairs for each dim from last to first
                # We only pad the current axis
                pad = [0] * (2 * result.dim())
                # Index in pad list: dims_from_end * 2 for before, +1 for after
                pad[dims_from_end * 2] = pad_before
                pad[dims_from_end * 2 + 1] = pad_after

                result = torch.nn.functional.pad(result, pad, mode="constant", value=0)

        return result

    return builder.call_function(_center_crop_pad, args=(x, shape, axes))


# =============================================================================
# RoiAlign operator
# =============================================================================


@register("RoiAlign")
def roi_align(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Region of Interest (RoI) Align operator.

    Performs RoI pooling with bilinear interpolation for sub-pixel accuracy.
    """
    x = builder.get_value(node.input[0])
    rois = builder.get_value(node.input[1])
    batch_indices = builder.get_value(node.input[2])

    # Get attributes with defaults
    mode = get_attribute(node, "mode", "avg")
    output_height = get_attribute(node, "output_height", 1)
    output_width = get_attribute(node, "output_width", 1)
    sampling_ratio = get_attribute(node, "sampling_ratio", 0)
    spatial_scale = get_attribute(node, "spatial_scale", 1.0)
    coordinate_transformation_mode = get_attribute(
        node, "coordinate_transformation_mode", "half_pixel"
    )

    # ONNX coordinate_transformation_mode:
    # - "half_pixel": pixel shift by -0.5, corresponds to aligned=True in PyTorch
    # - "output_half_pixel": no pixel shift (legacy), corresponds to aligned=False
    aligned = coordinate_transformation_mode == "half_pixel"

    def _roi_align(
        x,
        rois,
        batch_indices,
        mode,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale,
        aligned,
    ):
        from torchvision.ops import roi_align as tv_roi_align

        # PyTorch expects boxes in format [batch_idx, x1, y1, x2, y2]
        boxes = torch.cat([batch_indices.unsqueeze(1).float(), rois.float()], dim=1)
        output_size = (output_height, output_width)

        if mode == "avg":
            # Use torchvision's roi_align directly for average mode
            # sampling_ratio: ONNX uses 0 for adaptive, PyTorch uses -1
            torch_sampling = sampling_ratio if sampling_ratio > 0 else -1
            return tv_roi_align(
                x,
                boxes,
                output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=torch_sampling,
                aligned=aligned,
            )
        else:
            # Max mode: ONNX defines max pooling differently from standard
            # bilinear interpolation. For each sample point, it takes the
            # max of the 4 weighted corner values (not the sum).
            return _roi_align_max_mode(
                x,
                rois,
                batch_indices,
                output_height,
                output_width,
                sampling_ratio,
                spatial_scale,
                aligned,
            )

    return builder.call_function(
        _roi_align,
        args=(
            x,
            rois,
            batch_indices,
            mode,
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
            aligned,
        ),
    )


def _roi_align_max_mode(
    x,
    rois,
    batch_indices,
    output_height,
    output_width,
    sampling_ratio,
    spatial_scale,
    half_pixel,
):
    """ONNX RoiAlign with max pooling mode.

    For each output bin, samples at grid points and takes the MAX of the
    weighted corner values at each sampling point, then MAX across all
    sampling points in the bin.
    """
    num_rois = rois.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    output = torch.zeros(
        num_rois, channels, output_height, output_width, dtype=x.dtype, device=x.device
    )

    for n in range(num_rois):
        roi_batch_ind = int(batch_indices[n].item())
        roi = rois[n]

        # Apply spatial scale and offset
        offset = 0.5 if half_pixel else 0.0
        roi_start_w = float(roi[0]) * spatial_scale - offset
        roi_start_h = float(roi[1]) * spatial_scale - offset
        roi_end_w = float(roi[2]) * spatial_scale - offset
        roi_end_h = float(roi[3]) * spatial_scale - offset

        roi_width = roi_end_w - roi_start_w
        roi_height = roi_end_h - roi_start_h

        if not half_pixel:
            # Force malformed ROIs to be 1x1
            roi_width = max(roi_width, 1.0)
            roi_height = max(roi_height, 1.0)

        bin_size_h = roi_height / output_height
        bin_size_w = roi_width / output_width

        # Determine sampling grid size
        if sampling_ratio > 0:
            roi_bin_grid_h = sampling_ratio
            roi_bin_grid_w = sampling_ratio
        else:
            roi_bin_grid_h = int(
                torch.ceil(torch.tensor(roi_height / output_height)).item()
            )
            roi_bin_grid_w = int(
                torch.ceil(torch.tensor(roi_width / output_width)).item()
            )
            roi_bin_grid_h = max(1, roi_bin_grid_h)
            roi_bin_grid_w = max(1, roi_bin_grid_w)

        for c in range(channels):
            for ph in range(output_height):
                for pw in range(output_width):
                    output_val = None

                    for iy in range(roi_bin_grid_h):
                        yy = (
                            roi_start_h
                            + ph * bin_size_h
                            + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                        )
                        for ix in range(roi_bin_grid_w):
                            xx = (
                                roi_start_w
                                + pw * bin_size_w
                                + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                            )

                            # Check bounds
                            if yy < -1.0 or yy > height or xx < -1.0 or xx > width:
                                continue

                            y = max(yy, 0.0)
                            xc = max(xx, 0.0)

                            y_low = int(y)
                            x_low = int(xc)

                            if y_low >= height - 1:
                                y_high = y_low = height - 1
                                y = float(y_low)
                            else:
                                y_high = y_low + 1

                            if x_low >= width - 1:
                                x_high = x_low = width - 1
                                xc = float(x_low)
                            else:
                                x_high = x_low + 1

                            ly = y - y_low
                            lx = xc - x_low
                            hy = 1.0 - ly
                            hx = 1.0 - lx

                            # Weights
                            w1 = hy * hx
                            w2 = hy * lx
                            w3 = ly * hx
                            w4 = ly * lx

                            # Get corner values
                            v1 = x[roi_batch_ind, c, y_low, x_low].item()
                            v2 = x[roi_batch_ind, c, y_low, x_high].item()
                            v3 = x[roi_batch_ind, c, y_high, x_low].item()
                            v4 = x[roi_batch_ind, c, y_high, x_high].item()

                            # ONNX max mode: max of weighted corners
                            val = max(w1 * v1, w2 * v2, w3 * v3, w4 * v4)

                            if output_val is None:
                                output_val = val
                            else:
                                output_val = max(output_val, val)

                    if output_val is not None:
                        output[n, c, ph, pw] = output_val

    return output
