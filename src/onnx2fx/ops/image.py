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
        output = torch.zeros(
            N, C, *image_shape, dtype=x.dtype, device=x.device
        )

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
                valid = all(
                    0 <= output_pos[i] < image_shape[i] for i in range(n_dims)
                )
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
