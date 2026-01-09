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
