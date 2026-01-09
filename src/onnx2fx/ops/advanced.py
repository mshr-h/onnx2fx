# SPDX-License-Identifier: Apache-2.0
"""Advanced operators.

This module implements specialized ONNX operators including
Einsum, matrix determinant, and non-maximum suppression.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Einsum operator
# =============================================================================


@register("Einsum")
def einsum(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Einstein summation."""
    equation = get_attribute(node, "equation")
    inputs = [builder.get_value(name) for name in node.input]
    return builder.call_function(torch.einsum, args=(equation, *inputs))


# =============================================================================
# Matrix determinant
# =============================================================================


@register("Det")
def det(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Matrix determinant."""
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.linalg.det, args=(x,))


# =============================================================================
# Non-maximum suppression
# =============================================================================


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
