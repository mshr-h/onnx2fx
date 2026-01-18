# SPDX-License-Identifier: Apache-2.0
"""Advanced operators.

This module implements specialized ONNX operators including
Einsum, matrix determinant, non-maximum suppression, and STFT.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# STFT (Short-time Fourier Transform)
# =============================================================================


@register("STFT", since_version=17)
def stft(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Short-time Fourier Transform.

    ONNX STFT operator computes the STFT of the input signal.

    Inputs:
        signal: [batch_size, signal_length, 1] for real or [batch_size, signal_length, 2] for complex
        frame_step: scalar, the hop length
        window (optional): 1D window tensor
        frame_length (optional): scalar, the FFT size

    Attributes:
        onesided: int (default 1), whether to return one-sided output

    Output:
        [batch_size, frames, dft_unique_bins, 2] with real and imaginary components
    """
    signal = builder.get_value(node.input[0])
    frame_step = builder.get_value(node.input[1])

    # Optional window input
    window = None
    if len(node.input) > 2 and node.input[2]:
        window = builder.get_value(node.input[2])

    # Optional frame_length input
    frame_length = None
    if len(node.input) > 3 and node.input[3]:
        frame_length = builder.get_value(node.input[3])

    # Get onesided attribute (default is 1)
    onesided = get_attribute(node, "onesided", 1)

    def _stft(signal, frame_step, window, frame_length, onesided):
        # ONNX signal shape: [batch, signal_length, 1] (real) or [batch, signal_length, 2] (complex)
        # We need to convert to PyTorch format: [batch, signal_length]

        # Check if input is complex (last dim is 2)
        is_complex_input = signal.shape[-1] == 2

        if is_complex_input:
            # Convert to complex tensor
            signal_2d = torch.complex(signal[..., 0], signal[..., 1])
        else:
            # Squeeze the last dimension for real input
            signal_2d = signal.squeeze(-1)

        # Get scalar values
        hop_length = (
            int(frame_step.item())
            if isinstance(frame_step, torch.Tensor)
            else int(frame_step)
        )

        # Determine n_fft
        if frame_length is not None:
            n_fft = (
                int(frame_length.item())
                if isinstance(frame_length, torch.Tensor)
                else int(frame_length)
            )
        elif window is not None:
            n_fft = window.shape[0]
        else:
            raise ValueError("Either frame_length or window must be provided for STFT")

        # Determine onesided behavior
        # For complex input, onesided must be False
        onesided_bool = bool(onesided) and not is_complex_input

        # Call PyTorch stft
        # PyTorch stft returns [batch, n_fft, frames] (complex) or [batch, n_fft, frames, 2] (real)
        result = torch.stft(
            signal_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=False,  # ONNX does not pad
            onesided=onesided_bool,
            return_complex=True,
        )

        # result shape: [batch, bins, frames] (complex)
        # ONNX expects: [batch, frames, bins, 2]

        # Permute from [batch, bins, frames] to [batch, frames, bins]
        result = result.permute(0, 2, 1)

        # Convert complex to real representation [batch, frames, bins, 2]
        result = torch.view_as_real(result)

        return result

    return builder.call_function(
        _stft,
        args=(signal, frame_step, window, frame_length, onesided),
    )


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
