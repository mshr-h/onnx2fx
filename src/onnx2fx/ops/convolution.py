# SPDX-License-Identifier: Apache-2.0
"""Convolution operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import compute_same_padding, get_optional_input

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


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
    bias = get_optional_input(builder, node, 2)

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
            k_shape = kernel_shape or weight.shape[2:]
            pad_list = compute_same_padding(
                tuple(input_shape),
                tuple(k_shape),
                tuple(strides),
                tuple(dilations),
                auto_pad,
            )
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
    bias = get_optional_input(builder, node, 2)

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

    bias = get_optional_input(builder, node, 3)
    mask = get_optional_input(builder, node, 4)

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
