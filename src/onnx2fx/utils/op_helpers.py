# SPDX-License-Identifier: Apache-2.0
"""Helper utilities for operator implementations.

This module provides factory functions and helpers to reduce boilerplate
in operator implementations.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import onnx
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


def get_optional_input(
    builder: "GraphBuilder",
    node: onnx.NodeProto,
    index: int,
    default: Any = None,
) -> Any:
    """Get an optional input from a node, returning default if not present.

    This replaces the common pattern:
        if len(node.input) > N and node.input[N]:
            value = builder.get_value(node.input[N])

    Args:
        builder: The graph builder instance.
        node: The ONNX node.
        index: The input index to retrieve.
        default: Default value if input is not present.

    Returns:
        The input value or default.
    """
    if len(node.input) > index and node.input[index]:
        return builder.get_value(node.input[index])
    return default


def unary_op(
    torch_fn: Callable[..., torch.Tensor],
    doc: Optional[str] = None,
) -> Callable[["GraphBuilder", onnx.NodeProto], torch.fx.Node]:
    """Create a handler for simple unary operators.

    This replaces the common pattern:
        @register("OpName")
        def op_name(builder, node):
            x = builder.get_value(node.input[0])
            return builder.call_function(torch.op_name, args=(x,))

    Args:
        torch_fn: The PyTorch function to call.
        doc: Optional docstring for the handler.

    Returns:
        A handler function for the operator.
    """

    def handler(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
        x = builder.get_value(node.input[0])
        return builder.call_function(torch_fn, args=(x,))

    if doc:
        handler.__doc__ = doc
    else:
        handler.__doc__ = f"Element-wise {torch_fn.__name__}."
    return handler


def binary_op(
    torch_fn: Callable[..., torch.Tensor],
    doc: Optional[str] = None,
) -> Callable[["GraphBuilder", onnx.NodeProto], torch.fx.Node]:
    """Create a handler for simple binary operators.

    This replaces the common pattern:
        @register("OpName")
        def op_name(builder, node):
            lhs = builder.get_value(node.input[0])
            rhs = builder.get_value(node.input[1])
            return builder.call_function(torch.op_name, args=(lhs, rhs))

    Args:
        torch_fn: The PyTorch function to call.
        doc: Optional docstring for the handler.

    Returns:
        A handler function for the operator.
    """

    def handler(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
        lhs = builder.get_value(node.input[0])
        rhs = builder.get_value(node.input[1])
        return builder.call_function(torch_fn, args=(lhs, rhs))

    if doc:
        handler.__doc__ = doc
    else:
        handler.__doc__ = f"Element-wise {torch_fn.__name__}."
    return handler


def compute_same_padding(
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    mode: str,
    use_effective_kernel: bool = False,
) -> list[int]:
    """Compute SAME_UPPER or SAME_LOWER padding.

    This consolidates the repeated auto_pad handling logic from
    convolution.py and pooling.py.

    Args:
        input_shape: Spatial dimensions of input (excluding batch and channel).
        kernel_shape: Kernel dimensions.
        strides: Stride values.
        dilations: Dilation values.
        mode: Either "SAME_UPPER" or "SAME_LOWER".
        use_effective_kernel: If True, use effective kernel size
            ((k-1)*d + 1) for padding calculation (used by AvgPool/LpPool).
            If False, use the standard formula with separate dilation term.

    Returns:
        A list of padding values in F.pad format (reversed order):
        [xn_begin, xn_end, ..., x1_begin, x1_end]
    """
    # Calculate output shape with SAME padding
    # output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    output_shape = [(s + st - 1) // st for s, st in zip(input_shape, strides)]

    # Calculate total padding needed for each dimension
    if use_effective_kernel:
        # For AvgPool/LpPool: use effective kernel = (k-1)*d + 1
        effective_kernel = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]
        pad_total = [
            max(0, (o - 1) * st + ek - i)
            for i, o, ek, st in zip(input_shape, output_shape, effective_kernel, strides)
        ]
    else:
        # Standard formula: (output - 1) * stride + (kernel - 1) * dilation + 1 - input
        pad_total = [
            max(0, (o - 1) * st + (k - 1) * d + 1 - i)
            for i, o, k, st, d in zip(
                input_shape, output_shape, kernel_shape, strides, dilations
            )
        ]

    # Build pad list in F.pad format (reversed spatial order)
    pad_list: list[int] = []
    if mode == "SAME_UPPER":
        # SAME_UPPER: more padding at end (right/bottom)
        for p in reversed(pad_total):
            pad_list.extend([p // 2, p - p // 2])
    else:  # SAME_LOWER
        # SAME_LOWER: more padding at beginning (left/top)
        for p in reversed(pad_total):
            pad_list.extend([p - p // 2, p // 2])

    return pad_list


def pad_list_to_onnx_pads(pad_list: list[int], ndim: int) -> list[int]:
    """Convert F.pad format to ONNX pads format.

    F.pad format: [xn_begin, xn_end, ..., x1_begin, x1_end] (reversed)
    ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]

    Args:
        pad_list: Padding in F.pad format.
        ndim: Number of spatial dimensions.

    Returns:
        Padding in ONNX format.
    """
    pads_onnx = [0] * (2 * ndim)
    for i in range(ndim):
        pads_onnx[i] = pad_list[2 * (ndim - 1 - i)]
        pads_onnx[i + ndim] = pad_list[2 * (ndim - 1 - i) + 1]
    return pads_onnx


def apply_auto_pad(
    x: torch.Tensor,
    kernel_shape: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    auto_pad: str,
    pad_value: Union[int, float] = 0,
) -> tuple[torch.Tensor, int]:
    """Apply auto-padding to input tensor if needed.

    Args:
        x: Input tensor with shape (N, C, *spatial_dims).
        kernel_shape: Kernel dimensions.
        strides: Stride values.
        dilations: Dilation values.
        auto_pad: Auto-pad mode ("NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID").
        pad_value: Value to use for padding.

    Returns:
        Tuple of (padded tensor, padding value for conv/pool operation).
        If auto_pad is applied, the returned padding value is 0.
    """
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        input_shape = tuple(x.shape[2:])  # Spatial dimensions
        pad_list = compute_same_padding(
            input_shape, kernel_shape, strides, dilations, auto_pad
        )
        x = F.pad(x, pad_list, value=pad_value)
        return x, 0
    return x, 0  # NOTSET or VALID - no auto-padding applied
