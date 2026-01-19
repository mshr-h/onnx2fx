# SPDX-License-Identifier: Apache-2.0
"""Normalization operators."""

from typing import TYPE_CHECKING

import onnx
import torch
import torch.nn.functional as F

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import get_optional_input

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


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

    bias = get_optional_input(builder, node, 2)

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
