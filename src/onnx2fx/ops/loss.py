# SPDX-License-Identifier: Apache-2.0
"""Loss function operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import get_optional_input

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


@register("SoftmaxCrossEntropyLoss")
def softmax_cross_entropy_loss(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Softmax cross entropy loss."""
    scores = builder.get_value(node.input[0])
    labels = builder.get_value(node.input[1])
    weights = get_optional_input(builder, node, 2)

    ignore_index = get_attribute(node, "ignore_index", -100)
    reduction = get_attribute(node, "reduction", "mean")

    kwargs = {"ignore_index": ignore_index, "reduction": reduction}
    if weights is not None:
        kwargs["weight"] = weights

    return builder.call_function(
        torch.nn.functional.cross_entropy, args=(scores, labels), kwargs=kwargs
    )


@register("NegativeLogLikelihoodLoss")
def negative_log_likelihood_loss(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Negative log likelihood loss."""
    input_node = builder.get_value(node.input[0])
    target = builder.get_value(node.input[1])
    weight = get_optional_input(builder, node, 2)

    ignore_index = get_attribute(node, "ignore_index", -100)
    reduction = get_attribute(node, "reduction", "mean")

    kwargs = {"ignore_index": ignore_index, "reduction": reduction}
    if weight is not None:
        kwargs["weight"] = weight

    return builder.call_function(
        torch.nn.functional.nll_loss, args=(input_node, target), kwargs=kwargs
    )
