# SPDX-License-Identifier: Apache-2.0
"""Sequence operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


@register("SequenceConstruct")
def sequence_construct(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Construct a sequence (list) from input tensors."""
    inputs = [builder.get_value(name) for name in node.input]
    return builder.call_function(lambda *args: list(args), args=tuple(inputs))


@register("SequenceAt")
def sequence_at(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get element at position from sequence."""
    seq = builder.get_value(node.input[0])
    position = builder.get_value(node.input[1])

    def _seq_at(s: list, p: torch.Tensor) -> torch.Tensor:
        idx = int(p.item()) if hasattr(p, "item") else int(p)
        return s[idx]

    return builder.call_function(_seq_at, args=(seq, position))


@register("SequenceLength")
def sequence_length(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Get length of sequence."""
    seq = builder.get_value(node.input[0])

    def _seq_len(s: list) -> torch.Tensor:
        return torch.tensor(len(s), dtype=torch.int64)

    return builder.call_function(_seq_len, args=(seq,))


@register("SequenceEmpty")
def sequence_empty(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an empty sequence."""
    return builder.call_function(list, args=())


@register("SequenceInsert")
def sequence_insert(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Insert tensor into sequence at position."""
    seq = builder.get_value(node.input[0])
    tensor = builder.get_value(node.input[1])
    position = builder.get_value(node.input[2]) if len(node.input) > 2 else None

    if position is not None:

        def _seq_insert(s: list, t: torch.Tensor, p: torch.Tensor) -> list:
            idx = int(p.item()) if hasattr(p, "item") else int(p)
            return s[:idx] + [t] + s[idx:]

        return builder.call_function(_seq_insert, args=(seq, tensor, position))
    else:

        def _seq_append(s: list, t: torch.Tensor) -> list:
            return s + [t]

        return builder.call_function(_seq_append, args=(seq, tensor))


@register("SequenceErase")
def sequence_erase(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Remove element from sequence at position."""
    seq = builder.get_value(node.input[0])
    position = builder.get_value(node.input[1]) if len(node.input) > 1 else None

    if position is not None:

        def _seq_erase(s: list, p: torch.Tensor) -> list:
            idx = int(p.item()) if hasattr(p, "item") else int(p)
            return s[:idx] + s[idx + 1 :]

        return builder.call_function(_seq_erase, args=(seq, position))
    else:

        def _seq_pop(s: list) -> list:
            return s[:-1]

        return builder.call_function(_seq_pop, args=(seq,))


@register("ConcatFromSequence")
def concat_from_sequence(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Concatenate tensors from sequence."""
    seq = builder.get_value(node.input[0])

    axis = get_attribute(node, "axis", 0)
    new_axis = get_attribute(node, "new_axis", 0)

    if new_axis:

        def _stack_seq(s: list, ax: int) -> torch.Tensor:
            return torch.stack(s, dim=ax)

        return builder.call_function(_stack_seq, args=(seq, axis))
    else:

        def _concat_seq(s: list, ax: int) -> torch.Tensor:
            return torch.cat(s, dim=ax)

        return builder.call_function(_concat_seq, args=(seq, axis))


@register("SplitToSequence")
def split_to_sequence(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Split tensor into sequence of tensors."""
    x = builder.get_value(node.input[0])
    split = builder.get_value(node.input[1]) if len(node.input) > 1 else None

    axis = get_attribute(node, "axis", 0)

    if split is not None:

        def _split_seq(t: torch.Tensor, s: torch.Tensor, ax: int) -> list:
            sizes = s.tolist() if hasattr(s, "tolist") else [s]
            return list(torch.split(t, sizes, dim=ax))

        return builder.call_function(_split_seq, args=(x, split, axis))
    else:

        def _split_ones(t: torch.Tensor, ax: int) -> list:
            return list(torch.split(t, 1, dim=ax))

        return builder.call_function(_split_ones, args=(x, axis))
