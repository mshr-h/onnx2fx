# SPDX-License-Identifier: Apache-2.0
"""Sequence operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import get_optional_input

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
    position = get_optional_input(builder, node, 2)

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
    position = get_optional_input(builder, node, 1)

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
    split = get_optional_input(builder, node, 1)

    axis = get_attribute(node, "axis", 0)
    keepdims = get_attribute(node, "keepdims", 1)

    if split is not None:

        def _split_seq(t: torch.Tensor, s: torch.Tensor, ax: int, keep: int) -> list:
            sizes = s.tolist() if hasattr(s, "tolist") else [s]
            # Handle scalar split value (equal splits of size s)
            if isinstance(sizes, (int, float)):
                sizes = int(sizes)
            splits = list(torch.split(t, sizes, dim=ax))
            if not keep:
                # Squeeze only if split size is 1 for each chunk
                splits = [
                    chunk.squeeze(ax) if chunk.shape[ax] == 1 else chunk
                    for chunk in splits
                ]
            return splits

        return builder.call_function(_split_seq, args=(x, split, axis, keepdims))
    else:

        def _split_ones(t: torch.Tensor, ax: int, keep: int) -> list:
            splits = list(torch.split(t, 1, dim=ax))
            if not keep:
                splits = [chunk.squeeze(ax) for chunk in splits]
            return splits

        return builder.call_function(_split_ones, args=(x, axis, keepdims))


@register("ReverseSequence")
def reverse_sequence(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Reverse sequences in a tensor."""
    x = builder.get_value(node.input[0])
    sequence_lens = builder.get_value(node.input[1])

    batch_axis = get_attribute(node, "batch_axis", 1)
    time_axis = get_attribute(node, "time_axis", 0)

    def _reverse_sequence(x, sequence_lens, batch_axis, time_axis):
        result = x.clone()
        for i, seq_len in enumerate(sequence_lens):
            seq_len_val = (
                seq_len.item() if isinstance(seq_len, torch.Tensor) else seq_len
            )
            # Create indices for this batch
            idx = [slice(None)] * x.dim()
            idx[batch_axis] = i
            idx[time_axis] = slice(None, int(seq_len_val))

            reversed_idx = list(idx)
            reversed_idx[time_axis] = slice(int(seq_len_val) - 1, None, -1)

            result[tuple(idx)] = x[tuple(reversed_idx)]

        return result

    return builder.call_function(
        _reverse_sequence, args=(x, sequence_lens, batch_axis, time_axis)
    )


@register("Optional", since_version=15)
def optional_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Create an optional value.

    If an input is provided, wraps it in a list to represent "optional with value".
    If no input is provided, creates an empty optional (empty list or None).

    Representation:
    - optional(tensor): [tensor]
    - optional(sequence): [[tensors...]]
    - empty optional: [] or None
    """
    if len(node.input) > 0 and node.input[0]:
        # Wrap the input in a list to represent optional with value
        value = builder.get_value(node.input[0])

        def _wrap_optional(v):
            return [v]

        return builder.call_function(_wrap_optional, args=(value,))
    else:
        # Create an empty optional (empty list)
        return builder.call_function(list, args=())


@register("OptionalHasElement", since_version=15)
def optional_has_element(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Check if an optional value has an element.

    An optional is considered empty if:
    - It is None
    - It is an empty list (ONNX representation of empty optional)
    """
    optional_input = get_optional_input(builder, node, 0)

    if optional_input is not None:

        def _has_element(opt):
            # Handle list representation of optional (used in ONNX test data)
            if isinstance(opt, list):
                return torch.tensor(len(opt) > 0, dtype=torch.bool)
            # Handle None representation
            return torch.tensor(opt is not None, dtype=torch.bool)

        return builder.call_function(_has_element, args=(optional_input,))
    else:
        # No input provided means empty optional
        return builder.call_function(lambda: torch.tensor(False, dtype=torch.bool))


@register("OptionalGetElement", since_version=15)
def optional_get_element(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Get the element from an optional value.

    Handles both None and list representations of optionals.
    Raises an error if the optional is empty.

    The behavior depends on the input type:
    - optional(tensor): [tensor] → return tensor
    - optional(sequence): [[t1, t2, ...]] → return [t1, t2, ...]
    - sequence (plain): [t1, t2, ...] → return as-is (sequence IS the value)
    """
    input_name = node.input[0]
    optional_input = builder.get_value(input_name)

    # Check if the input is declared as optional type in the model
    is_optional = builder.is_optional_type(input_name)

    if is_optional:
        # Input is optional type - need to unwrap
        # However, in ONNX Loops, the optional type can be "refined" to a non-optional
        # after the first iteration. We need to handle both cases:
        # 1. True optional wrapper: [value] (length 1) -> return value
        # 2. Plain value after refinement: return as-is

        def _get_element_from_optional(opt):
            if opt is None:
                raise ValueError("Cannot get element from empty optional")
            if isinstance(opt, list):
                if len(opt) == 0:
                    raise ValueError("Cannot get element from empty optional")
                if len(opt) == 1:
                    # This looks like an optional wrapper [value] - unwrap it
                    return opt[0]
                else:
                    # Length > 1: this is a plain sequence (after loop refinement)
                    # Return as-is since it's already unwrapped
                    return opt
            # Not a list - return as-is (tensor, etc.)
            return opt

        return builder.call_function(_get_element_from_optional, args=(optional_input,))
    else:
        # Input is a sequence type - return as-is
        # (The sequence itself is used as the "element" of an implicit optional)

        def _get_element_from_sequence(seq):
            if seq is None:
                raise ValueError("Cannot get element from empty optional")
            # Return sequence as-is
            return seq

        return builder.call_function(_get_element_from_sequence, args=(optional_input,))
