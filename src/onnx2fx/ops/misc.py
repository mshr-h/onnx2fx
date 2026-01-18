# SPDX-License-Identifier: Apache-2.0
"""Miscellaneous operators.

This module implements ONNX operators that don't fit into other categories.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# String operators (limited support)
# =============================================================================


@register("StringNormalizer")
def string_normalizer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """String normalization for numpy string arrays.

    Applies case normalization and stopword filtering to string arrays.
    """
    from ..utils.attributes import get_attribute

    x = builder.get_value(node.input[0])
    case_change_action = get_attribute(node, "case_change_action", "NONE")
    is_case_sensitive = get_attribute(node, "is_case_sensitive", 0)
    stopwords = get_attribute(node, "stopwords", [])
    locale = get_attribute(node, "locale", "")

    def _string_normalizer(
        arr, case_action: str, case_sensitive: int, stops: list, loc: str
    ):
        import numpy as np

        original_shape = arr.shape
        is_2d = len(original_shape) == 2

        if is_2d:
            # Process each row separately for 2D arrays
            result_rows = []
            for row_idx in range(original_shape[0]):
                row = arr[row_idx]
                # Filter stopwords from this row
                if stops:
                    if case_sensitive:
                        stop_set = set(stops)
                        filtered = [s for s in row if s not in stop_set]
                    else:
                        stop_set_lower = {s.lower() for s in stops}
                        filtered = [
                            s
                            for s in row
                            if not isinstance(s, str) or s.lower() not in stop_set_lower
                        ]
                else:
                    filtered = list(row)

                # Apply case change
                if case_action == "LOWER":
                    filtered = [
                        s.lower() if isinstance(s, str) else s for s in filtered
                    ]
                elif case_action == "UPPER":
                    filtered = [
                        s.upper() if isinstance(s, str) else s for s in filtered
                    ]

                result_rows.append(filtered)

            # Find max length and build output
            if result_rows:
                # Pad shorter rows if needed (shouldn't happen for valid ONNX)
                output = np.array(result_rows, dtype=object)
                return output
            return np.array([[]], dtype=object)

        # 1D case
        flat = arr.flatten()

        # Filter stopwords first (before case change for matching)
        if stops:
            if case_sensitive:
                stop_set = set(stops)
                flat = np.array([s for s in flat if s not in stop_set], dtype=object)
            else:
                stop_set_lower = {s.lower() for s in stops}
                result = []
                for s in flat:
                    if isinstance(s, str):
                        if s.lower() not in stop_set_lower:
                            result.append(s)
                    else:
                        result.append(s)
                flat = (
                    np.array(result, dtype=object)
                    if result
                    else np.array([], dtype=object)
                )

        # Apply case change after filtering
        if case_action == "LOWER":
            flat = np.array(
                [s.lower() if isinstance(s, str) else s for s in flat], dtype=object
            )
        elif case_action == "UPPER":
            flat = np.array(
                [s.upper() if isinstance(s, str) else s for s in flat], dtype=object
            )

        # Handle empty result - ONNX spec says return [''] for empty
        if len(flat) == 0:
            return np.array([""], dtype=object)

        return flat

    return builder.call_function(
        _string_normalizer,
        args=(x, case_change_action, is_case_sensitive, stopwords, locale),
    )


# =============================================================================
# Training operators (ai.onnx.preview.training domain)
# =============================================================================


@register("Gradient", domain="ai.onnx.preview.training")
def gradient(builder: "GraphBuilder", node: onnx.NodeProto) -> list:
    """Compute gradients of y with respect to xs using PyTorch autograd.

    This operator computes symbolic gradients as specified in the ONNX
    ai.onnx.preview.training domain. It recomputes the forward pass with
    requires_grad=True to enable autograd.
    """
    from ..utils.attributes import get_attribute

    # Get the xs (inputs to differentiate with respect to) and y (output to differentiate)
    xs_names = get_attribute(node, "xs", [])
    y_name = get_attribute(node, "y", "")

    # Get the xs nodes - these are the graph inputs we differentiate with respect to
    xs_nodes = [builder.get_value(name) for name in xs_names]

    # Find the path from xs to y by analyzing the ONNX graph
    # We need to collect all nodes that contribute to y
    graph = builder.model.graph
    node_outputs = {}  # output_name -> node
    for n in graph.node:
        for out in n.output:
            node_outputs[out] = n

    # Trace back from y to find all operations needed
    def trace_ops(target_name, collected_nodes):
        """Recursively collect all nodes needed to compute target_name."""
        if target_name in xs_names or target_name not in node_outputs:
            return
        n = node_outputs[target_name]
        if n not in collected_nodes:
            collected_nodes.append(n)
            for inp in n.input:
                trace_ops(inp, collected_nodes)

    ops_to_y = []
    trace_ops(y_name, ops_to_y)
    ops_to_y.reverse()  # Order from inputs to output

    # Create a function that recomputes y from xs with autograd enabled
    ops_info = [(n.op_type, list(n.input), list(n.output)) for n in ops_to_y]

    def _compute_gradient_symbolic(ops_info, xs_names, y_name, *xs_values):
        """Recompute forward pass and compute gradients."""
        # Create tensors with requires_grad
        env = {}
        for name, val in zip(xs_names, xs_values):
            if isinstance(val, torch.Tensor):
                env[name] = val.detach().clone().requires_grad_(True)
            else:
                env[name] = torch.tensor(val, dtype=torch.float32, requires_grad=True)

        # Replay operations
        for op_type, inputs, outputs in ops_info:
            input_tensors = [env[inp] for inp in inputs if inp in env]
            if op_type == "Add":
                result = input_tensors[0] + input_tensors[1]
            elif op_type == "Mul":
                result = input_tensors[0] * input_tensors[1]
            elif op_type == "Sub":
                result = input_tensors[0] - input_tensors[1]
            elif op_type == "Div":
                result = input_tensors[0] / input_tensors[1]
            else:
                # Fallback for unsupported ops
                result = input_tensors[0] if input_tensors else torch.tensor(0.0)
            for out in outputs:
                env[out] = result

        # Get y and compute gradients
        y = env.get(y_name)
        if y is None:
            return tuple(torch.zeros_like(x) for x in xs_values)

        xs_tensors = [env[name] for name in xs_names]
        grads = torch.autograd.grad(
            outputs=y,
            inputs=xs_tensors,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            allow_unused=True,
        )
        return grads

    # Create the gradient computation node
    result = builder.call_function(
        _compute_gradient_symbolic,
        args=(ops_info, xs_names, y_name, *xs_nodes),
    )

    # Return list of gradient outputs (one per xs input)
    outputs = []
    for i in range(len(xs_names)):

        def _get_grad_i(grads, idx=i):
            return grads[idx]

        outputs.append(builder.call_function(_get_grad_i, args=(result,)))

    return outputs
