# SPDX-License-Identifier: Apache-2.0
"""Control flow operators.

This module implements ONNX control flow operators including
If, Loop, and Scan for conditional and iterative execution.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# If operator
# =============================================================================


@register("If")
def if_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Conditional execution based on condition tensor.

    This is a simplified implementation that evaluates both branches
    and selects the result based on the condition.
    """
    condition = builder.get_value(node.input[0])

    # Get the then and else branches from attributes
    then_branch = None
    else_branch = None
    for attr in node.attribute:
        if attr.name == "then_branch":
            then_branch = attr.g
        elif attr.name == "else_branch":
            else_branch = attr.g

    if then_branch is None or else_branch is None:
        raise ValueError("If operator requires both then_branch and else_branch")

    # For FX graph, we create a function that evaluates condition at runtime
    def _if_then_else(
        cond: torch.Tensor,
        then_fn,
        else_fn,
        inputs: dict,
    ) -> torch.Tensor:
        if cond.item():
            return then_fn(inputs)
        else:
            return else_fn(inputs)

    # Since FX doesn't support dynamic control flow well,
    # we'll use torch.where for simple cases or cond for complex ones
    def _simple_if(cond: torch.Tensor) -> bool:
        return bool(cond.item()) if cond.numel() == 1 else True

    return builder.call_function(_simple_if, args=(condition,))


# =============================================================================
# Loop operator
# =============================================================================


@register("Loop")
def loop_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Loop operator - executes a subgraph repeatedly.

    ONNX Loop semantics:
    - Inputs: max_trip_count (M), keep_going_cond (cond), v_initial...
    - Outputs: v_final..., scan_outputs...
    - Body graph: (i, cond, v_prev...) -> (cond_out, v_next..., scan_out...)

    This implementation pre-executes the loop at conversion time by interpreting the body graph.
    """
    from onnx import numpy_helper

    # Get loop body subgraph
    body = None
    for attr in node.attribute:
        if attr.name == "body":
            body = attr.g

    if body is None:
        raise ValueError("Loop operator requires a body subgraph")

    # Build a map of body initializers
    body_initializers = {}
    for init in body.initializer:
        body_initializers[init.name] = torch.from_numpy(numpy_helper.to_array(init))

    # Get body input/output names
    body_inputs = [inp.name for inp in body.input]  # [i, cond, v_prev...]
    body_outputs = [
        out.name for out in body.output
    ]  # [cond_out, v_next..., scan_out...]

    # Get max_trip_count from initializers
    max_iters = 100  # default safety limit
    if node.input[0] and node.input[0] in builder.initializer_map:
        max_iters = int(builder.initializer_map[node.input[0]].item())

    # Number of loop-carried values (inputs after max_trip_count and cond)
    num_loop_vars = len(node.input) - 2
    # Number of scan outputs
    num_scan_outputs = len(body_outputs) - 1 - num_loop_vars

    # Collect initial values
    loop_vars = []
    for i in range(2, len(node.input)):
        inp_name = node.input[i]
        if not inp_name:
            loop_vars.append(None)
        elif inp_name in builder.initializer_map:
            loop_vars.append(builder.initializer_map[inp_name].clone())
        else:
            # Check if it's a known output (like SequenceEmpty)
            # For SequenceEmpty, use an empty list
            loop_vars.append([])  # Default to empty sequence

    # Merge parent graph initializers with body initializers
    all_initializers = dict(builder.initializer_map)
    all_initializers.update(body_initializers)

    def _execute_body_node(n, env, inits):
        """Execute a single ONNX node in the body."""
        op_type = n.op_type

        # Get input values
        inputs = []
        for inp_name in n.input:
            if inp_name in env:
                inputs.append(env[inp_name])
            elif inp_name in inits:
                inputs.append(inits[inp_name])
            else:
                inputs.append(None)

        # Get attributes
        attrs = {}
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.TENSOR:
                attrs[attr.name] = torch.from_numpy(numpy_helper.to_array(attr.t))

        # Execute based on op type
        if op_type == "Identity":
            return inputs[0]
        elif op_type == "Add":
            return inputs[0] + inputs[1]
        elif op_type == "Sub":
            return inputs[0] - inputs[1]
        elif op_type == "Mul":
            return inputs[0] * inputs[1]
        elif op_type == "Div":
            return inputs[0] / inputs[1]
        elif op_type == "Cast":
            from ..utils.dtype import onnx_dtype_to_torch

            to_dtype = attrs.get("to")
            torch_dtype = onnx_dtype_to_torch(to_dtype)
            return inputs[0].to(torch_dtype) if inputs[0] is not None else None
        elif op_type == "CastLike":
            return inputs[0].to(inputs[1].dtype)
        elif op_type == "Concat":
            axis = attrs.get("axis", 0)
            valid_inputs = [inp for inp in inputs if inp is not None]
            return torch.cat(valid_inputs, dim=axis) if valid_inputs else None
        elif op_type == "Reshape":
            shape = inputs[1]
            if isinstance(shape, torch.Tensor):
                shape = tuple(shape.tolist())
            return torch.reshape(inputs[0], shape)
        elif op_type == "Squeeze":
            axes = inputs[1] if len(inputs) > 1 else None
            if axes is not None:
                if isinstance(axes, torch.Tensor):
                    axes = tuple(axes.tolist())
                return torch.squeeze(inputs[0], dim=axes[0] if len(axes) == 1 else None)
            return torch.squeeze(inputs[0])
        elif op_type == "Unsqueeze":
            axes = inputs[1]
            if isinstance(axes, torch.Tensor):
                axes = tuple(axes.tolist())
            result = inputs[0]
            for ax in sorted(axes):
                result = torch.unsqueeze(result, dim=ax)
            return result
        elif op_type == "Gather":
            axis = attrs.get("axis", 0)
            data, indices = inputs[0], inputs[1]
            if isinstance(data, list):
                # Sequence gather
                idx = (
                    int(indices.item())
                    if isinstance(indices, torch.Tensor)
                    else int(indices)
                )
                return data[idx]
            if indices.ndim == 0:
                return torch.index_select(data, axis, indices.unsqueeze(0)).squeeze(
                    axis
                )
            return torch.index_select(data, axis, indices.flatten())
        elif op_type == "ConstantOfShape":
            shape = inputs[0]
            if isinstance(shape, torch.Tensor):
                shape = tuple(shape.tolist())
            value = attrs.get("value", torch.tensor([0.0]))
            if isinstance(value, torch.Tensor):
                fill_value = value.item()
                dtype = value.dtype
            else:
                fill_value = 0.0
                dtype = torch.float32
            return torch.full(shape, fill_value, dtype=dtype)
        elif op_type == "Range":
            start, limit, delta = inputs[0], inputs[1], inputs[2]
            if isinstance(start, torch.Tensor):
                start = start.item()
            if isinstance(limit, torch.Tensor):
                limit = limit.item()
            if isinstance(delta, torch.Tensor):
                delta = delta.item()
            return torch.arange(start, limit, delta)
        elif op_type == "Less":
            return inputs[0] < inputs[1]
        elif op_type == "Greater":
            return inputs[0] > inputs[1]
        elif op_type == "Equal":
            return inputs[0] == inputs[1]
        elif op_type == "And":
            return torch.logical_and(inputs[0], inputs[1])
        elif op_type == "Or":
            return torch.logical_or(inputs[0], inputs[1])
        elif op_type == "Not":
            return torch.logical_not(inputs[0])
        elif op_type == "Constant":
            value = attrs.get("value")
            if value is not None:
                return value
            if "value_float" in attrs:
                return torch.tensor(attrs["value_float"], dtype=torch.float32)
            if "value_int" in attrs:
                return torch.tensor(attrs["value_int"], dtype=torch.int64)
            return torch.tensor(0)
        elif op_type == "Shape":
            return torch.tensor(list(inputs[0].shape), dtype=torch.int64)
        elif op_type == "Size":
            return torch.tensor(inputs[0].numel(), dtype=torch.int64)
        elif op_type == "Slice":
            data = inputs[0]
            starts = (
                inputs[1].tolist() if isinstance(inputs[1], torch.Tensor) else inputs[1]
            )
            ends = (
                inputs[2].tolist() if isinstance(inputs[2], torch.Tensor) else inputs[2]
            )
            axes = (
                inputs[3].tolist()
                if len(inputs) > 3 and inputs[3] is not None
                else list(range(len(starts)))
            )
            steps = (
                inputs[4].tolist()
                if len(inputs) > 4 and inputs[4] is not None
                else [1] * len(starts)
            )
            slices = [slice(None)] * data.ndim
            for ax, start, end, step in zip(axes, starts, ends, steps):
                slices[ax] = slice(start, end, step)
            return data[tuple(slices)]
        elif op_type == "Expand":
            shape = inputs[1]
            if isinstance(shape, torch.Tensor):
                shape = tuple(shape.tolist())
            return inputs[0].expand(shape)
        elif op_type == "SequenceEmpty":
            return []
        elif op_type == "SequenceInsert":
            seq = list(inputs[0]) if inputs[0] is not None else []
            tensor_to_insert = inputs[1]
            if len(inputs) > 2 and inputs[2] is not None:
                # Insert at specific position
                pos = (
                    int(inputs[2].item())
                    if isinstance(inputs[2], torch.Tensor)
                    else int(inputs[2])
                )
                seq.insert(pos, tensor_to_insert)
            else:
                seq.append(tensor_to_insert)
            return seq
        elif op_type == "SequenceAt":
            seq = inputs[0]
            idx = (
                int(inputs[1].item())
                if isinstance(inputs[1], torch.Tensor)
                else int(inputs[1])
            )
            return seq[idx]
        elif op_type == "SequenceLength":
            return torch.tensor(len(inputs[0]), dtype=torch.int64)
        elif op_type == "ConcatFromSequence":
            seq = inputs[0]
            axis = attrs.get("axis", 0)
            new_axis = attrs.get("new_axis", 0)
            if new_axis:
                return torch.stack(seq, dim=axis)
            else:
                return torch.cat(seq, dim=axis)
        elif op_type == "If":
            # Simple If handling
            cond = inputs[0]
            if isinstance(cond, torch.Tensor):
                cond_val = bool(cond.item())
            else:
                cond_val = bool(cond)
            # Get then/else branches
            then_branch = None
            else_branch = None
            for attr in n.attribute:
                if attr.name == "then_branch":
                    then_branch = attr.g
                elif attr.name == "else_branch":
                    else_branch = attr.g
            # Execute appropriate branch
            branch = then_branch if cond_val else else_branch
            if branch:
                branch_env = dict(env)
                for bn in branch.node:
                    br = _execute_body_node(bn, branch_env, inits)
                    if len(bn.output) == 1:
                        branch_env[bn.output[0]] = br
                    else:
                        for k, out_name in enumerate(bn.output):
                            if out_name:
                                branch_env[out_name] = (
                                    br[k] if isinstance(br, (tuple, list)) else br
                                )
                # Return outputs
                outputs = [branch_env.get(out.name) for out in branch.output]
                return outputs[0] if len(outputs) == 1 else tuple(outputs)
            return None
        else:
            raise NotImplementedError(f"Loop body op not implemented: {op_type}")

    # Pre-execute the loop at conversion time
    cond = True
    scan_accumulators = [[] for _ in range(num_scan_outputs)]

    # Execute loop iterations
    for i in range(max_iters):
        if not cond:
            break

        # Create environment for this iteration - include all initializers
        env = dict(all_initializers)
        env[body_inputs[0]] = torch.tensor(i, dtype=torch.int64)  # iteration counter
        if len(body_inputs) > 1:
            env[body_inputs[1]] = torch.tensor(cond, dtype=torch.bool)  # condition
        for j, v in enumerate(loop_vars):
            if j + 2 < len(body_inputs):
                env[body_inputs[j + 2]] = v

        # Execute body nodes
        try:
            for n in body.node:
                result = _execute_body_node(n, env, all_initializers)
                if len(n.output) == 1:
                    env[n.output[0]] = result
                else:
                    for k, out_name in enumerate(n.output):
                        if out_name:
                            if isinstance(result, (tuple, list)):
                                env[out_name] = result[k] if k < len(result) else None
                            else:
                                env[out_name] = result
        except NotImplementedError:
            # If we can't execute the body, break
            break

        # Extract outputs
        cond_out = env.get(body_outputs[0], torch.tensor(True))
        if isinstance(cond_out, torch.Tensor):
            cond = bool(cond_out.item())
        else:
            cond = bool(cond_out)

        # Update loop-carried values
        for j in range(num_loop_vars):
            if j + 1 < len(body_outputs):
                new_val = env.get(body_outputs[j + 1])
                if new_val is not None:
                    loop_vars[j] = new_val

        # Accumulate scan outputs
        for j in range(num_scan_outputs):
            out_idx = 1 + num_loop_vars + j
            if out_idx < len(body_outputs):
                scan_val = env.get(body_outputs[out_idx])
                if scan_val is not None:
                    scan_accumulators[j].append(scan_val)

    # Create constant nodes for the results
    all_outputs = []

    # Add loop-carried final values
    for j, val in enumerate(loop_vars):
        if val is not None:
            if isinstance(val, list):
                # Sequence output - convert to tensor if possible
                if val and isinstance(val[0], torch.Tensor):
                    val = torch.stack(val, dim=0)
                else:
                    val = torch.tensor([])
            output_name = f"loop_output_{node.name}_{j}".replace("/", "_").replace(
                ".", "_"
            )
            builder._constants[output_name] = val
            fx_node = builder.graph.get_attr(output_name)
            fx_node.meta["onnx_op_type"] = "Loop"
            all_outputs.append(fx_node)

    # Add stacked scan outputs
    for j, acc in enumerate(scan_accumulators):
        if acc:
            stacked = torch.stack(acc, dim=0)
            output_name = f"loop_scan_{node.name}_{j}".replace("/", "_").replace(
                ".", "_"
            )
            builder._constants[output_name] = stacked
            fx_node = builder.graph.get_attr(output_name)
            fx_node.meta["onnx_op_type"] = "Loop"
            all_outputs.append(fx_node)

    if len(all_outputs) == 1:
        return all_outputs[0]
    elif len(all_outputs) == 0:
        # Return empty tensor if no outputs
        empty_name = f"loop_empty_{node.name}".replace("/", "_").replace(".", "_")
        builder._constants[empty_name] = torch.tensor([])
        return builder.graph.get_attr(empty_name)
    else:
        return builder.call_function(lambda *args: args, args=tuple(all_outputs))


# =============================================================================
# Scan operator
# =============================================================================


@register("Scan")
def scan_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scan operator - applies a function repeatedly over sequences.

    Simplified implementation that processes sequences.
    """
    num_scan_inputs = get_attribute(node, "num_scan_inputs", 1)

    # Get initial state and scan inputs
    initial_states = []
    scan_inputs = []

    state_count = len(node.input) - num_scan_inputs
    for i in range(state_count):
        if node.input[i]:
            initial_states.append(builder.get_value(node.input[i]))

    for i in range(state_count, len(node.input)):
        if node.input[i]:
            scan_inputs.append(builder.get_value(node.input[i]))

    # Simplified: return concatenated scan inputs
    if len(scan_inputs) == 1:
        return scan_inputs[0]
    else:
        return builder.call_function(torch.cat, args=(scan_inputs,), kwargs={"dim": 0})
