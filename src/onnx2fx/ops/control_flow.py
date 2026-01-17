# SPDX-License-Identifier: Apache-2.0
"""Control flow operators.

This module implements ONNX control flow operators like Loop and If.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import onnx
import torch
import torch.nn as nn
import torch.fx
from onnx import numpy_helper

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


def _build_subgraph_module(
    body_graph: onnx.GraphProto,
    parent_env: Dict[str, torch.fx.Node],
    parent_opset_versions: Dict[str, int],
) -> Tuple[torch.fx.GraphModule, List[str], List[str], List[str]]:
    """Build an FX GraphModule from an ONNX subgraph.

    Parameters
    ----------
    body_graph : onnx.GraphProto
        The ONNX subgraph to convert.
    parent_env : Dict[str, torch.fx.Node]
        Environment from parent graph (for accessing outer scope values).
    parent_opset_versions : Dict[str, int]
        Opset versions from parent model.

    Returns
    -------
    Tuple[torch.fx.GraphModule, List[str], List[str], List[str]]
        The FX GraphModule, list of input names, list of output names, and outer refs.
    """
    from ..op_registry import get_handler

    graph = torch.fx.Graph()
    env: Dict[str, torch.fx.Node] = {}
    constants: Dict[str, torch.Tensor] = {}

    # Get input and output names
    input_names = [inp.name for inp in body_graph.input]
    output_names = [out.name for out in body_graph.output]

    # Load initializers from subgraph
    initializer_map: Dict[str, torch.Tensor] = {}
    for initializer in body_graph.initializer:
        np_array = numpy_helper.to_array(initializer)
        initializer_map[initializer.name] = torch.from_numpy(np_array.copy())

    # Register initializers as constants
    for name, tensor in initializer_map.items():
        safe_name = name.replace(".", "_").replace("/", "_")
        constants[safe_name] = tensor
        fx_node = graph.get_attr(safe_name)
        env[name] = fx_node

    # Create placeholders for subgraph inputs
    for inp in body_graph.input:
        if inp.name in env:
            continue  # Skip if already loaded as initializer
        safe_name = inp.name.replace(".", "_").replace("/", "_").replace("-", "_")
        placeholder = graph.placeholder(safe_name)
        env[inp.name] = placeholder

    # Add references to parent scope values that are used in the subgraph
    # These will be passed as additional inputs
    outer_refs: List[str] = []
    for node in body_graph.node:
        for inp_name in node.input:
            if inp_name and inp_name not in env and inp_name in parent_env:
                if inp_name not in outer_refs:  # Avoid duplicates
                    outer_refs.append(inp_name)
                    safe_name = inp_name.replace(".", "_").replace("/", "_").replace("-", "_")
                    placeholder = graph.placeholder(f"outer_{safe_name}")
                    env[inp_name] = placeholder

    # Create a minimal builder-like object for handler calls
    class SubgraphBuilder:
        def __init__(self):
            self.graph = graph
            self.env = env
            self._opset_versions = parent_opset_versions
            self._constants = constants
            self._submodules: Dict[str, nn.Module] = {}
            self.initializer_map = initializer_map

        @property
        def opset_version(self) -> int:
            return self._opset_versions.get("", 1)

        def get_opset_version(self, domain: str = "") -> int:
            return self._opset_versions.get(domain, 1)

        def get_value(self, name: str) -> torch.fx.Node:
            if name not in self.env:
                raise KeyError(f"Value '{name}' not found in subgraph environment")
            return self.env[name]

        def has_value(self, name: str) -> bool:
            return name in self.env

        def call_function(
            self,
            func,
            args: tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
        ) -> torch.fx.Node:
            return self.graph.call_function(func, args=tuple(args), kwargs=kwargs or {})

        def register_submodule(self, name: str, module: nn.Module) -> str:
            safe_name = name.replace(".", "_").replace("/", "_").replace("-", "_")
            if safe_name in self._submodules:
                counter = 0
                while f"{safe_name}_{counter}" in self._submodules:
                    counter += 1
                safe_name = f"{safe_name}_{counter}"
            self._submodules[safe_name] = module
            return safe_name

        def call_module(
            self,
            module_name: str,
            args: tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
        ) -> torch.fx.Node:
            return self.graph.call_module(module_name, args=tuple(args), kwargs=kwargs or {})

    builder = SubgraphBuilder()

    # Convert nodes
    for node in body_graph.node:
        domain = node.domain if node.domain else ""
        opset = builder.get_opset_version(domain)
        handler = get_handler(node.op_type, domain, opset)
        if handler is None:
            raise ValueError(
                f"Unsupported operator in subgraph: {node.op_type} (domain={domain})"
            )
        fx_node = handler(builder, node)

        # Handle outputs
        if len(node.output) == 1:
            env[node.output[0]] = fx_node
        else:
            for i, output_name in enumerate(node.output):
                if output_name:
                    getitem_node = graph.call_function(
                        lambda x, idx=i: x[idx] if isinstance(x, (tuple, list)) else x,
                        args=(fx_node, i),
                    )
                    env[output_name] = getitem_node

    # Create output - return tuple of output values
    output_nodes = []
    for out_name in output_names:
        if out_name in env:
            output_nodes.append(env[out_name])
        else:
            raise KeyError(f"Output '{out_name}' not found in subgraph environment")

    if len(output_nodes) == 1:
        graph.output(output_nodes[0])
    else:
        graph.output(tuple(output_nodes))

    # Create the module
    root_module = nn.Module()
    for name, tensor in constants.items():
        root_module.register_buffer(name, tensor)
    for name, submod in builder._submodules.items():
        root_module.add_module(name, submod)

    module = torch.fx.GraphModule(root_module, graph)
    module.graph.lint()

    return module, input_names, output_names, outer_refs


class LoopModule(nn.Module):
    """Module that executes an ONNX Loop."""

    def __init__(
        self,
        body_module: torch.fx.GraphModule,
        n_loop_carried: int,
        n_scan_outputs: int,
        n_loop_vars: int,
        n_outer_vars: int,
    ):
        super().__init__()
        self.body = body_module
        self.n_loop_carried = n_loop_carried
        self.n_scan_outputs = n_scan_outputs
        self.n_loop_vars = n_loop_vars
        self.n_outer_vars = n_outer_vars

    def forward(self, max_iters, init_cond, *args) -> Tuple[torch.Tensor, ...]:
        """Execute the loop.

        Args are: loop_vars..., outer_vals...
        Returns final loop-carried values followed by concatenated scan outputs.
        """
        # Split args into loop_vars and outer_vals
        loop_vars = list(args[:self.n_loop_vars])
        outer_vals = list(args[self.n_loop_vars:])

        # Determine max iterations
        if max_iters is not None:
            max_i = int(max_iters.item()) if hasattr(max_iters, 'item') else int(max_iters)
        else:
            max_i = 2**63 - 1  # Very large number

        # Initial condition
        if init_cond is not None:
            cond = bool(init_cond.item()) if hasattr(init_cond, 'item') else bool(init_cond)
        else:
            cond = True

        # Current loop-carried values
        current_vars = list(loop_vars)

        # Scan output accumulators
        scan_outputs: List[List[torch.Tensor]] = [[] for _ in range(self.n_scan_outputs)]

        i = 0
        while i < max_i and cond:
            # Prepare inputs for body: iteration_num, condition, loop_carried..., outer...
            iter_tensor = torch.tensor(i, dtype=torch.int64)
            cond_tensor = torch.tensor(cond, dtype=torch.bool)

            # Call body function
            body_inputs = [iter_tensor, cond_tensor] + current_vars + outer_vals
            outputs = self.body(*body_inputs)

            # Handle single vs multiple outputs
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # First output is new condition
            new_cond = outputs[0]
            cond = bool(new_cond.item()) if hasattr(new_cond, 'item') else bool(new_cond)

            # Next n_loop_carried outputs are updated loop-carried values
            current_vars = list(outputs[1:1 + self.n_loop_carried])

            # Remaining outputs are scan outputs for this iteration
            for j in range(self.n_scan_outputs):
                scan_outputs[j].append(outputs[1 + self.n_loop_carried + j])

            i += 1

        # Prepare final outputs: loop-carried values, then stacked scan outputs
        final_outputs = list(current_vars)
        for scan_list in scan_outputs:
            if scan_list:
                # Stack scan outputs along a new first dimension
                final_outputs.append(torch.stack(scan_list, dim=0))
            else:
                # Empty scan output - create empty tensor
                final_outputs.append(torch.tensor([]))

        return tuple(final_outputs)


class IfModule(nn.Module):
    """Module that executes an ONNX If conditional."""

    def __init__(
        self,
        then_module: torch.fx.GraphModule,
        else_module: torch.fx.GraphModule,
        n_then_outer: int,
        n_else_outer: int,
    ):
        super().__init__()
        self.then_branch = then_module
        self.else_branch = else_module
        self.n_then_outer = n_then_outer
        self.n_else_outer = n_else_outer

    def forward(self, condition, *args) -> Any:
        """Execute the conditional.

        Args are: then_outer..., else_outer...
        Returns the outputs of the selected branch.
        """
        then_outer = list(args[:self.n_then_outer])
        else_outer = list(args[self.n_then_outer:])

        cond_val = bool(condition.item()) if hasattr(condition, 'item') else bool(condition)

        if cond_val:
            result = self.then_branch(*then_outer)
        else:
            result = self.else_branch(*else_outer)

        return result


@register("Loop")
def loop_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Loop operator.

    Loop has inputs: M (max trip count), cond (initial condition), v_initial... (loop-carried deps)
    Loop has attribute: body (GraphProto)
    Body inputs: iteration_num, condition, loop_carried_deps...
    Body outputs: condition, loop_carried_deps..., scan_outputs...
    """
    # Get body subgraph
    body_graph = get_attribute(node, "body")
    if body_graph is None:
        raise ValueError("Loop operator requires 'body' attribute")

    # Get inputs
    max_trip_count = builder.get_value(node.input[0]) if node.input[0] else None
    initial_cond = builder.get_value(node.input[1]) if len(node.input) > 1 and node.input[1] else None
    loop_carried_inputs = [
        builder.get_value(node.input[i]) for i in range(2, len(node.input))
    ]

    # Build subgraph module
    body_module, body_input_names, body_output_names, outer_refs = _build_subgraph_module(
        body_graph, builder.env, builder._opset_versions
    )

    # Get outer scope values that the body references
    outer_values = [builder.get_value(name) for name in outer_refs]

    # Number of loop-carried dependencies (excluding iteration_num and condition in body inputs)
    n_loop_carried = len(body_input_names) - 2
    # Number of scan outputs
    n_scan_outputs = len(body_output_names) - 1 - n_loop_carried

    # Create the loop module
    loop_module = LoopModule(
        body_module,
        n_loop_carried,
        n_scan_outputs,
        len(loop_carried_inputs),
        len(outer_values),
    )

    # Register the loop module
    module_name = builder.register_submodule(f"loop_{node.name or 'op'}", loop_module)

    # Build flat args for call_module: max_iters, init_cond, loop_vars..., outer_vals...
    args = [max_trip_count, initial_cond] + loop_carried_inputs + outer_values

    result = builder.call_module(module_name, args=tuple(args))

    return result


@register("If")
def if_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX If operator.

    If has input: cond (boolean condition)
    If has attributes: then_branch (GraphProto), else_branch (GraphProto)
    Both branches must have the same number and types of outputs.
    """
    # Get condition
    cond = builder.get_value(node.input[0])

    # Get branch subgraphs
    then_graph = get_attribute(node, "then_branch")
    else_graph = get_attribute(node, "else_branch")

    if then_graph is None or else_graph is None:
        raise ValueError("If operator requires 'then_branch' and 'else_branch' attributes")

    # Build subgraph modules for both branches
    then_module, then_input_names, then_output_names, then_outer_refs = _build_subgraph_module(
        then_graph, builder.env, builder._opset_versions
    )
    else_module, else_input_names, else_output_names, else_outer_refs = _build_subgraph_module(
        else_graph, builder.env, builder._opset_versions
    )

    # Get outer scope values for both branches
    then_outer_values = [builder.get_value(name) for name in then_outer_refs]
    else_outer_values = [builder.get_value(name) for name in else_outer_refs]

    # Create the if module
    if_module = IfModule(
        then_module,
        else_module,
        len(then_outer_values),
        len(else_outer_values),
    )

    # Register the if module
    module_name = builder.register_submodule(f"if_{node.name or 'op'}", if_module)

    # Build flat args for call_module: condition, then_outer..., else_outer...
    args = [cond] + then_outer_values + else_outer_values

    result = builder.call_module(module_name, args=tuple(args))

    return result
