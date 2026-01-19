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

from ..utils.names import sanitize_name
from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import get_optional_input

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


def _collect_all_subgraph_inputs(node: onnx.NodeProto) -> set:
    """Recursively collect all inputs from a node including nested subgraph inputs.

    For control flow nodes like If and Loop, this also collects inputs that
    are referenced by nested subgraphs.

    Parameters
    ----------
    node : onnx.NodeProto
        The ONNX node to collect inputs from.

    Returns
    -------
    set
        Set of all input names referenced by this node and its nested subgraphs.
    """
    inputs = set(node.input)

    # Collect inputs from subgraphs (for If, Loop, etc.)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            subgraph = attr.g
            # Collect subgraph's own initializers and inputs as local values
            local_values = set()
            for init in subgraph.initializer:
                local_values.add(init.name)
            for inp in subgraph.input:
                local_values.add(inp.name)

            # Recursively collect inputs from subgraph nodes
            for sub_node in subgraph.node:
                sub_inputs = _collect_all_subgraph_inputs(sub_node)
                # Add outputs of this subgraph node to local values
                for out in sub_node.output:
                    if out:
                        local_values.add(out)
                # Inputs not satisfied locally are outer references
                for sub_inp in sub_inputs:
                    if sub_inp and sub_inp not in local_values:
                        inputs.add(sub_inp)

    return inputs


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
        safe_name = sanitize_name(name)
        constants[safe_name] = tensor
        fx_node = graph.get_attr(safe_name)
        env[name] = fx_node

    # Create placeholders for subgraph inputs
    for inp in body_graph.input:
        if inp.name in env:
            continue  # Skip if already loaded as initializer
        safe_name = sanitize_name(inp.name)
        placeholder = graph.placeholder(safe_name)
        env[inp.name] = placeholder

    # Collect all values that will be produced by nodes in this subgraph
    subgraph_outputs = set()
    for node in body_graph.node:
        for out in node.output:
            if out:
                subgraph_outputs.add(out)

    # Add references to parent scope values that are used in the subgraph
    # (including nested subgraphs). These will be passed as additional inputs.
    outer_refs: List[str] = []
    for node in body_graph.node:
        # Collect all inputs including from nested subgraphs
        all_inputs = _collect_all_subgraph_inputs(node)
        for inp_name in all_inputs:
            # Skip if empty, already in env, or will be produced by a node in this subgraph
            if not inp_name or inp_name in env or inp_name in subgraph_outputs:
                continue
            if inp_name in parent_env:
                if inp_name not in outer_refs:  # Avoid duplicates
                    outer_refs.append(inp_name)
                    safe_name = sanitize_name(inp_name)
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
            return self.graph.call_module(
                module_name, args=tuple(args), kwargs=kwargs or {}
            )

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
        loop_vars = list(args[: self.n_loop_vars])
        outer_vals = list(args[self.n_loop_vars :])

        # Determine max iterations
        if max_iters is not None:
            max_i = (
                int(max_iters.item()) if hasattr(max_iters, "item") else int(max_iters)
            )
        else:
            max_i = 2**63 - 1  # Very large number

        # Initial condition
        if init_cond is not None:
            cond = (
                bool(init_cond.item())
                if hasattr(init_cond, "item")
                else bool(init_cond)
            )
        else:
            cond = True

        # Current loop-carried values
        current_vars = list(loop_vars)

        # Scan output accumulators
        scan_outputs: List[List[torch.Tensor]] = [
            [] for _ in range(self.n_scan_outputs)
        ]

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
            cond = (
                bool(new_cond.item()) if hasattr(new_cond, "item") else bool(new_cond)
            )

            # Next n_loop_carried outputs are updated loop-carried values
            current_vars = list(outputs[1 : 1 + self.n_loop_carried])

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
        then_outer = list(args[: self.n_then_outer])
        else_outer = list(args[self.n_then_outer :])

        cond_val = (
            bool(condition.item()) if hasattr(condition, "item") else bool(condition)
        )

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
    initial_cond = get_optional_input(builder, node, 1)
    loop_carried_inputs = [
        builder.get_value(node.input[i]) for i in range(2, len(node.input))
    ]

    # Build subgraph module
    body_module, body_input_names, body_output_names, outer_refs = (
        _build_subgraph_module(body_graph, builder.env, builder._opset_versions)
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


class ScanModule(nn.Module):
    """Module that executes an ONNX Scan operation."""

    def __init__(
        self,
        body_module: torch.fx.GraphModule,
        n_state_vars: int,
        n_scan_inputs: int,
        n_scan_outputs: int,
        n_outer_vars: int,
        scan_input_axes: List[int],
        scan_output_axes: List[int],
        scan_input_directions: List[int],
        scan_output_directions: List[int],
    ):
        super().__init__()
        self.body = body_module
        self.n_state_vars = n_state_vars
        self.n_scan_inputs = n_scan_inputs
        self.n_scan_outputs = n_scan_outputs
        self.n_outer_vars = n_outer_vars
        self.scan_input_axes = scan_input_axes
        self.scan_output_axes = scan_output_axes
        self.scan_input_directions = scan_input_directions
        self.scan_output_directions = scan_output_directions

    def forward(self, *args) -> Tuple[torch.Tensor, ...]:
        """Execute the scan.

        Args are: state_vars..., scan_inputs..., outer_vals...
        Returns final state variables followed by scan outputs.
        """
        # Split args
        state_vars = list(args[: self.n_state_vars])
        scan_inputs = list(
            args[self.n_state_vars : self.n_state_vars + self.n_scan_inputs]
        )
        outer_vals = list(args[self.n_state_vars + self.n_scan_inputs :])

        # Determine sequence length from first scan input
        if self.n_scan_inputs > 0:
            first_input = scan_inputs[0]
            axis = self.scan_input_axes[0] if self.scan_input_axes else 0
            sequence_length = first_input.shape[axis]
        else:
            sequence_length = 0

        # Initialize scan output accumulators
        scan_outputs: List[List[torch.Tensor]] = [
            [] for _ in range(self.n_scan_outputs)
        ]

        # Current state
        current_state = list(state_vars)

        # Execute loop
        for t in range(sequence_length):
            # Extract scan input elements for this iteration
            scan_input_elts = []
            for i, scan_input in enumerate(scan_inputs):
                axis = self.scan_input_axes[i] if i < len(self.scan_input_axes) else 0
                direction = (
                    self.scan_input_directions[i]
                    if i < len(self.scan_input_directions)
                    else 0
                )
                # Reverse direction: 0 = forward, 1 = reverse
                idx = sequence_length - 1 - t if direction == 1 else t
                # Select along the axis
                elt = torch.select(scan_input, axis, idx)
                scan_input_elts.append(elt)

            # Call body: inputs are state_vars..., scan_input_elts..., outer_vals...
            body_inputs = current_state + scan_input_elts + outer_vals
            outputs = self.body(*body_inputs)

            # Handle single vs multiple outputs
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # First n_state_vars outputs are updated state
            current_state = list(outputs[: self.n_state_vars])

            # Remaining outputs are scan output elements
            for j in range(self.n_scan_outputs):
                scan_outputs[j].append(outputs[self.n_state_vars + j])

        # Prepare final outputs: state variables, then stacked scan outputs
        final_outputs = list(current_state)
        for j, scan_list in enumerate(scan_outputs):
            if scan_list:
                axis = self.scan_output_axes[j] if j < len(self.scan_output_axes) else 0
                direction = (
                    self.scan_output_directions[j]
                    if j < len(self.scan_output_directions)
                    else 0
                )
                # Stack along the specified axis
                stacked = torch.stack(scan_list, dim=axis)
                # Reverse if direction is 1 (prepending = reverse order)
                if direction == 1:
                    stacked = torch.flip(stacked, dims=[axis])
                final_outputs.append(stacked)
            else:
                # Empty scan output
                final_outputs.append(torch.tensor([]))

        return tuple(final_outputs)


@register("Scan", since_version=9)
def scan_op(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Scan operator (version 9+).

    Scan iterates over one or more scan_input tensors, constructing scan_output tensors.
    It combines ideas from general recurrences, functional programming constructs
    such as scan, fold, map, and zip.

    Inputs: initial_state_and_scan_inputs (variadic) - N state vars followed by M scan inputs
    Outputs: final_state_and_scan_outputs (variadic) - N final states followed by K scan outputs

    Attributes:
    - body: The graph run each iteration
    - num_scan_inputs: Number of scan inputs M
    - scan_input_axes: Axis to scan for each scan input (default: 0)
    - scan_input_directions: Direction for each scan input (0=forward, 1=reverse)
    - scan_output_axes: Axis for each scan output (default: 0)
    - scan_output_directions: Direction for each scan output (0=append, 1=prepend)
    """
    # Get body subgraph
    body_graph = get_attribute(node, "body")
    if body_graph is None:
        raise ValueError("Scan operator requires 'body' attribute")

    # Get num_scan_inputs attribute (required)
    num_scan_inputs = get_attribute(node, "num_scan_inputs")
    if num_scan_inputs is None:
        raise ValueError("Scan operator requires 'num_scan_inputs' attribute")

    # Get optional attributes
    scan_input_axes = get_attribute(node, "scan_input_axes") or []
    scan_input_directions = get_attribute(node, "scan_input_directions") or []
    scan_output_axes = get_attribute(node, "scan_output_axes") or []
    scan_output_directions = get_attribute(node, "scan_output_directions") or []

    # Parse inputs: first (len(node.input) - num_scan_inputs) are state variables
    n_state_vars = len(node.input) - num_scan_inputs

    state_inputs = [builder.get_value(node.input[i]) for i in range(n_state_vars)]
    scan_inputs = [
        builder.get_value(node.input[i]) for i in range(n_state_vars, len(node.input))
    ]

    # Build subgraph module
    body_module, body_input_names, body_output_names, outer_refs = (
        _build_subgraph_module(body_graph, builder.env, builder._opset_versions)
    )

    # Get outer scope values that the body references
    outer_values = [builder.get_value(name) for name in outer_refs]

    # Number of scan outputs
    n_scan_outputs = len(body_output_names) - n_state_vars

    # Create the scan module
    scan_module = ScanModule(
        body_module,
        n_state_vars,
        num_scan_inputs,
        n_scan_outputs,
        len(outer_values),
        list(scan_input_axes),
        list(scan_output_axes),
        list(scan_input_directions),
        list(scan_output_directions),
    )

    # Register the scan module
    module_name = builder.register_submodule(f"scan_{node.name or 'op'}", scan_module)

    # Build args: state_vars..., scan_inputs..., outer_vals...
    args = state_inputs + scan_inputs + outer_values

    result = builder.call_module(module_name, args=tuple(args))

    return result


@register("Scan", since_version=8)
def scan_op_v8(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Scan operator (version 8).

    Version 8 has batching support and different input format:
    - First input is optional sequence_lens
    - Requires batch dimension (axis 0) and sequence dimension (axis 1)

    For simplicity, this implementation handles the common case where
    sequence_lens is empty (all sequences have same length).
    """
    # Get body subgraph
    body_graph = get_attribute(node, "body")
    if body_graph is None:
        raise ValueError("Scan operator requires 'body' attribute")

    # Get num_scan_inputs attribute (required)
    num_scan_inputs = get_attribute(node, "num_scan_inputs")
    if num_scan_inputs is None:
        raise ValueError("Scan operator requires 'num_scan_inputs' attribute")

    # Get optional directions attribute (v8 uses 'directions' instead of scan_input_directions)
    directions = get_attribute(node, "directions") or []

    # In v8, first input is optional sequence_lens, rest are state vars + scan inputs
    # Check if first input is empty (sequence_lens is optional)
    has_sequence_lens = node.input[0] != ""

    if has_sequence_lens:
        # sequence_lens provided - we ignore it for now (assume fixed length)
        start_idx = 1
    else:
        start_idx = 1  # Skip the empty sequence_lens input

    # Parse remaining inputs: state variables followed by scan inputs
    remaining_inputs = list(node.input[start_idx:])
    n_state_vars = len(remaining_inputs) - num_scan_inputs

    state_inputs = [builder.get_value(remaining_inputs[i]) for i in range(n_state_vars)]
    scan_inputs = [
        builder.get_value(remaining_inputs[i])
        for i in range(n_state_vars, len(remaining_inputs))
    ]

    # Build subgraph module
    body_module, body_input_names, body_output_names, outer_refs = (
        _build_subgraph_module(body_graph, builder.env, builder._opset_versions)
    )

    # Get outer scope values that the body references
    outer_values = [builder.get_value(name) for name in outer_refs]

    # Number of scan outputs
    n_scan_outputs = len(body_output_names) - n_state_vars

    # Create the scan module for v8 (handles batching)
    # Note: In v8, batch axis is 0 and sequence axis is 1 (handled in ScanModuleV8)
    scan_module = ScanModuleV8(
        body_module,
        n_state_vars,
        num_scan_inputs,
        n_scan_outputs,
        len(outer_values),
        list(directions),
    )

    # Register the scan module
    module_name = builder.register_submodule(
        f"scan_v8_{node.name or 'op'}", scan_module
    )

    # Build args: state_vars..., scan_inputs..., outer_vals...
    args = state_inputs + scan_inputs + outer_values

    result = builder.call_module(module_name, args=tuple(args))

    return result


class ScanModuleV8(nn.Module):
    """Module that executes an ONNX Scan operation (version 8 with batching)."""

    def __init__(
        self,
        body_module: torch.fx.GraphModule,
        n_state_vars: int,
        n_scan_inputs: int,
        n_scan_outputs: int,
        n_outer_vars: int,
        directions: List[int],
    ):
        super().__init__()
        self.body = body_module
        self.n_state_vars = n_state_vars
        self.n_scan_inputs = n_scan_inputs
        self.n_scan_outputs = n_scan_outputs
        self.n_outer_vars = n_outer_vars
        self.directions = directions

    def forward(self, *args) -> Tuple[torch.Tensor, ...]:
        """Execute the scan with batching.

        In v8, tensors have shape [batch, sequence, ...].
        State variables have shape [batch, ...].

        Args are: state_vars..., scan_inputs..., outer_vals...
        Returns final state variables followed by scan outputs.
        """
        # Split args
        state_vars = list(args[: self.n_state_vars])
        scan_inputs = list(
            args[self.n_state_vars : self.n_state_vars + self.n_scan_inputs]
        )
        outer_vals = list(args[self.n_state_vars + self.n_scan_inputs :])

        # Get batch size and sequence length from first scan input
        if self.n_scan_inputs > 0:
            first_input = scan_inputs[0]
            batch_size = first_input.shape[0]
            sequence_length = first_input.shape[1]
        else:
            batch_size = state_vars[0].shape[0] if state_vars else 1
            sequence_length = 0

        # Process each batch
        batch_final_states: List[List[torch.Tensor]] = [
            [] for _ in range(self.n_state_vars)
        ]
        batch_scan_outputs: List[List[torch.Tensor]] = [
            [] for _ in range(self.n_scan_outputs)
        ]

        for batch in range(batch_size):
            # Get batch slice of state variables
            current_state = [sv[batch] for sv in state_vars]

            # Initialize scan output accumulators for this batch
            scan_outputs: List[List[torch.Tensor]] = [
                [] for _ in range(self.n_scan_outputs)
            ]

            # Execute loop over sequence
            for t in range(sequence_length):
                # Extract scan input elements for this batch and time step
                scan_input_elts = []
                for i, scan_input in enumerate(scan_inputs):
                    direction = self.directions[i] if i < len(self.directions) else 0
                    idx = sequence_length - 1 - t if direction == 1 else t
                    # scan_input has shape [batch, sequence, ...]
                    elt = scan_input[batch, idx]
                    scan_input_elts.append(elt)

                # Call body
                body_inputs = current_state + scan_input_elts + outer_vals
                outputs = self.body(*body_inputs)

                if not isinstance(outputs, tuple):
                    outputs = (outputs,)

                # Update state
                current_state = list(outputs[: self.n_state_vars])

                # Collect scan outputs
                for j in range(self.n_scan_outputs):
                    scan_outputs[j].append(outputs[self.n_state_vars + j])

            # Store final state for this batch
            for i, state in enumerate(current_state):
                batch_final_states[i].append(state)

            # Stack scan outputs for this batch
            for j, scan_list in enumerate(scan_outputs):
                if scan_list:
                    stacked = torch.stack(scan_list, dim=0)
                    batch_scan_outputs[j].append(stacked)

        # Stack across batches
        final_outputs: List[torch.Tensor] = []

        # Final state variables: stack across batch
        for states in batch_final_states:
            final_outputs.append(torch.stack(states, dim=0))

        # Scan outputs: stack across batch
        for outputs in batch_scan_outputs:
            if outputs:
                final_outputs.append(torch.stack(outputs, dim=0))

        return tuple(final_outputs)


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
        raise ValueError(
            "If operator requires 'then_branch' and 'else_branch' attributes"
        )

    # Build subgraph modules for both branches
    then_module, then_input_names, then_output_names, then_outer_refs = (
        _build_subgraph_module(then_graph, builder.env, builder._opset_versions)
    )
    else_module, else_input_names, else_output_names, else_outer_refs = (
        _build_subgraph_module(else_graph, builder.env, builder._opset_versions)
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
