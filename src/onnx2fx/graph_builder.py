# SPDX-License-Identifier: Apache-2.0
from collections import deque
from typing import Dict, List, Optional, Tuple, Sequence

import torch
import onnx
from onnx import numpy_helper

from .exceptions import UnsupportedOpError, ValueNotFoundError
from .op_registry import get_handler
from .utils.dtype import DTYPE_MAP

# Import ops module to register all operators
from . import ops  # noqa: F401


def _topological_sort(
    nodes: List[onnx.NodeProto],
    graph_inputs: set,
    initializers: set,
) -> List[onnx.NodeProto]:
    """Topologically sort ONNX graph nodes using Kahn's algorithm.

    Some ONNX models have nodes in non-topological order (e.g., Cast nodes
    at the end of the graph but their outputs used earlier). This function
    reorders nodes so dependencies are processed before their consumers.

    Parameters
    ----------
    nodes : List[onnx.NodeProto]
        The list of ONNX nodes to sort.
    graph_inputs : set
        Set of graph input names.
    initializers : set
        Set of initializer names.

    Returns
    -------
    List[onnx.NodeProto]
        Topologically sorted list of nodes.
    """
    if not nodes:
        return []

    # Build output->node mapping (which node produces each output)
    output_to_node: Dict[str, onnx.NodeProto] = {}
    for node in nodes:
        for output in node.output:
            if output:  # Skip empty outputs
                output_to_node[output] = node

    # Available values: graph inputs + initializers
    available = graph_inputs | initializers

    # Compute in-degree for each node (number of unsatisfied dependencies)
    in_degree: Dict[int, int] = {}
    node_id: Dict[int, onnx.NodeProto] = {}
    for i, node in enumerate(nodes):
        node_id[id(node)] = node
        # Count inputs that are neither available nor empty
        deps = 0
        for inp in node.input:
            if inp and inp not in available:
                deps += 1
        in_degree[id(node)] = deps

    # Initialize queue with nodes that have no dependencies
    queue = deque()
    for node in nodes:
        if in_degree[id(node)] == 0:
            queue.append(node)

    sorted_nodes: List[onnx.NodeProto] = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)

        # Mark this node's outputs as available
        for output in node.output:
            if output:
                available.add(output)

        # Reduce in-degree for nodes that depend on this node's outputs
        for candidate in nodes:
            if in_degree[id(candidate)] > 0:
                # Check if any of candidate's inputs are now satisfied
                for inp in candidate.input:
                    if inp in node.output and inp:
                        in_degree[id(candidate)] -= 1
                        if in_degree[id(candidate)] == 0:
                            queue.append(candidate)

    # If we couldn't sort all nodes, there's a cycle or missing dependency
    # Fall back to original order
    if len(sorted_nodes) != len(nodes):
        return list(nodes)

    return sorted_nodes


class GraphBuilder:
    def __init__(self, model: onnx.ModelProto):
        # Try shape inference but preserve original model if it fails
        # (shape_inference may drop graph contents for large models with external data)
        try:
            inferred_model = onnx.shape_inference.infer_shapes(model)
            # Check if shape inference preserved the model structure
            if len(inferred_model.graph.node) > 0:
                model = inferred_model
            # If nodes were lost, keep original model
        except Exception:
            pass
        self.model: onnx.ModelProto = model
        self.graph: torch.fx.Graph = torch.fx.Graph()
        self.value_info_map = self._create_value_info_map()
        self.initializer_map = self._create_initializer_map()
        self.input_names: List[str] = []
        self.env: Dict[str, torch.fx.Node] = {}
        self._constants: Dict[str, torch.Tensor] = {}
        self._opset_versions: Dict[str, int] = self._extract_opset_versions()

    def _extract_opset_versions(self) -> Dict[str, int]:
        """Extract opset versions for all domains from the model.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping domain names to their opset versions.
            Empty string "" represents the default ONNX domain.
        """
        versions: Dict[str, int] = {}
        for opset in self.model.opset_import:
            domain = opset.domain if opset.domain else ""
            versions[domain] = opset.version
        return versions

    @property
    def opset_version(self) -> int:
        """Get the opset version for the default ONNX domain.

        Returns
        -------
        int
            The opset version number. Defaults to 1 if not specified.
        """
        return self._opset_versions.get("", 1)

    def get_opset_version(self, domain: str = "") -> int:
        """Get the opset version for a specific domain.

        Parameters
        ----------
        domain : str, optional
            The ONNX domain. Default is "" (standard ONNX domain).

        Returns
        -------
        int
            The opset version number. Defaults to 1 if not specified.
        """
        return self._opset_versions.get(domain, 1)

    def build(self) -> torch.fx.GraphModule:
        self._load_initializers()
        self._create_placeholders()
        self._convert_nodes()
        self._create_outputs()
        root_module = torch.nn.Module()
        # Register constants as buffers
        for name, tensor in self._constants.items():
            root_module.register_buffer(name.replace(".", "_"), tensor)
        module = torch.fx.GraphModule(root_module, self.graph)
        module.graph.lint()
        return module

    def get_value(self, name: str) -> torch.fx.Node:
        """Get a value (node) by name from the environment.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        torch.fx.Node
            The corresponding FX node.

        Raises
        ------
        KeyError
            If the name is not found in the environment.
        """
        if name not in self.env:
            raise ValueNotFoundError(name, available=list(self.env.keys()))
        return self.env[name]

    def has_value(self, name: str) -> bool:
        """Check if a value exists in the environment."""
        return name in self.env

    def call_function(
        self,
        func,
        args: Sequence[torch.fx.Node | object] = (),
        kwargs: Dict[str, object] = {},
    ) -> torch.fx.Node:
        fx_node = self.graph.call_function(func, args=tuple(args), kwargs=kwargs or {})
        return fx_node

    def _create_value_info_map(
        self,
    ) -> Dict[str, Tuple[Optional[List[Optional[int]]], Optional[torch.dtype]]]:
        """Build a mapping from value names to their shape and dtype info."""

        def extract_tensor_shape(
            value: onnx.ValueInfoProto,
        ) -> Optional[List[Optional[int]]]:
            """Extract a list-based representation of a tensor shape from a value info."""

            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                return None
            dims: List[Optional[int]] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(None)
                else:
                    dims.append(None)
            return dims

        def extract_tensor_dtype(value: onnx.ValueInfoProto) -> Optional[torch.dtype]:
            """Extract the Torch dtype that corresponds to a value info."""

            return DTYPE_MAP.get(value.type.tensor_type.elem_type)

        info_map = {}
        for value_info in (
            list(self.model.graph.input)
            + list(self.model.graph.value_info)
            + list(self.model.graph.output)
        ):
            info_map[value_info.name] = (
                extract_tensor_shape(value_info),
                extract_tensor_dtype(value_info),
            )
        return info_map

    def _create_initializer_map(self) -> Dict[str, torch.Tensor]:
        """Build a mapping from initializer names to PyTorch tensors."""
        init_map = {}
        for initializer in self.model.graph.initializer:
            np_array = numpy_helper.to_array(initializer)
            init_map[initializer.name] = torch.from_numpy(np_array.copy())
        return init_map

    def _load_initializers(self) -> None:
        """Load ONNX initializers as constant nodes in the FX graph."""
        for name, tensor in self.initializer_map.items():
            # Store in constants dict for later registration as buffers
            safe_name = name.replace(".", "_").replace("/", "_")
            self._constants[safe_name] = tensor

            # Create a get_attr node to access the buffer
            fx_node = self.graph.get_attr(safe_name)
            fx_node.meta["onnx_op_type"] = "Initializer"
            fx_node.meta["onnx_name"] = name
            fx_node.meta["onnx_shape"] = list(tensor.shape)
            fx_node.meta["onnx_dtype"] = tensor.dtype
            self.env[name] = fx_node

    def _create_placeholders(self) -> None:
        """Create FX placeholder nodes for graph inputs.

        Note: Inputs that are already loaded as initializers are skipped.
        """
        for value in self.model.graph.input:
            # Skip if already loaded as initializer
            if value.name in self.env:
                continue

            # Sanitize name for valid Python identifier
            safe_name = value.name.replace(".", "_").replace("/", "_").replace("-", "_")
            placeholder = self.graph.placeholder(safe_name)
            info = self.value_info_map.get(value.name)
            placeholder.meta["onnx_shape"] = info[0] if info else None
            placeholder.meta["onnx_dtype"] = info[1] if info else None
            placeholder.meta["onnx_op_type"] = "Input"
            placeholder.meta["onnx_name"] = value.name
            self.env[value.name] = placeholder
            self.input_names.append(value.name)

    def _convert_nodes(self) -> None:
        # Get graph inputs and initializers for topological sort
        graph_inputs = {inp.name for inp in self.model.graph.input}
        initializers = set(self.initializer_map.keys())

        # Topologically sort nodes to handle out-of-order dependencies
        sorted_nodes = _topological_sort(
            list(self.model.graph.node),
            graph_inputs,
            initializers,
        )

        for node in sorted_nodes:
            # Get handler with domain and opset version support
            domain = node.domain if node.domain else ""
            opset = self.get_opset_version(domain)
            handler = get_handler(node.op_type, domain, opset)
            if handler is None:
                raise UnsupportedOpError(
                    node.op_type, domain=domain, opset_version=opset
                )
            fx_node = handler(self, node)

            # Handle multiple outputs
            if len(node.output) == 1:
                self.env[node.output[0]] = fx_node
            else:
                # For multi-output nodes, fx_node should be a tuple or we extract via getitem
                for i, output_name in enumerate(node.output):
                    if output_name:  # Skip empty output names
                        # Create a getitem node to extract each output
                        getitem_node = self.graph.call_function(
                            lambda x, idx=i: x[idx]
                            if isinstance(x, (tuple, list))
                            else x,
                            args=(fx_node, i),
                        )
                        self.env[output_name] = getitem_node

    def _create_outputs(self) -> None:
        output_nodes = [self.get_value(value.name) for value in self.model.graph.output]
        if len(output_nodes) == 1:
            self.graph.output(output_nodes[0])
        else:
            self.graph.output(tuple(output_nodes))


__all__ = ["GraphBuilder"]
