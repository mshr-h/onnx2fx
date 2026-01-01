# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Tuple, Sequence

import torch
import onnx
from onnx import numpy_helper

from .op_registry import OP_REGISTRY, get_handler
from .utils.dtype import DTYPE_MAP

# Import ops module to register all operators
from . import ops  # noqa: F401


class GraphBuilder:
    def __init__(self, model: onnx.ModelProto):
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        self.model: onnx.ModelProto = model
        self.graph: torch.fx.Graph = torch.fx.Graph()
        self.value_info_map = self._create_value_info_map()
        self.initializer_map = self._create_initializer_map()
        self.input_names: List[str] = []
        self.env: Dict[str, torch.fx.Node] = {}
        self._constants: Dict[str, torch.Tensor] = {}

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
            raise KeyError(f"Value '{name}' not found in environment. "
                          f"Available: {list(self.env.keys())}")
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

            placeholder = self.graph.placeholder(value.name)
            info = self.value_info_map.get(value.name)
            placeholder.meta["onnx_shape"] = info[0] if info else None
            placeholder.meta["onnx_dtype"] = info[1] if info else None
            placeholder.meta["onnx_op_type"] = "Input"
            placeholder.meta["onnx_name"] = value.name
            self.env[value.name] = placeholder
            self.input_names.append(value.name)

    def _convert_nodes(self) -> None:
        for node in self.model.graph.node:
            # Get handler with domain support
            domain = node.domain if node.domain else ""
            handler = get_handler(node.op_type, domain)
            if handler is None:
                domain_str = f" (domain: {domain})" if domain else ""
                raise NotImplementedError(
                    f"Unsupported ONNX op type: {node.op_type}{domain_str}"
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
                            lambda x, idx=i: x[idx] if isinstance(x, (tuple, list)) else x,
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
