# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Tuple, Sequence

import torch
import onnx

from .op_registry import OP_REGISTRY


_DTYPE_MAP: Dict[int, torch.dtype] = {
    onnx.TensorProto.FLOAT: torch.float32,
}


class GraphBuilder:
    def __init__(self, model: onnx.ModelProto):
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        self.model: onnx.ModelProto = model
        self.graph: torch.fx.Graph = torch.fx.Graph()
        self.value_info_map = self._create_value_info_map()
        self.input_names: List[str] = []
        self.env: Dict[str, torch.fx.Node] = {}

    def build(self) -> torch.fx.GraphModule:
        self._create_placeholders()
        self._convert_nodes()
        self._create_outputs()
        module = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        module.graph.lint()
        return module

    def get_value(self, name: str) -> torch.fx.Node:
        return self.env[name]

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

            return _DTYPE_MAP.get(value.type.tensor_type.elem_type)

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

    def _create_placeholders(self) -> None:
        for value in self.model.graph.input:
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
            handler = OP_REGISTRY.get(node.op_type)
            if handler is None:
                raise NotImplementedError(f"Unsupported ONNX op type: {node.op_type}")
            fx_node = handler(self, node)

            # TODO: Handle multiple outputs (Sequence, Dict)
            self.env[node.output[0]] = fx_node

    def _create_outputs(self) -> None:
        output_nodes = [self.get_value(value.name) for value in self.model.graph.output]
        if len(output_nodes) == 1:
            self.graph.output(output_nodes[0])
        else:
            self.graph.output(tuple(output_nodes))


__all__ = ["GraphBuilder"]
