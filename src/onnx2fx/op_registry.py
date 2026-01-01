from typing import TYPE_CHECKING, Dict, Callable
import onnx
import torch

from .utils.attributes import get_attribute

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .graph_builder import GraphBuilder

OpHandler = Callable[["GraphBuilder", onnx.NodeProto], object]

OP_REGISTRY: Dict[str, OpHandler] = {}


def register(op_type: str) -> Callable[[OpHandler], OpHandler]:
    def decorator(func: OpHandler) -> OpHandler:
        OP_REGISTRY[op_type] = func
        return func

    return decorator


@register("Add")
def add(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(torch.add, args=(lhs, rhs))


@register("Constant")
def constant(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Handle Constant op by extracting value and creating a constant node."""
    # Try to get the value from different attribute types
    value = get_attribute(node, "value")
    if value is None:
        # Check for scalar attributes
        value_float = get_attribute(node, "value_float")
        if value_float is not None:
            value = torch.tensor(value_float, dtype=torch.float32)
        value_int = get_attribute(node, "value_int")
        if value_int is not None:
            value = torch.tensor(value_int, dtype=torch.int64)
        value_floats = get_attribute(node, "value_floats")
        if value_floats is not None:
            value = torch.tensor(value_floats, dtype=torch.float32)
        value_ints = get_attribute(node, "value_ints")
        if value_ints is not None:
            value = torch.tensor(value_ints, dtype=torch.int64)

    if value is None:
        raise ValueError(f"Constant node {node.name} has no value attribute")

    # Store as a constant in the builder
    output_name = node.output[0]
    safe_name = output_name.replace(".", "_").replace("/", "_")
    builder._constants[safe_name] = value

    # Create a get_attr node
    fx_node = builder.graph.get_attr(safe_name)
    fx_node.meta["onnx_op_type"] = "Constant"
    fx_node.meta["onnx_name"] = output_name
    fx_node.meta["onnx_shape"] = list(value.shape) if hasattr(value, "shape") else []
    fx_node.meta["onnx_dtype"] = value.dtype if hasattr(value, "dtype") else None
    return fx_node
