from typing import TYPE_CHECKING, Dict, Callable
import onnx
import torch

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
