# SPDX-License-Identifier: Apache-2.0
"""ONNX operator registry."""

from typing import TYPE_CHECKING, Dict, Callable

import onnx

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .graph_builder import GraphBuilder

OpHandler = Callable[["GraphBuilder", onnx.NodeProto], object]

OP_REGISTRY: Dict[str, OpHandler] = {}


def register(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an ONNX operator handler.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name (e.g., "Add", "Relu").

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func: OpHandler) -> OpHandler:
        OP_REGISTRY[op_type] = func
        return func

    return decorator


def get_supported_ops() -> list:
    """Get list of supported ONNX operators.

    Returns
    -------
    list
        Sorted list of supported operator names.
    """
    return sorted(OP_REGISTRY.keys())
