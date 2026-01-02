# SPDX-License-Identifier: Apache-2.0
"""Utilities for parsing ONNX node attributes."""

from typing import Any, Dict, Optional

import onnx


def get_attribute(
    node: onnx.NodeProto,
    name: str,
    default: Optional[Any] = None,
) -> Any:
    """Get a single attribute value from an ONNX node.

    Parameters
    ----------
    node : onnx.NodeProto
        The ONNX node.
    name : str
        The attribute name.
    default : Optional[Any]
        Default value if attribute is not found.

    Returns
    -------
    Any
        The attribute value, or default if not found.
    """
    for attr in node.attribute:
        if attr.name == name:
            return _parse_attribute_value(attr)
    return default


def get_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    """Get all attributes from an ONNX node as a dictionary.

    Parameters
    ----------
    node : onnx.NodeProto
        The ONNX node.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping attribute names to their values.
    """
    return {attr.name: _parse_attribute_value(attr) for attr in node.attribute}


def _parse_attribute_value(attr: onnx.AttributeProto) -> Any:
    """Parse an ONNX attribute into a Python value.

    Parameters
    ----------
    attr : onnx.AttributeProto
        The ONNX attribute.

    Returns
    -------
    Any
        The parsed Python value.
    """
    match attr.type:
        case onnx.AttributeProto.FLOAT:
            return attr.f
        case onnx.AttributeProto.INT:
            return attr.i
        case onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        case onnx.AttributeProto.TENSOR:
            return _parse_tensor(attr.t)
        case onnx.AttributeProto.GRAPH:
            return attr.g
        case onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        case onnx.AttributeProto.INTS:
            return list(attr.ints)
        case onnx.AttributeProto.STRINGS:
            return [
                s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings
            ]
        case onnx.AttributeProto.TENSORS:
            return [_parse_tensor(t) for t in attr.tensors]
        case onnx.AttributeProto.GRAPHS:
            return list(attr.graphs)
        case onnx.AttributeProto.SPARSE_TENSOR:
            return attr.sparse_tensor
        case onnx.AttributeProto.SPARSE_TENSORS:
            return list(attr.sparse_tensors)
        case onnx.AttributeProto.TYPE_PROTO:
            return attr.tp
        case onnx.AttributeProto.TYPE_PROTOS:
            return list(attr.type_protos)
        case _:
            raise ValueError(f"Unsupported attribute type: {attr.type}")


def _parse_tensor(tensor: onnx.TensorProto) -> "torch.Tensor":
    """Convert an ONNX TensorProto to a PyTorch tensor.

    Parameters
    ----------
    tensor : onnx.TensorProto
        The ONNX tensor.

    Returns
    -------
    torch.Tensor
        The converted PyTorch tensor.
    """
    import torch
    from onnx import numpy_helper

    np_array = numpy_helper.to_array(tensor)
    return torch.from_numpy(np_array.copy())


__all__ = [
    "get_attribute",
    "get_attributes",
]
