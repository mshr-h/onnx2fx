# SPDX-License-Identifier: Apache-2.0
"""Helpers for ONNX external data tensors."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import onnx

from ..exceptions import ExternalDataError, UnsupportedDTypeError
from .dtype import DTYPE_MAP


def _load_onnx_mapping() -> Any:  # pragma: no cover - version-dependent
    try:
        return importlib.import_module("onnx.mapping")
    except Exception:
        return importlib.import_module("onnx._mapping")


onnx_mapping: Any = _load_onnx_mapping()


@dataclass(frozen=True)
class ExternalDataInfo:
    """Resolved external data metadata."""

    path: str
    offset: int
    length: int
    shape: Tuple[int, ...]
    numpy_dtype: np.dtype


def _tensor_name(tensor: onnx.TensorProto) -> str:
    return tensor.name or "<unnamed>"


def _get_numpy_dtype(onnx_dtype: int, *, tensor_name: str) -> np.dtype:
    if DTYPE_MAP.get(onnx_dtype) is None:
        raise UnsupportedDTypeError(
            onnx_dtype=onnx_dtype,
            tensor_name=tensor_name,
            details="dtype not supported by onnx2fx",
        )
    np_type = onnx_mapping.TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype)
    if np_type is None:
        raise UnsupportedDTypeError(
            onnx_dtype=onnx_dtype,
            tensor_name=tensor_name,
            details="dtype has no NumPy mapping",
        )
    try:
        return np.dtype(np_type)
    except Exception as exc:  # pragma: no cover - defensive
        raise UnsupportedDTypeError(
            onnx_dtype=onnx_dtype,
            tensor_name=tensor_name,
            details=f"dtype not supported by NumPy memmap ({exc})",
        ) from exc


def _parse_external_data_kv(
    tensor: onnx.TensorProto, *, tensor_name: str
) -> Dict[str, str]:
    data = {entry.key: entry.value for entry in tensor.external_data}
    if not data:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message="external data metadata is empty",
        )
    return data


def _require_int_field(
    data: Dict[str, str],
    *,
    field: str,
    tensor_name: str,
) -> int:
    if field not in data:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message=f"missing external data field '{field}'",
        )
    try:
        return int(data[field])
    except ValueError as exc:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message=f"invalid external data field '{field}': {data[field]}",
        ) from exc


def _resolve_external_path(
    location: str, base_dir: str | None, *, tensor_name: str
) -> str:
    if os.path.isabs(location):
        return location
    if not base_dir:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message="base_dir is required for relative external data paths",
        )
    return os.path.normpath(os.path.join(base_dir, location))


def _expected_nbytes(shape: Iterable[int], np_dtype: np.dtype) -> int:
    dims = list(shape)
    if len(dims) == 0:
        element_count = 1
    else:
        element_count = math.prod(dims)
    return int(element_count) * int(np_dtype.itemsize)


def resolve_external_data(
    tensor: onnx.TensorProto,
    *,
    base_dir: str | None,
    strict: bool = True,
) -> ExternalDataInfo:
    tensor_name = _tensor_name(tensor)
    data = _parse_external_data_kv(tensor, tensor_name=tensor_name)
    if "location" not in data:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message="missing external data field 'location'",
        )
    location = data["location"]
    offset = _require_int_field(data, field="offset", tensor_name=tensor_name)
    length = _require_int_field(data, field="length", tensor_name=tensor_name)
    if offset < 0 or length < 0:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message="external data offset/length must be non-negative",
        )

    path = _resolve_external_path(location, base_dir, tensor_name=tensor_name)
    if not os.path.isfile(path):
        raise ExternalDataError(
            tensor_name=tensor_name,
            message=f"external data file not found: {path}",
        )

    np_dtype = _get_numpy_dtype(tensor.data_type, tensor_name=tensor_name)
    expected = _expected_nbytes(tensor.dims, np_dtype)
    file_size = os.path.getsize(path)

    if offset + length > file_size:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message=(
                "external data range exceeds file size "
                f"(offset={offset}, length={length}, file_size={file_size})"
            ),
        )

    if strict and length != expected:
        raise ExternalDataError(
            tensor_name=tensor_name,
            message=(
                f"external data length mismatch (length={length}, expected={expected})"
            ),
        )

    return ExternalDataInfo(
        path=path,
        offset=offset,
        length=length,
        shape=tuple(int(dim) for dim in tensor.dims),
        numpy_dtype=np_dtype,
    )


def iter_all_graphs(model: onnx.ModelProto) -> Iterable[onnx.GraphProto]:
    """Yield all graphs, including nested subgraphs."""
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        yield graph
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    graphs.append(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    graphs.extend(list(attr.graphs))


def validate_external_data_model(
    model: onnx.ModelProto,
    *,
    base_dir: str | None,
    strict: bool = True,
) -> None:
    """Validate external data metadata for all tensors in a model."""
    for graph in iter_all_graphs(model):
        for tensor in graph.initializer:
            if (
                tensor.data_location == onnx.TensorProto.EXTERNAL
                or tensor.external_data
            ):
                resolve_external_data(tensor, base_dir=base_dir, strict=strict)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor = attr.t
                    if (
                        tensor.data_location == onnx.TensorProto.EXTERNAL
                        or tensor.external_data
                    ):
                        resolve_external_data(tensor, base_dir=base_dir, strict=strict)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for tensor in attr.tensors:
                        if (
                            tensor.data_location == onnx.TensorProto.EXTERNAL
                            or tensor.external_data
                        ):
                            resolve_external_data(
                                tensor, base_dir=base_dir, strict=strict
                            )


__all__ = [
    "ExternalDataInfo",
    "resolve_external_data",
    "validate_external_data_model",
]
