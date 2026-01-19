# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and utilities for onnx2fx tests."""

from typing import Callable, Dict, Tuple, Union

import onnx
import pytest
import torch

from onnx2fx import convert
from onnx2fx.op_registry import registry_context as _registry_context
from onnxscript import (
    opset11,
    opset12,
    opset13,
    opset14,
    opset15,
    opset16,
    opset17,
    opset18,
    opset19,
    opset20,
    opset21,
    opset22,
    opset23,
)

# Available opset modules for parametrized tests (opset 11-23)
OPSET_MODULES = [
    opset11,
    opset12,
    opset13,
    opset14,
    opset15,
    opset16,
    opset17,
    opset18,
    opset19,
    opset20,
    opset21,
    opset22,
    opset23,
]

# Opsets that support Einsum (12+)
EINSUM_OPSET_MODULES = [
    opset12,
    opset13,
    opset14,
    opset15,
    opset16,
    opset17,
    opset18,
    opset19,
    opset20,
    opset21,
    opset22,
    opset23,
]

# Default opset for non-parametrized tests
DEFAULT_OPSET = opset23


@pytest.fixture
def opset_modules():
    """Fixture providing all available opset modules."""
    return OPSET_MODULES


@pytest.fixture
def einsum_opset_modules():
    """Fixture providing opset modules that support Einsum (12+)."""
    return EINSUM_OPSET_MODULES


@pytest.fixture
def default_opset():
    """Fixture providing the default opset module."""
    return DEFAULT_OPSET


@pytest.fixture
def registry_context():
    """Isolate registry mutations within a test."""
    with _registry_context():
        yield


def opset_id(opset):
    """Helper function for parametrize ids."""
    return f"opset{opset.version}"


def run_onnx_test(
    model: Union[onnx.ModelProto, Callable[[], onnx.ModelProto]],
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    expected: Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    check_shape_only: bool = False,
) -> torch.fx.GraphModule:
    """Run a standard ONNX conversion test.

    This helper encapsulates the common happy-path test pattern:
    convert ONNX model → run inference → assert output matches expected.

    Parameters
    ----------
    model : Union[onnx.ModelProto, Callable[[], onnx.ModelProto]]
        The ONNX model or a callable that returns one (e.g., script.to_model_proto).
    inputs : Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]
        Input tensor(s) - single tensor, tuple for positional, or dict for named.
    expected : Union[Tensor, Tuple[Tensor, ...], Callable]
        Expected output, or a callable that computes expected from inputs.
    rtol : float
        Relative tolerance for assert_close.
    atol : float
        Absolute tolerance for assert_close.
    check_shape_only : bool
        If True, only check output shape (useful for random ops).

    Returns
    -------
    torch.fx.GraphModule
        The converted FX module (for additional assertions if needed).

    Examples
    --------
    >>> # Simple case with single input
    >>> run_onnx_test(relu_script.to_model_proto, x, torch.relu(x))
    >>>
    >>> # With callable expected
    >>> run_onnx_test(model, x, lambda x: torch.relu(x))
    >>>
    >>> # Multi-input
    >>> run_onnx_test(matmul_model, (a, b), torch.matmul(a, b))
    """
    # Get model if callable
    if callable(model) and not isinstance(model, onnx.ModelProto):
        model = model()

    # Convert ONNX to FX
    fx_model = convert(model)

    # Normalize inputs to tuple
    if isinstance(inputs, torch.Tensor):
        input_tuple = (inputs,)
    elif isinstance(inputs, dict):
        input_tuple = inputs  # Keep as dict for **kwargs
    else:
        input_tuple = inputs

    # Run inference
    with torch.inference_mode():
        if isinstance(input_tuple, dict):
            result = fx_model(**input_tuple)
        else:
            result = fx_model(*input_tuple)

    # Compute expected if callable
    if callable(expected):
        if isinstance(input_tuple, dict):
            expected = expected(**input_tuple)
        else:
            expected = expected(*input_tuple)

    # Assert
    if check_shape_only:
        if isinstance(result, (tuple, list)):
            assert len(result) == len(expected)
            for r, e in zip(result, expected):
                assert r.shape == e.shape, f"Shape mismatch: {r.shape} vs {e.shape}"
        else:
            assert result.shape == expected.shape, (
                f"Shape mismatch: {result.shape} vs {expected.shape}"
            )
    else:
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    return fx_model
