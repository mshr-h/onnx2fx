# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and utilities for onnx2fx tests."""

from typing import Any, Callable, Dict, List, Tuple, Union
import warnings
import tempfile

import onnx
import pytest
import torch
import numpy as np

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


# Auto-use fixtures for global test setup
@pytest.fixture(autouse=True)
def reset_warnings():
    """Reset warnings for each test to ensure clean warning state.

    Inspired by datasette-enrichments' autouse fixture pattern.
    This ensures each test starts with a clean warning state.
    """
    warnings.simplefilter("default")
    yield
    warnings.resetwarnings()


@pytest.fixture(autouse=True)
def deterministic_mode():
    """Set deterministic mode for reproducible tests.

    Auto-use fixture that ensures torch operations are deterministic
    for consistent test results across runs.
    """
    # Store original state
    original_deterministic = torch.are_deterministic_algorithms_enabled()

    # Enable deterministic mode
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    yield

    # Restore original state
    torch.use_deterministic_algorithms(mode=original_deterministic, warn_only=True)


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
    """Helper function for parametrize ids.

    Examples
    --------
    >>> @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    >>> def test_something(opset):
    >>>     ...
    """
    return f"opset{opset.version}"


def assert_model_shapes(
    fx_model: torch.fx.GraphModule,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    expected_output_shape: Union[torch.Size, Tuple[torch.Size, ...]],
) -> torch.Tensor:
    """Helper to assert only output shapes match.

    Useful for testing operators with random or non-deterministic outputs.

    Parameters
    ----------
    fx_model : torch.fx.GraphModule
        The converted FX model to test.
    inputs : Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]
        Input tensor(s) for the model.
    expected_output_shape : Union[torch.Size, Tuple[torch.Size, ...]]
        Expected output shape(s).

    Returns
    -------
    torch.Tensor
        The actual model output (for further inspection if needed).
    """
    # Normalize inputs to tuple
    if isinstance(inputs, torch.Tensor):
        input_tuple = (inputs,)
    elif isinstance(inputs, dict):
        input_tuple = inputs
    else:
        input_tuple = inputs

    # Run inference
    with torch.inference_mode():
        if isinstance(input_tuple, dict):
            result = fx_model(**input_tuple)
        else:
            result = fx_model(*input_tuple)

    # Check shapes
    if isinstance(result, (tuple, list)):
        if not isinstance(expected_output_shape, (tuple, list)):
            expected_output_shape = (expected_output_shape,)
        assert len(result) == len(expected_output_shape), (
            f"Number of outputs mismatch: {len(result)} vs {len(expected_output_shape)}"
        )
        for i, (r, e) in enumerate(zip(result, expected_output_shape)):
            assert r.shape == e, f"Output {i} shape mismatch: {r.shape} vs {e}"
    else:
        assert result.shape == expected_output_shape, (
            f"Shape mismatch: {result.shape} vs {expected_output_shape}"
        )

    return result


def create_simple_model(
    opset_version: int = 23,
    *,
    input_shapes: Tuple[Tuple[int, ...], ...] = ((1, 3, 224, 224),),
    input_names: Tuple[str, ...] = ("input",),
    output_names: Tuple[str, ...] = ("output",),
) -> onnx.ModelProto:
    """Create a simple ONNX model for testing.

    Helper function to quickly create test models with specified shapes.
    Inspired by datasette-enrichments' test helper patterns.

    Parameters
    ----------
    opset_version : int
        ONNX opset version to use.
    input_shapes : Tuple[Tuple[int, ...], ...]
        Shapes for input tensors.
    input_names : Tuple[str, ...]
        Names for input tensors.
    output_names : Tuple[str, ...]
        Names for output tensors.

    Returns
    -------
    onnx.ModelProto
        A simple ONNX model (identity operation).
    """
    import onnx.helper as helper

    # Create input/output value infos
    inputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(shape))
        for name, shape in zip(input_names, input_shapes)
    ]

    outputs = [
        helper.make_tensor_value_info(
            name,
            onnx.TensorProto.FLOAT,
            None,  # Dynamic shape
        )
        for name in output_names
    ]

    # Create simple identity node
    node = helper.make_node(
        "Identity", inputs=[input_names[0]], outputs=[output_names[0]]
    )

    # Create graph
    graph = helper.make_graph(
        [node],
        "test_graph",
        inputs,
        outputs,
    )

    # Create model
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    return model


def run_onnx_test(
    model: Union[onnx.ModelProto, Callable[[], onnx.ModelProto], torch.fx.GraphModule],
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
    output_transform: Callable[[Any], Any] | None = None,
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
    # Allow passing an already converted module to avoid repeated conversions
    # in loop-heavy tests.
    if isinstance(model, torch.fx.GraphModule):
        fx_model = model
    else:
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
        with torch.inference_mode():
            if isinstance(input_tuple, dict):
                expected = expected(**input_tuple)
            else:
                expected = expected(*input_tuple)

    # Some ONNX ops (notably control-flow) may return a 1-tuple even for a
    # single logical output. Normalize that case to keep tests concise.
    if (
        isinstance(result, (tuple, list))
        and len(result) == 1
        and not isinstance(expected, (tuple, list))
    ):
        result = result[0]

    if output_transform is not None:
        result = output_transform(result)
        expected = output_transform(expected)

    # Assert
    if check_shape_only:
        if isinstance(result, (tuple, list)):
            assert len(result) == len(expected)
            for r, e in zip(result, expected):
                assert r.shape == e.shape, f"Shape mismatch: {r.shape} vs {e.shape}"
        else:
            assert isinstance(result, torch.Tensor)
            assert isinstance(expected, torch.Tensor)
            assert result.shape == expected.shape, (
                f"Shape mismatch: {result.shape} vs {expected.shape}"
            )
    else:
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    return fx_model


def convert_onnx_model(
    model: Union[onnx.ModelProto, Callable[[], onnx.ModelProto]],
) -> torch.fx.GraphModule:
    """Convert an ONNX model to FX using the shared convert entry point.

    This helper standardizes conversion usage across tests.
    """
    if callable(model) and not isinstance(model, onnx.ModelProto):
        model = model()
    return convert(model)


def run_onnxruntime_iobinding(
    model: onnx.ModelProto,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    *,
    providers: List[str] | None = None,
):
    """Run ONNX Runtime using io_binding to minimize copies.

    Parameters
    ----------
    model : onnx.ModelProto
        The ONNX model to run.
    inputs : Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]
        Input tensor(s) - single tensor, tuple for positional, or dict for named.
    providers : List[str], optional
        ONNX Runtime execution providers.

    Returns
    -------
    list
        Outputs as NumPy arrays.
    """
    import onnxruntime as ort

    import os

    if isinstance(inputs, torch.Tensor):
        input_tuple = (inputs,)
    elif isinstance(inputs, dict):
        input_tuple = inputs
    else:
        input_tuple = inputs

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
    finally:
        # The session loads the model into memory, so we can delete the temp file.
        try:
            os.unlink(model_path)
        except OSError:
            pass

    io_binding = session.io_binding()

    if isinstance(input_tuple, dict):
        input_items = input_tuple.items()
    else:
        input_names = [i.name for i in session.get_inputs()]
        input_items = zip(input_names, input_tuple)

    for name, value in input_items:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)

        # onnxruntime 1.23+ API
        io_binding.bind_cpu_input(name, value)

    for output in session.get_outputs():
        io_binding.bind_output(output.name, "cpu")

    session.run_with_iobinding(io_binding)
    return io_binding.copy_outputs_to_cpu()
