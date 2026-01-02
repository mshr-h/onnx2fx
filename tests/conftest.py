# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and utilities for onnx2fx tests."""

import pytest
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


def opset_id(opset):
    """Helper function for parametrize ids."""
    return f"opset{opset.version}"
