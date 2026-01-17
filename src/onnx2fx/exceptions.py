# SPDX-License-Identifier: Apache-2.0
"""Custom exceptions for onnx2fx."""


class Onnx2FxError(Exception):
    """Base exception for all onnx2fx errors.

    This is the parent class for all custom exceptions in onnx2fx.
    Users can catch this exception to handle any onnx2fx-related error.
    """

    pass


class UnsupportedOpError(Onnx2FxError):
    """Raised when an ONNX operator is not supported.

    This exception is raised during conversion when the converter
    encounters an ONNX operator that has no registered handler.

    Parameters
    ----------
    op_type : str
        The unsupported ONNX operator type.
    domain : str, optional
        The ONNX domain of the operator.
    opset_version : int, optional
        The opset version being used.
    """

    def __init__(
        self,
        op_type: str,
        domain: str = "",
        opset_version: int | None = None,
    ):
        self.op_type = op_type
        self.domain = domain
        self.opset_version = opset_version

        domain_str = f" (domain: {domain})" if domain else ""
        version_str = f" at opset {opset_version}" if opset_version else ""
        message = f"Unsupported ONNX operator: {op_type}{domain_str}{version_str}"
        super().__init__(message)


class ConversionError(Onnx2FxError):
    """Raised when conversion fails due to an error in the conversion process.

    This exception is raised when the conversion process encounters
    an error that prevents successful completion.

    Parameters
    ----------
    message : str
        A description of the conversion error.
    node_name : str, optional
        The name of the ONNX node where the error occurred.
    op_type : str, optional
        The ONNX operator type where the error occurred.
    """

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        op_type: str | None = None,
    ):
        self.node_name = node_name
        self.op_type = op_type

        context_parts = []
        if node_name:
            context_parts.append(f"node: {node_name}")
        if op_type:
            context_parts.append(f"op: {op_type}")

        if context_parts:
            context = f" [{', '.join(context_parts)}]"
        else:
            context = ""

        full_message = f"Conversion failed{context}: {message}"
        super().__init__(full_message)


class ValueNotFoundError(Onnx2FxError):
    """Raised when a value is not found in the environment.

    This exception is raised when trying to access a tensor value
    that has not been defined in the conversion environment.

    Parameters
    ----------
    name : str
        The name of the value that was not found.
    available : list[str], optional
        List of available value names for debugging.
    """

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available

        message = f"Value '{name}' not found in environment"
        if available:
            message += f". Available: {available}"
        super().__init__(message)
