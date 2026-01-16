# SPDX-License-Identifier: Apache-2.0
"""ONNX operator registry with custom operator and opset version support."""

from typing import TYPE_CHECKING, Dict, Callable, Optional, Union, List, Tuple

import onnx

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .graph_builder import GraphBuilder

OpHandler = Callable[["GraphBuilder", onnx.NodeProto], object]

# Registry: {domain: {op_type: [(since_version, handler), ...]}}
# Handlers are stored in descending version order for efficient lookup.
# Empty string "" represents the default ONNX domain.
_VERSIONED_REGISTRY: Dict[str, Dict[str, List[Tuple[int, OpHandler]]]] = {"": {}}


def register(
    op_type: str, domain: str = "", since_version: int = 1
) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an ONNX operator handler with version support.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name (e.g., "Add", "Relu").
    domain : str, optional
        The ONNX domain (e.g., "com.microsoft"). Default is "" (standard ONNX domain).
    since_version : int, optional
        The minimum opset version this handler supports. Default is 1.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    Register a standard ONNX operator (all versions):

    >>> @register("MyOp")
    ... def my_op(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(torch.relu, args=(x,))

    Register version-specific handlers:

    >>> @register("Softmax", since_version=1)
    ... def softmax_v1(builder, node):
    ...     # opset 1-12: axis defaults to 1
    ...     ...

    >>> @register("Softmax", since_version=13)
    ... def softmax_v13(builder, node):
    ...     # opset 13+: axis defaults to -1
    ...     ...

    Register a custom domain operator:

    >>> @register("CustomOp", domain="com.mycompany")
    ... def custom_op(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(my_custom_function, args=(x,))
    """

    def decorator(func: OpHandler) -> OpHandler:
        if domain not in _VERSIONED_REGISTRY:
            _VERSIONED_REGISTRY[domain] = {}
        if op_type not in _VERSIONED_REGISTRY[domain]:
            _VERSIONED_REGISTRY[domain][op_type] = []

        handlers = _VERSIONED_REGISTRY[domain][op_type]
        # Remove existing handler with same since_version to allow re-registration
        handlers[:] = [(v, h) for v, h in handlers if v != since_version]
        handlers.append((since_version, func))
        # Keep sorted in descending order by version for efficient lookup
        handlers.sort(key=lambda x: x[0], reverse=True)

        return func

    return decorator


def register_custom_op(
    op_type: str,
    handler: Optional[OpHandler] = None,
    domain: str = "",
    since_version: int = 1,
) -> Union[OpHandler, Callable[[OpHandler], OpHandler]]:
    """Register a custom ONNX operator handler with version support.

    This function can be used as a decorator or called directly to register
    custom operator handlers for ONNX operators that are not natively supported.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    handler : OpHandler, optional
        The handler function. If not provided, returns a decorator.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).
    since_version : int, optional
        The minimum opset version this handler supports. Default is 1.

    Returns
    -------
    OpHandler or Callable
        If handler is provided, returns the handler.
        Otherwise, returns a decorator function.

    Examples
    --------
    Using as a decorator:

    >>> @register_custom_op("MyCustomOp")
    ... def my_custom_op(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(torch.sigmoid, args=(x,))

    Using as a function:

    >>> def my_handler(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(torch.tanh, args=(x,))
    >>> register_custom_op("TanhCustom", my_handler)

    Registering for a custom domain:

    >>> @register_custom_op("BiasGelu", domain="com.microsoft")
    ... def bias_gelu(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     bias = builder.get_value(node.input[1])
    ...     return builder.call_function(
    ...         lambda t, b: torch.nn.functional.gelu(t + b),
    ...         args=(x, bias)
    ...     )

    Registering version-specific handlers:

    >>> @register_custom_op("MyOp", since_version=1)
    ... def my_op_v1(builder, node): ...

    >>> @register_custom_op("MyOp", since_version=13)
    ... def my_op_v13(builder, node): ...
    """
    if handler is not None:
        # Direct call: register_custom_op("Op", handler)
        if domain not in _VERSIONED_REGISTRY:
            _VERSIONED_REGISTRY[domain] = {}
        if op_type not in _VERSIONED_REGISTRY[domain]:
            _VERSIONED_REGISTRY[domain][op_type] = []

        handlers = _VERSIONED_REGISTRY[domain][op_type]
        # Remove existing handler with same since_version
        handlers[:] = [(v, h) for v, h in handlers if v != since_version]
        handlers.append((since_version, handler))
        handlers.sort(key=lambda x: x[0], reverse=True)

        return handler
    else:
        # Decorator usage: @register_custom_op("Op")
        return register(op_type, domain, since_version)


def unregister_op(
    op_type: str, domain: str = "", since_version: Optional[int] = None
) -> bool:
    """Unregister an operator handler.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).
    since_version : int, optional
        The specific version handler to remove. If None, removes all versions.

    Returns
    -------
    bool
        True if the operator was unregistered, False if it wasn't registered.
    """
    if domain not in _VERSIONED_REGISTRY:
        return False
    if op_type not in _VERSIONED_REGISTRY[domain]:
        return False

    handlers = _VERSIONED_REGISTRY[domain][op_type]
    if since_version is None:
        # Remove all versions
        del _VERSIONED_REGISTRY[domain][op_type]
        return True
    else:
        # Remove specific version
        original_len = len(handlers)
        handlers[:] = [(v, h) for v, h in handlers if v != since_version]
        if len(handlers) < original_len:
            if not handlers:
                del _VERSIONED_REGISTRY[domain][op_type]
            return True
        return False


def get_handler(
    op_type: str, domain: str = "", opset_version: int = 23
) -> Optional[OpHandler]:
    """Get the handler for an operator at a specific opset version.

    Finds the handler with the highest since_version that is <= opset_version.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).
    opset_version : int, optional
        The target opset version. Default is 23 (current latest).

    Returns
    -------
    OpHandler or None
        The appropriate handler function, or None if not found.
    """
    # Normalize domain: "ai.onnx" is equivalent to ""
    if domain in ("ai.onnx", "ai.onnx.ml"):
        domain = ""

    if domain not in _VERSIONED_REGISTRY:
        return None

    handlers = _VERSIONED_REGISTRY[domain].get(op_type)
    if not handlers:
        return None

    # Handlers are sorted in descending order by since_version
    # Find the first handler where since_version <= opset_version
    for since_version, handler in handlers:
        if since_version <= opset_version:
            return handler

    return None


def is_supported(op_type: str, domain: str = "", opset_version: int = 23) -> bool:
    """Check if an operator is supported.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).
    opset_version : int, optional
        The target opset version. Default is 23.

    Returns
    -------
    bool
        True if the operator is supported.
    """
    return get_handler(op_type, domain, opset_version) is not None


def get_supported_ops(domain: str = "") -> list:
    """Get list of supported ONNX operators for a domain.

    Parameters
    ----------
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).

    Returns
    -------
    list
        Sorted list of supported operator names.
    """
    if domain in _VERSIONED_REGISTRY:
        return sorted(_VERSIONED_REGISTRY[domain].keys())
    return []


def get_all_supported_ops() -> Dict[str, list]:
    """Get all supported operators across all domains.

    Returns
    -------
    Dict[str, list]
        Dictionary mapping domain names to sorted lists of operator names.
    """
    return {domain: sorted(ops.keys()) for domain, ops in _VERSIONED_REGISTRY.items()}


def get_registered_domains() -> list:
    """Get list of registered domains.

    Returns
    -------
    list
        List of domain names.
    """
    return list(_VERSIONED_REGISTRY.keys())


def get_handler_versions(op_type: str, domain: str = "") -> List[int]:
    """Get all registered opset versions for an operator.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).

    Returns
    -------
    List[int]
        List of registered since_version values, sorted in ascending order.
    """
    if domain in _VERSIONED_REGISTRY:
        handlers = _VERSIONED_REGISTRY[domain].get(op_type, [])
        return sorted([v for v, _ in handlers])
    return []
