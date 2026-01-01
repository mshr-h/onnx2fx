# SPDX-License-Identifier: Apache-2.0
"""ONNX operator registry with custom operator support."""

from typing import TYPE_CHECKING, Dict, Callable, Optional, Union

import onnx
import torch

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .graph_builder import GraphBuilder

OpHandler = Callable[["GraphBuilder", onnx.NodeProto], object]

# Registry: {domain: {op_type: handler}}
# Empty string "" represents the default ONNX domain
_DOMAIN_REGISTRY: Dict[str, Dict[str, OpHandler]] = {"": {}}

# Backward compatibility alias
OP_REGISTRY: Dict[str, OpHandler] = _DOMAIN_REGISTRY[""]


def register(op_type: str, domain: str = "") -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an ONNX operator handler.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name (e.g., "Add", "Relu").
    domain : str, optional
        The ONNX domain (e.g., "com.microsoft"). Default is "" (standard ONNX domain).

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    Register a standard ONNX operator:

    >>> @register("MyOp")
    ... def my_op(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(torch.relu, args=(x,))

    Register a custom domain operator:

    >>> @register("CustomOp", domain="com.mycompany")
    ... def custom_op(builder, node):
    ...     x = builder.get_value(node.input[0])
    ...     return builder.call_function(my_custom_function, args=(x,))
    """

    def decorator(func: OpHandler) -> OpHandler:
        if domain not in _DOMAIN_REGISTRY:
            _DOMAIN_REGISTRY[domain] = {}
        _DOMAIN_REGISTRY[domain][op_type] = func
        return func

    return decorator


def register_custom_op(
    op_type: str,
    handler: Optional[OpHandler] = None,
    domain: str = "",
) -> Union[OpHandler, Callable[[OpHandler], OpHandler]]:
    """Register a custom ONNX operator handler.

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
    """
    if handler is not None:
        # Direct call: register_custom_op("Op", handler)
        if domain not in _DOMAIN_REGISTRY:
            _DOMAIN_REGISTRY[domain] = {}
        _DOMAIN_REGISTRY[domain][op_type] = handler
        return handler
    else:
        # Decorator usage: @register_custom_op("Op")
        return register(op_type, domain)


def unregister_op(op_type: str, domain: str = "") -> bool:
    """Unregister an operator handler.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).

    Returns
    -------
    bool
        True if the operator was unregistered, False if it wasn't registered.
    """
    if domain in _DOMAIN_REGISTRY and op_type in _DOMAIN_REGISTRY[domain]:
        del _DOMAIN_REGISTRY[domain][op_type]
        return True
    return False


def get_handler(op_type: str, domain: str = "") -> Optional[OpHandler]:
    """Get the handler for an operator.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).

    Returns
    -------
    OpHandler or None
        The handler function, or None if not found.
    """
    # Normalize domain: "ai.onnx" is equivalent to ""
    if domain in ("ai.onnx", "ai.onnx.ml"):
        domain = ""

    if domain in _DOMAIN_REGISTRY:
        return _DOMAIN_REGISTRY[domain].get(op_type)
    return None


def is_supported(op_type: str, domain: str = "") -> bool:
    """Check if an operator is supported.

    Parameters
    ----------
    op_type : str
        The ONNX operator type name.
    domain : str, optional
        The ONNX domain. Default is "" (standard ONNX domain).

    Returns
    -------
    bool
        True if the operator is supported.
    """
    return get_handler(op_type, domain) is not None


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
    if domain in _DOMAIN_REGISTRY:
        return sorted(_DOMAIN_REGISTRY[domain].keys())
    return []


def get_all_supported_ops() -> Dict[str, list]:
    """Get all supported operators across all domains.

    Returns
    -------
    Dict[str, list]
        Dictionary mapping domain names to sorted lists of operator names.
    """
    return {domain: sorted(ops.keys()) for domain, ops in _DOMAIN_REGISTRY.items()}


def get_registered_domains() -> list:
    """Get list of registered domains.

    Returns
    -------
    list
        List of domain names.
    """
    return list(_DOMAIN_REGISTRY.keys())

