# SPDX-License-Identifier: Apache-2.0
"""ONNX model analysis utilities for identifying operator support."""

from typing import Union, Dict, List, Set, Tuple
from dataclasses import dataclass, field

import onnx

from ..op_registry import is_supported


@dataclass
class AnalysisResult:
    """Result of analyzing an ONNX model for operator support.

    Attributes
    ----------
    total_nodes : int
        Total number of nodes in the model graph.
    unique_ops : Set[Tuple[str, str]]
        Set of unique (op_type, domain) tuples.
    supported_ops : List[Tuple[str, str]]
        List of supported (op_type, domain) tuples.
    unsupported_ops : List[Tuple[str, str, int]]
        List of unsupported (op_type, domain, opset_version) tuples.
    opset_versions : Dict[str, int]
        Mapping of domain to opset version.
    op_counts : Dict[Tuple[str, str], int]
        Count of each (op_type, domain) in the model.
    """

    total_nodes: int = 0
    unique_ops: Set[Tuple[str, str]] = field(default_factory=set)
    supported_ops: List[Tuple[str, str]] = field(default_factory=list)
    unsupported_ops: List[Tuple[str, str, int]] = field(default_factory=list)
    opset_versions: Dict[str, int] = field(default_factory=dict)
    op_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def is_fully_supported(self) -> bool:
        """Check if all operators in the model are supported."""
        return len(self.unsupported_ops) == 0

    def summary(self) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = [
            "ONNX Model Analysis Summary",
            "=" * 40,
            f"Total nodes: {self.total_nodes}",
            f"Unique operators: {len(self.unique_ops)}",
            f"Supported: {len(self.supported_ops)}",
            f"Unsupported: {len(self.unsupported_ops)}",
            "",
            "Opset versions:",
        ]
        for domain, version in sorted(self.opset_versions.items()):
            domain_display = domain if domain else "(default ONNX)"
            lines.append(f"  {domain_display}: {version}")

        if self.unsupported_ops:
            lines.append("")
            lines.append("Unsupported operators:")
            for op_type, domain, opset in self.unsupported_ops:
                domain_display = domain if domain else "(default)"
                lines.append(
                    f"  - {op_type} (domain: {domain_display}, opset: {opset})"
                )

        if self.supported_ops:
            lines.append("")
            lines.append("Supported operators:")
            for op_type, domain in sorted(self.supported_ops):
                domain_display = domain if domain else "(default)"
                count = self.op_counts.get((op_type, domain), 0)
                lines.append(f"  - {op_type} (domain: {domain_display}) x{count}")

        return "\n".join(lines)


def analyze_model(model: Union[onnx.ModelProto, str]) -> AnalysisResult:
    """Analyze an ONNX model and identify supported/unsupported operators.

    This function iterates through all nodes in an ONNX model graph and
    checks each operator against the onnx2fx registry to determine
    which operators are supported for conversion.

    Parameters
    ----------
    model : onnx.ModelProto or str
        The ONNX model or path to the ONNX model file.

    Returns
    -------
    AnalysisResult
        Analysis result containing supported/unsupported operators,
        opset versions, and operator counts.

    Examples
    --------
    >>> import onnx
    >>> from onnx2fx import analyze_model
    >>> model = onnx.load("model.onnx")
    >>> result = analyze_model(model)
    >>> print(result.summary())
    >>> if not result.is_fully_supported():
    ...     print("Missing operators:", result.unsupported_ops)
    """
    if isinstance(model, str):
        model = onnx.load(model)

    result = AnalysisResult()

    # Extract opset versions
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else ""
        result.opset_versions[domain] = opset.version

    # Analyze all nodes
    for node in model.graph.node:
        result.total_nodes += 1

        op_type = node.op_type
        domain = node.domain if node.domain else ""
        opset_version = result.opset_versions.get(domain, 1)

        op_key = (op_type, domain)
        result.unique_ops.add(op_key)

        # Count occurrences
        result.op_counts[op_key] = result.op_counts.get(op_key, 0) + 1

        # Check if supported (only add to supported/unsupported once per unique op)
        if op_key not in [(op, dom) for op, dom in result.supported_ops]:
            if op_key not in [(op, dom, _) for op, dom, _ in result.unsupported_ops]:
                if is_supported(op_type, domain, opset_version):
                    result.supported_ops.append(op_key)
                else:
                    result.unsupported_ops.append((op_type, domain, opset_version))

    return result
