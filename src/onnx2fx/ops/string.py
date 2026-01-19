# SPDX-License-Identifier: Apache-2.0
"""String operators.

This module implements ONNX operators for string processing.
"""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# String operators
# =============================================================================


@register("StringNormalizer")
def string_normalizer(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """String normalization for numpy string arrays.

    Applies case normalization and stopword filtering to string arrays.
    """
    from ..utils.attributes import get_attribute

    x = builder.get_value(node.input[0])
    case_change_action = get_attribute(node, "case_change_action", "NONE")
    is_case_sensitive = get_attribute(node, "is_case_sensitive", 0)
    stopwords = get_attribute(node, "stopwords", [])
    locale = get_attribute(node, "locale", "")

    def _string_normalizer(
        arr, case_action: str, case_sensitive: int, stops: list, loc: str
    ):
        import numpy as np

        original_shape = arr.shape
        is_2d = len(original_shape) == 2

        if is_2d:
            # Process each row separately for 2D arrays
            result_rows = []
            for row_idx in range(original_shape[0]):
                row = arr[row_idx]
                # Filter stopwords from this row
                if stops:
                    if case_sensitive:
                        stop_set = set(stops)
                        filtered = [s for s in row if s not in stop_set]
                    else:
                        stop_set_lower = {s.lower() for s in stops}
                        filtered = [
                            s
                            for s in row
                            if not isinstance(s, str) or s.lower() not in stop_set_lower
                        ]
                else:
                    filtered = list(row)

                # Apply case change
                if case_action == "LOWER":
                    filtered = [
                        s.lower() if isinstance(s, str) else s for s in filtered
                    ]
                elif case_action == "UPPER":
                    filtered = [
                        s.upper() if isinstance(s, str) else s for s in filtered
                    ]

                result_rows.append(filtered)

            # Find max length and build output
            if result_rows:
                # Pad shorter rows if needed (shouldn't happen for valid ONNX)
                output = np.array(result_rows, dtype=object)
                return output
            return np.array([[]], dtype=object)

        # 1D case
        flat = arr.flatten()

        # Filter stopwords first (before case change for matching)
        if stops:
            if case_sensitive:
                stop_set = set(stops)
                flat = np.array([s for s in flat if s not in stop_set], dtype=object)
            else:
                stop_set_lower = {s.lower() for s in stops}
                result = []
                for s in flat:
                    if isinstance(s, str):
                        if s.lower() not in stop_set_lower:
                            result.append(s)
                    else:
                        result.append(s)
                flat = (
                    np.array(result, dtype=object)
                    if result
                    else np.array([], dtype=object)
                )

        # Apply case change after filtering
        if case_action == "LOWER":
            flat = np.array(
                [s.lower() if isinstance(s, str) else s for s in flat], dtype=object
            )
        elif case_action == "UPPER":
            flat = np.array(
                [s.upper() if isinstance(s, str) else s for s in flat], dtype=object
            )

        # Handle empty result - ONNX spec says return [''] for empty
        if len(flat) == 0:
            return np.array([""], dtype=object)

        return flat

    return builder.call_function(
        _string_normalizer,
        args=(x, case_change_action, is_case_sensitive, stopwords, locale),
    )
