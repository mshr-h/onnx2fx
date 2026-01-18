# SPDX-License-Identifier: Apache-2.0
"""Training operators from ai.onnx.preview.training domain."""

from typing import TYPE_CHECKING, Tuple

import onnx
import torch
import torch.fx

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


def _momentum_update(
    R: torch.Tensor,
    T: torch.Tensor,
    X: torch.Tensor,
    G: torch.Tensor,
    V: torch.Tensor,
    alpha: float,
    beta: float,
    norm_coefficient: float,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute one iteration of stochastic gradient update with momentum.

    Parameters
    ----------
    R : torch.Tensor
        Learning rate (scalar).
    T : torch.Tensor
        Training iteration count (scalar, int64).
    X : torch.Tensor
        Parameter tensor to optimize.
    G : torch.Tensor
        Gradient of X.
    V : torch.Tensor
        Accumulated momentum of X.
    alpha : float
        Decay coefficient of previous accumulated gradient (momentum).
    beta : float
        Scaling coefficient of current gradient.
    norm_coefficient : float
        L2-norm regularization coefficient.
    mode : str
        Either "standard" or "nesterov".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (X_new, V_new) - updated parameter and momentum.
    """
    # Add L2 regularization gradient: gradient of 0.5 * norm_coefficient * ||X||^2
    G_regularized = norm_coefficient * X + G

    # In the first training iteration (T == 0), beta should always be 1
    beta_adjusted = beta if T.item() > 0 else 1.0

    # Compute new momentum
    V_new = alpha * V + beta_adjusted * G_regularized

    if mode == "nesterov":
        # Nesterov momentum: X_new = X - R * (G_regularized + alpha * V_new)
        X_new = X - R * (G_regularized + alpha * V_new)
    else:
        # Standard momentum: X_new = X - R * V_new
        X_new = X - R * V_new

    return X_new, V_new


@register("Momentum", domain="ai.onnx.preview.training")
def momentum(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> Tuple[torch.fx.Node, ...]:
    """Momentum optimizer operator.

    Compute one iteration of stochastic gradient update with momentum.
    This operator can conduct the optimization of multiple tensor variables.

    Inputs:
        R: Learning rate (scalar).
        T: Training iteration count (scalar, int64).
        inputs (variadic): X_1, X_2, ..., X_n (parameters), G_1, G_2, ..., G_n (gradients),
                          V_1, V_2, ..., V_n (momentums).

    Outputs:
        X_1_new, X_2_new, ..., X_n_new, V_1_new, V_2_new, ..., V_n_new.

    Attributes:
        alpha: Decay coefficient of previous momentum.
        beta: Scaling coefficient of current gradient.
        mode: "standard" or "nesterov".
        norm_coefficient: L2-norm regularization coefficient.
    """
    # Get attributes
    alpha = get_attribute(node, "alpha", 0.9)
    beta = get_attribute(node, "beta", 1.0)
    mode = get_attribute(node, "mode", "standard")
    norm_coefficient = get_attribute(node, "norm_coefficient", 0.0)

    # Decode mode if bytes
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")

    # Get inputs: R, T, then groups of (X, G, V)
    # Input format: R, T, X_1, X_2, ..., X_n, G_1, G_2, ..., G_n, V_1, V_2, ..., V_n
    # Number of tensors to optimize: (num_inputs - 2) / 3
    num_inputs = len(node.input)
    num_tensors = (num_inputs - 2) // 3

    R = builder.get_value(node.input[0])
    T = builder.get_value(node.input[1])

    # Collect X, G, V tensors
    X_inputs = [builder.get_value(node.input[2 + i]) for i in range(num_tensors)]
    G_inputs = [
        builder.get_value(node.input[2 + num_tensors + i]) for i in range(num_tensors)
    ]
    V_inputs = [
        builder.get_value(node.input[2 + 2 * num_tensors + i])
        for i in range(num_tensors)
    ]

    # Process each tensor pair and collect results
    results = []
    for i in range(num_tensors):
        result = builder.call_function(
            _momentum_update,
            args=(
                R,
                T,
                X_inputs[i],
                G_inputs[i],
                V_inputs[i],
                alpha,
                beta,
                norm_coefficient,
                mode,
            ),
        )
        results.append(result)

    # If only one tensor, return the tuple directly
    if num_tensors == 1:
        return results[0]

    # For multiple tensors, we need to flatten results:
    # Output format: X_1_new, X_2_new, ..., X_n_new, V_1_new, V_2_new, ..., V_n_new
    # Create a helper function to extract and reorder the outputs
    def _reorder_momentum_outputs(*results):
        X_news = [r[0] for r in results]
        V_news = [r[1] for r in results]
        return tuple(X_news + V_news)

    return builder.call_function(_reorder_momentum_outputs, args=tuple(results))
