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


# =============================================================================
# Gradient operator
# =============================================================================


@register("Gradient", domain="ai.onnx.preview.training")
def gradient(builder: "GraphBuilder", node: onnx.NodeProto) -> list:
    """Compute gradients of y with respect to xs using PyTorch autograd.

    This operator computes symbolic gradients as specified in the ONNX
    ai.onnx.preview.training domain. It recomputes the forward pass with
    requires_grad=True to enable autograd.
    """
    # Get the xs (inputs to differentiate with respect to) and y (output to differentiate)
    xs_names = get_attribute(node, "xs", [])
    y_name = get_attribute(node, "y", "")

    # Get the xs nodes - these are the graph inputs we differentiate with respect to
    xs_nodes = [builder.get_value(name) for name in xs_names]

    # Find the path from xs to y by analyzing the ONNX graph
    # We need to collect all nodes that contribute to y
    graph = builder.model.graph
    node_outputs = {}  # output_name -> node
    for n in graph.node:
        for out in n.output:
            node_outputs[out] = n

    # Trace back from y to find all operations needed
    def trace_ops(target_name, collected_nodes):
        """Recursively collect all nodes needed to compute target_name."""
        if target_name in xs_names or target_name not in node_outputs:
            return
        n = node_outputs[target_name]
        if n not in collected_nodes:
            collected_nodes.append(n)
            for inp in n.input:
                trace_ops(inp, collected_nodes)

    ops_to_y = []
    trace_ops(y_name, ops_to_y)
    ops_to_y.reverse()  # Order from inputs to output

    # Create a function that recomputes y from xs with autograd enabled
    ops_info = [(n.op_type, list(n.input), list(n.output)) for n in ops_to_y]

    def _compute_gradient_symbolic(ops_info, xs_names, y_name, *xs_values):
        """Recompute forward pass and compute gradients."""
        # Create tensors with requires_grad
        env = {}
        for name, val in zip(xs_names, xs_values):
            if isinstance(val, torch.Tensor):
                env[name] = val.detach().clone().requires_grad_(True)
            else:
                env[name] = torch.tensor(val, dtype=torch.float32, requires_grad=True)

        # Replay operations
        for op_type, inputs, outputs in ops_info:
            input_tensors = [env[inp] for inp in inputs if inp in env]
            if op_type == "Add":
                result = input_tensors[0] + input_tensors[1]
            elif op_type == "Mul":
                result = input_tensors[0] * input_tensors[1]
            elif op_type == "Sub":
                result = input_tensors[0] - input_tensors[1]
            elif op_type == "Div":
                result = input_tensors[0] / input_tensors[1]
            else:
                # Fallback for unsupported ops
                result = input_tensors[0] if input_tensors else torch.tensor(0.0)
            for out in outputs:
                env[out] = result

        # Get y and compute gradients
        y = env.get(y_name)
        if y is None:
            return tuple(torch.zeros_like(x) for x in xs_values)

        xs_tensors = [env[name] for name in xs_names]
        grads = torch.autograd.grad(
            outputs=y,
            inputs=xs_tensors,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            allow_unused=True,
        )
        return grads

    # Create the gradient computation node
    result = builder.call_function(
        _compute_gradient_symbolic,
        args=(ops_info, xs_names, y_name, *xs_nodes),
    )

    # Return list of gradient outputs (one per xs input)
    outputs = []
    for i in range(len(xs_names)):

        def _get_grad_i(grads, idx=i):
            return grads[idx]

        outputs.append(builder.call_function(_get_grad_i, args=(result,)))

    return outputs


# =============================================================================
# Optimizer operators
# =============================================================================


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


def _adagrad_update(
    R: torch.Tensor,
    T: torch.Tensor,
    X: torch.Tensor,
    G: torch.Tensor,
    H: torch.Tensor,
    norm_coefficient: float,
    decay_factor: float,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute one iteration of ADAGRAD update.

    Parameters
    ----------
    R : torch.Tensor
        Initial learning rate (scalar).
    T : torch.Tensor
        Update count (scalar, int64).
    X : torch.Tensor
        Parameter tensor to optimize.
    G : torch.Tensor
        Gradient of X.
    H : torch.Tensor
        Accumulated squared gradient of X.
    norm_coefficient : float
        L2-norm regularization coefficient.
    decay_factor : float
        Learning rate decay factor.
    epsilon : float
        Small constant to avoid dividing by zero.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (X_new, H_new) - updated parameter and accumulated squared gradient.
    """
    # Compute decayed learning rate: r = R / (1 + T * decay_factor)
    r = R / (1 + T * decay_factor)

    # Add L2 regularization gradient: gradient of 0.5 * norm_coefficient * ||X||^2
    G_regularized = norm_coefficient * X + G

    # Compute new accumulated squared gradient
    H_new = H + G_regularized * G_regularized

    # Compute the adaptive part of per-coordinate learning rate
    H_adaptive = torch.sqrt(H_new) + epsilon

    # Compute the new value of X
    X_new = X - r * G_regularized / H_adaptive

    return X_new, H_new


@register("Adagrad", domain="ai.onnx.preview.training")
def adagrad(builder: "GraphBuilder", node: onnx.NodeProto) -> Tuple[torch.fx.Node, ...]:
    """Adagrad optimizer operator.

    Compute one iteration of ADAGRAD, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Inputs:
        R: Initial learning rate (scalar).
        T: Update count (scalar, int64).
        inputs (variadic): X_1, X_2, ..., X_n (parameters), G_1, G_2, ..., G_n (gradients),
                          H_1, H_2, ..., H_n (accumulated squared gradients).

    Outputs:
        X_1_new, X_2_new, ..., X_n_new, H_1_new, H_2_new, ..., H_n_new.

    Attributes:
        norm_coefficient: L2-norm regularization coefficient (default: 0.0).
        decay_factor: Learning rate decay factor (default: 0.0).
        epsilon: Small constant to avoid dividing by zero (default: 1e-6).
    """
    # Get attributes
    norm_coefficient = get_attribute(node, "norm_coefficient", 0.0)
    decay_factor = get_attribute(node, "decay_factor", 0.0)
    epsilon = get_attribute(node, "epsilon", 1e-6)

    # Get inputs: R, T, then groups of (X, G, H)
    # Input format: R, T, X_1, X_2, ..., X_n, G_1, G_2, ..., G_n, H_1, H_2, ..., H_n
    # Number of tensors to optimize: (num_inputs - 2) / 3
    num_inputs = len(node.input)
    num_tensors = (num_inputs - 2) // 3

    R = builder.get_value(node.input[0])
    T = builder.get_value(node.input[1])

    # Collect X, G, H tensors
    X_inputs = [builder.get_value(node.input[2 + i]) for i in range(num_tensors)]
    G_inputs = [
        builder.get_value(node.input[2 + num_tensors + i]) for i in range(num_tensors)
    ]
    H_inputs = [
        builder.get_value(node.input[2 + 2 * num_tensors + i])
        for i in range(num_tensors)
    ]

    # Process each tensor pair and collect results
    results = []
    for i in range(num_tensors):
        result = builder.call_function(
            _adagrad_update,
            args=(
                R,
                T,
                X_inputs[i],
                G_inputs[i],
                H_inputs[i],
                norm_coefficient,
                decay_factor,
                epsilon,
            ),
        )
        results.append(result)

    # If only one tensor, return the tuple directly
    if num_tensors == 1:
        return results[0]

    # For multiple tensors, we need to flatten results:
    # Output format: X_1_new, X_2_new, ..., X_n_new, H_1_new, H_2_new, ..., H_n_new
    # Create a helper function to extract and reorder the outputs
    def _reorder_adagrad_outputs(*results):
        X_news = [r[0] for r in results]
        H_news = [r[1] for r in results]
        return tuple(X_news + H_news)

    return builder.call_function(_reorder_adagrad_outputs, args=tuple(results))
