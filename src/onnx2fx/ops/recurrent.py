# SPDX-License-Identifier: Apache-2.0
"""Recurrent neural network operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute
from ..utils.op_helpers import get_optional_input

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Recurrent neural network operators
# =============================================================================


@register("LSTM")
def lstm(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """LSTM (Long Short-Term Memory) operator.

    Computes an one-layer LSTM.

    ONNX LSTM Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, 4*hidden_size, input_size]
    - R: recurrence weight [num_directions, 4*hidden_size, hidden_size]
    - B (optional): bias [num_directions, 8*hidden_size]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]
    - initial_c (optional): [num_directions, batch_size, hidden_size]
    - P (optional): peephole weights [num_directions, 3*hidden_size]

    ONNX LSTM Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]
    - Y_c (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    - Ct = ft (.) Ct-1 + it (.) ct
    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    - Ht = ot (.) h(Ct)
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = get_optional_input(builder, node, 3)
    sequence_lens = get_optional_input(builder, node, 4)
    initial_h = get_optional_input(builder, node, 5)
    initial_c = get_optional_input(builder, node, 6)
    peepholes = get_optional_input(builder, node, 7)

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    input_forget = get_attribute(node, "input_forget", 0)
    # activations = get_attribute(node, "activations", ["Sigmoid", "Tanh", "Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""
    output_y_c = len(node.output) > 2 and node.output[2] != ""

    def _lstm_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        initial_c,
        peepholes,
        hidden_size,
        direction,
        layout,
        input_forget,
        output_y,
        output_y_h,
        output_y_c,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Initialize cell state if not provided
        if initial_c is None:
            initial_c = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []
        all_y_c = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, 4*hidden_size, input_size]
            # ONNX order: [Wi, Wo, Wf, Wc] concatenated (input, output, forget, cell)
            w_dir = w[dir_idx]  # [4*hidden_size, input_size]
            w_i = w_dir[0:hidden_size, :]  # [hidden_size, input_size]
            w_o = w_dir[hidden_size : 2 * hidden_size, :]
            w_f = w_dir[2 * hidden_size : 3 * hidden_size, :]
            w_c = w_dir[3 * hidden_size : 4 * hidden_size, :]

            # R shape: [num_directions, 4*hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [4*hidden_size, hidden_size]
            r_i = r_dir[0:hidden_size, :]  # [hidden_size, hidden_size]
            r_o = r_dir[hidden_size : 2 * hidden_size, :]
            r_f = r_dir[2 * hidden_size : 3 * hidden_size, :]
            r_c = r_dir[3 * hidden_size : 4 * hidden_size, :]

            # Biases (optional)
            # B shape: [num_directions, 8*hidden_size]
            # = [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
            if b is not None:
                b_dir = b[dir_idx]  # [8*hidden_size]
                wb_i = b_dir[0:hidden_size]
                wb_o = b_dir[hidden_size : 2 * hidden_size]
                wb_f = b_dir[2 * hidden_size : 3 * hidden_size]
                wb_c = b_dir[3 * hidden_size : 4 * hidden_size]
                rb_i = b_dir[4 * hidden_size : 5 * hidden_size]
                rb_o = b_dir[5 * hidden_size : 6 * hidden_size]
                rb_f = b_dir[6 * hidden_size : 7 * hidden_size]
                rb_c = b_dir[7 * hidden_size : 8 * hidden_size]
            else:
                wb_i = wb_o = wb_f = wb_c = rb_i = rb_o = rb_f = rb_c = 0

            # Peepholes (optional)
            # P shape: [num_directions, 3*hidden_size] = [Pi, Po, Pf]
            if peepholes is not None:
                p_dir = peepholes[dir_idx]  # [3*hidden_size]
                p_i = p_dir[0:hidden_size]
                p_o = p_dir[hidden_size : 2 * hidden_size]
                p_f = p_dir[2 * hidden_size : 3 * hidden_size]
            else:
                p_i = p_o = p_f = 0

            # Initial hidden state and cell state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]
            c_t = initial_c[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute gates
                # it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
                i_t = torch.sigmoid(x_t @ w_i.T + h_t @ r_i.T + p_i * c_t + wb_i + rb_i)

                # ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
                f_t = torch.sigmoid(x_t @ w_f.T + h_t @ r_f.T + p_f * c_t + wb_f + rb_f)

                # Handle input_forget (coupled input-forget gate)
                if input_forget:
                    f_t = 1 - i_t

                # ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                c_tilde = torch.tanh(x_t @ w_c.T + h_t @ r_c.T + wb_c + rb_c)

                # Ct = ft (.) Ct-1 + it (.) ct
                c_t = f_t * c_t + i_t * c_tilde

                # ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
                o_t = torch.sigmoid(x_t @ w_o.T + h_t @ r_o.T + p_o * c_t + wb_o + rb_o)

                # Ht = ot (.) tanh(Ct)
                h_t = o_t * torch.tanh(c_t)

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)
            all_y_c.append(c_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Y_c: [num_directions, batch_size, hidden_size]
        y_c = torch.stack(all_y_c, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)
            # Convert Y_c from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_c = y_c.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h and output_y_c:
            return (y, y_h, y_c)
        elif output_y and output_y_h:
            return (y, y_h)
        elif output_y and output_y_c:
            return (y, y_c)
        elif output_y_h and output_y_c:
            return (y_h, y_c)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        elif output_y_c:
            return y_c
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _lstm_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            initial_c,
            peepholes,
            hidden_size,
            direction,
            layout,
            input_forget,
            output_y,
            output_y_h,
            output_y_c,
        ),
    )


@register("GRU")
def gru(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """GRU (Gated Recurrent Unit) operator.

    Computes an one-layer GRU.

    ONNX GRU Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, 3*hidden_size, input_size]
    - R: recurrence weight [num_directions, 3*hidden_size, hidden_size]
    - B (optional): bias [num_directions, 6*hidden_size]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]

    ONNX GRU Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Sigmoid, g=Tanh):
    - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # linear_before_reset=0
    - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)  # linear_before_reset!=0
    - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = get_optional_input(builder, node, 3)
    sequence_lens = get_optional_input(builder, node, 4)
    initial_h = get_optional_input(builder, node, 5)

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    linear_before_reset = get_attribute(node, "linear_before_reset", 0)
    # activations = get_attribute(node, "activations", ["Sigmoid", "Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""

    def _gru_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        layout,
        linear_before_reset,
        output_y,
        output_y_h,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, 3*hidden_size, input_size]
            # ONNX order: [Wz, Wr, Wh] concatenated
            w_dir = w[dir_idx]  # [3*hidden_size, input_size]
            w_z = w_dir[0:hidden_size, :]  # [hidden_size, input_size]
            w_r = w_dir[hidden_size : 2 * hidden_size, :]
            w_h = w_dir[2 * hidden_size : 3 * hidden_size, :]

            # R shape: [num_directions, 3*hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [3*hidden_size, hidden_size]
            r_z = r_dir[0:hidden_size, :]  # [hidden_size, hidden_size]
            r_r = r_dir[hidden_size : 2 * hidden_size, :]
            r_h = r_dir[2 * hidden_size : 3 * hidden_size, :]

            # Biases (optional)
            # B shape: [num_directions, 6*hidden_size] = [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
            if b is not None:
                b_dir = b[dir_idx]  # [6*hidden_size]
                wb_z = b_dir[0:hidden_size]
                wb_r = b_dir[hidden_size : 2 * hidden_size]
                wb_h = b_dir[2 * hidden_size : 3 * hidden_size]
                rb_z = b_dir[3 * hidden_size : 4 * hidden_size]
                rb_r = b_dir[4 * hidden_size : 5 * hidden_size]
                rb_h = b_dir[5 * hidden_size : 6 * hidden_size]
            else:
                wb_z = wb_r = wb_h = rb_z = rb_r = rb_h = 0

            # Initial hidden state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute gates
                # zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
                z_t = torch.sigmoid(
                    x_t @ w_z.T + h_t @ r_z.T + wb_z + rb_z
                )  # [batch_size, hidden_size]

                # rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
                r_t = torch.sigmoid(x_t @ w_r.T + h_t @ r_r.T + wb_r + rb_r)

                # ht = tanh(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # linear_before_reset=0
                # ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)  # linear_before_reset!=0
                if linear_before_reset:
                    h_tilde = torch.tanh(
                        x_t @ w_h.T + r_t * (h_t @ r_h.T + rb_h) + wb_h
                    )
                else:
                    h_tilde = torch.tanh(
                        x_t @ w_h.T + (r_t * h_t) @ r_h.T + rb_h + wb_h
                    )

                # Ht = (1 - zt) (.) ht + zt (.) Ht-1
                h_t = (1 - z_t) * h_tilde + z_t * h_t

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h:
            return (y, y_h)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _gru_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            layout,
            linear_before_reset,
            output_y,
            output_y_h,
        ),
    )


@register("RNN")
def rnn(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """RNN (Simple Recurrent Neural Network) operator.

    Computes an one-layer simple RNN.

    ONNX RNN Inputs:
    - X: input tensor [seq_length, batch_size, input_size] (layout=0)
         or [batch_size, seq_length, input_size] (layout=1)
    - W: weight tensor [num_directions, hidden_size, input_size]
    - R: recurrence weight [num_directions, hidden_size, hidden_size]
    - B (optional): bias [num_directions, 2*hidden_size] = [Wbi, Rbi]
    - sequence_lens (optional): [batch_size]
    - initial_h (optional): [num_directions, batch_size, hidden_size]

    ONNX RNN Outputs:
    - Y (optional): [seq_length, num_directions, batch_size, hidden_size]
    - Y_h (optional): [num_directions, batch_size, hidden_size]

    Equations (Default: f=Tanh):
    - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    """
    # Get inputs
    x = builder.get_value(node.input[0])
    w = builder.get_value(node.input[1])
    r = builder.get_value(node.input[2])

    # Optional inputs
    b = get_optional_input(builder, node, 3)
    sequence_lens = get_optional_input(builder, node, 4)
    initial_h = get_optional_input(builder, node, 5)

    # Get attributes
    hidden_size = get_attribute(node, "hidden_size")
    direction = get_attribute(node, "direction", "forward")
    layout = get_attribute(node, "layout", 0)
    # activations = get_attribute(node, "activations", ["Tanh"])
    # clip = get_attribute(node, "clip", None)

    # Determine output requirements
    output_y = len(node.output) > 0 and node.output[0] != ""
    output_y_h = len(node.output) > 1 and node.output[1] != ""

    def _rnn_impl(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        layout,
        output_y,
        output_y_h,
    ):
        # Handle layout: convert to seq_first format for processing
        # layout=0: [seq_length, batch_size, input_size]
        # layout=1: [batch_size, seq_length, input_size]
        if layout == 1:
            x = x.transpose(0, 1)

        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        # Initialize hidden state if not provided
        if initial_h is None:
            initial_h = torch.zeros(
                num_directions, batch_size, hidden_size, dtype=x.dtype, device=x.device
            )

        # Process each direction
        all_y = []
        all_y_h = []

        for dir_idx in range(num_directions):
            # Get weights for this direction
            # W shape: [num_directions, hidden_size, input_size]
            w_dir = w[dir_idx]  # [hidden_size, input_size]

            # R shape: [num_directions, hidden_size, hidden_size]
            r_dir = r[dir_idx]  # [hidden_size, hidden_size]

            # Biases (optional)
            # B shape: [num_directions, 2*hidden_size] = [Wbi, Rbi]
            if b is not None:
                b_dir = b[dir_idx]  # [2*hidden_size]
                wb_i = b_dir[0:hidden_size]
                rb_i = b_dir[hidden_size : 2 * hidden_size]
            else:
                wb_i = rb_i = 0

            # Initial hidden state for this direction
            h_t = initial_h[dir_idx]  # [batch_size, hidden_size]

            # Process sequence
            outputs = []
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                time_steps = range(seq_length - 1, -1, -1)
            else:
                time_steps = range(seq_length)

            for t in time_steps:
                x_t = x[t]  # [batch_size, input_size]

                # Compute: Ht = tanh(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                h_t = torch.tanh(x_t @ w_dir.T + h_t @ r_dir.T + wb_i + rb_i)

                outputs.append(h_t)

            # Stack outputs
            if direction == "reverse" or (
                direction == "bidirectional" and dir_idx == 1
            ):
                outputs = outputs[::-1]

            # [seq_length, batch_size, hidden_size]
            dir_y = torch.stack(outputs, dim=0)
            all_y.append(dir_y)
            all_y_h.append(h_t)

        # Combine directions
        # Y: [seq_length, num_directions, batch_size, hidden_size]
        y = torch.stack(all_y, dim=1)

        # Y_h: [num_directions, batch_size, hidden_size]
        y_h = torch.stack(all_y_h, dim=0)

        # Handle layout for output
        if layout == 1:
            # Convert Y from [seq_length, num_directions, batch_size, hidden_size]
            # to [batch_size, seq_length, num_directions, hidden_size]
            y = y.permute(2, 0, 1, 3)
            # Convert Y_h from [num_directions, batch_size, hidden_size]
            # to [batch_size, num_directions, hidden_size]
            y_h = y_h.transpose(0, 1)

        # Return based on required outputs
        if output_y and output_y_h:
            return (y, y_h)
        elif output_y:
            return y
        elif output_y_h:
            return y_h
        else:
            return y_h  # Default to returning Y_h

    return builder.call_function(
        _rnn_impl,
        args=(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            layout,
            output_y,
            output_y_h,
        ),
    )
