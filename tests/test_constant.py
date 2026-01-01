# SPDX-License-Identifier: Apache-2.0

import torch
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset15 as op

from onnx2fx.converter import convert


class TestConstantOp:
    """Test Constant operator support."""

    @script()
    def constant_add_script(x: FLOAT) -> FLOAT:
        # Create a constant and add it to input
        c = op.Constant(value=onnx.numpy_helper.from_array(
            torch.ones(2, 4).numpy(), name="const"
        ))
        return op.Add(x, c)

    def test_constant_operation(self):
        """Test Constant op is correctly converted."""
        example_input = torch.randn(2, 4)
        onnx_model = self.constant_add_script.to_model_proto()

        fx_model = convert(onnx_model)

        # Eager mode evaluation
        eager_output = self.constant_add_script(example_input.numpy())

        with torch.no_grad():
            fx_output = fx_model(example_input)

        assert torch.allclose(
            torch.from_numpy(eager_output), fx_output, atol=1e-5
        ), "Constant op outputs do not match!"
