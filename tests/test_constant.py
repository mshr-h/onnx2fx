# SPDX-License-Identifier: Apache-2.0
"""Tests for Constant operator."""

import pytest
import torch
import onnx
from onnx import helper, TensorProto
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES


class TestConstantOp:
    """Test Constant operator support."""

    @script()
    def constant_add_script(x: FLOAT) -> FLOAT:
        # Create a constant and add it to input
        c = op.Constant(
            value=onnx.numpy_helper.from_array(torch.ones(2, 4).numpy(), name="const")
        )
        return op.Add(x, c)

    def test_constant_operation(self):
        """Test Constant op is correctly converted."""
        example_input = torch.randn(2, 4)
        onnx_model = self.constant_add_script.to_model_proto()

        fx_model = convert(onnx_model)

        # Eager mode evaluation
        eager_output = self.constant_add_script(example_input.numpy())

        with torch.inference_mode():
            fx_output = fx_model(example_input)

        (
            torch.testing.assert_close(
                torch.from_numpy(eager_output), fx_output, rtol=1e-5, atol=1e-5
            ),
            ("Constant op outputs do not match!"),
        )


class TestConstantOpMultiOpset:
    """Test Constant operator across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_constant_all_opsets(self, opset):
        """Constant should work across all opsets."""
        const_value = onnx.numpy_helper.from_array(
            torch.tensor([1.0, 2.0, 3.0]).numpy(), name="const"
        )

        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
        output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])

        const_node = helper.make_node("Constant", [], ["const"], value=const_value)
        add_node = helper.make_node("Add", ["X", "const"], ["Y"])

        graph = helper.make_graph(
            [const_node, add_node], "test", [input_info], [output_info]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_model = convert(model)

        x = torch.tensor([4.0, 5.0, 6.0])
        result = fx_model(x)
        expected = x + torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected)
