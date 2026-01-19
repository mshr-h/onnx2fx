# SPDX-License-Identifier: Apache-2.0
"""Tests for Constant operator."""

import pytest
import torch
import onnx
from onnx import helper, TensorProto
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from conftest import OPSET_MODULES, opset_id, run_onnx_test


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
        x = torch.randn(2, 4)
        expected = x + torch.ones(2, 4)
        run_onnx_test(self.constant_add_script.to_model_proto, x, expected)


class TestConstantOpMultiOpset:
    """Test Constant operator across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
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

        x = torch.tensor([4.0, 5.0, 6.0])
        expected = x + torch.tensor([1.0, 2.0, 3.0])
        run_onnx_test(model, x, expected)
