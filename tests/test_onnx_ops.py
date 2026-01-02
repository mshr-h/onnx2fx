# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from onnx import helper, TensorProto
from onnxscript import FLOAT, script
from onnxscript import opset11, opset12, opset13, opset14, opset15
from onnxscript import opset16, opset17, opset18, opset19, opset20
from onnxscript import opset21, opset22, opset23
from onnxscript import opset22 as op

from onnx2fx.converter import convert

# Available opset modules for parametrized tests
OPSET_MODULES = [
    opset11,
    opset12,
    opset13,
    opset14,
    opset15,
    opset16,
    opset17,
    opset18,
    opset19,
    opset20,
    opset21,
    opset22,
    opset23,
]


class TestAddOps:
    """Test models with different data types."""

    @script()
    def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Add(x, y)

    def test_float32_model(self):
        """Test float32 model conversion."""
        x = torch.randn(2, 4, dtype=torch.float32)
        y = torch.randn(2, 4, dtype=torch.float32)
        onnx_model = self.add_script.to_model_proto()

        fx_model = convert(onnx_model)

        with torch.inference_mode():
            fx_output = fx_model(x, y)

        expected = x + y
        assert torch.allclose(fx_output, expected), "Float32 output mismatch!"


class TestAddOpsMultiOpset:
    """Test Add operation across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_add_all_opsets(self, opset):
        """Add should work across all opsets."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        z_info = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4])

        add_node = helper.make_node("Add", ["X", "Y"], ["Z"])

        graph = helper.make_graph([add_node], "test", [x_info, y_info], [z_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        result = fx_model(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)
