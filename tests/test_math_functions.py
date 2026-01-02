# SPDX-License-Identifier: Apache-2.0
"""Tests for math function operators."""

import onnx
import torch
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert


class TestMathOps:
    """Test additional math operators."""

    @script()
    def erf_script(x: FLOAT) -> FLOAT:
        return op.Erf(x)

    @script()
    def isnan_script(x: FLOAT) -> FLOAT:
        return op.IsNaN(x)

    def test_erf(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.erf_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.erf(x))

    def test_isnan(self):
        x = torch.tensor([1.0, float("nan"), 2.0, float("nan")])
        fx_model = convert(self.isnan_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        assert torch.equal(result, torch.isnan(x))


class TestRangeOp:
    """Test Range operator."""

    def test_range_float(self):
        """Test Range with float values."""
        start_info = onnx.helper.make_tensor_value_info(
            "start", onnx.TensorProto.FLOAT, []
        )
        limit_info = onnx.helper.make_tensor_value_info(
            "limit", onnx.TensorProto.FLOAT, []
        )
        delta_info = onnx.helper.make_tensor_value_info(
            "delta", onnx.TensorProto.FLOAT, []
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        range_node = onnx.helper.make_node("Range", ["start", "limit", "delta"], ["Y"])

        graph = onnx.helper.make_graph(
            [range_node], "test", [start_info, limit_info, delta_info], [y_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        start = torch.tensor(0.0)
        limit = torch.tensor(5.0)
        delta = torch.tensor(1.0)

        with torch.inference_mode():
            result = fx_model(start, limit, delta)

        expected = torch.arange(0.0, 5.0, 1.0)
        torch.testing.assert_close(result, expected)


class TestCumSumOp:
    """Test CumSum operator."""

    def test_cumsum(self):
        """Test cumulative sum."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
        axis_info = onnx.helper.make_tensor_value_info(
            "axis", onnx.TensorProto.INT64, []
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        cumsum_node = onnx.helper.make_node("CumSum", ["X", "axis"], ["Y"])

        graph = onnx.helper.make_graph(
            [cumsum_node], "test", [x_info, axis_info], [y_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 4)
        axis = torch.tensor(1)

        with torch.inference_mode():
            result = fx_model(x, axis)

        expected = torch.cumsum(x, dim=1)
        torch.testing.assert_close(result, expected)


class TestTriluOp:
    """Test Trilu operator."""

    def test_triu(self):
        """Test upper triangular."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        trilu_node = onnx.helper.make_node("Trilu", ["X"], ["Y"], upper=1)

        graph = onnx.helper.make_graph([trilu_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        x = torch.randn(3, 3)

        with torch.inference_mode():
            result = fx_model(x)

        expected = torch.triu(x)
        torch.testing.assert_close(result, expected)

    def test_tril(self):
        """Test lower triangular."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        trilu_node = onnx.helper.make_node("Trilu", ["X"], ["Y"], upper=0)

        graph = onnx.helper.make_graph([trilu_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        x = torch.randn(3, 3)

        with torch.inference_mode():
            result = fx_model(x)

        expected = torch.tril(x)
        torch.testing.assert_close(result, expected)
