# SPDX-License-Identifier: Apache-2.0
"""Tests for math operators including trigonometric, hyperbolic, and special functions."""

import onnx
import torch
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import run_onnx_test


# =============================================================================
# Trigonometric functions
# =============================================================================


class TestTrigOps:
    """Test trigonometric operators."""

    @script()
    def sin_script(x: FLOAT) -> FLOAT:
        return op.Sin(x)

    @script()
    def cos_script(x: FLOAT) -> FLOAT:
        return op.Cos(x)

    @script()
    def tan_script(x: FLOAT) -> FLOAT:
        return op.Tan(x)

    def test_sin(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.sin_script.to_model_proto, x, torch.sin(x))

    def test_cos(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.cos_script.to_model_proto, x, torch.cos(x))

    def test_tan(self):
        x = torch.randn(2, 4) * 0.5  # Avoid large values
        run_onnx_test(
            self.tan_script.to_model_proto, x, torch.tan(x), atol=1e-5, rtol=1e-5
        )


# =============================================================================
# Special math functions
# =============================================================================


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
        run_onnx_test(self.erf_script.to_model_proto, x, torch.erf(x))

    def test_isnan(self):
        x = torch.tensor([1.0, float("nan"), 2.0, float("nan")])
        fx_model = convert(self.isnan_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        assert torch.equal(result, torch.isnan(x))


# =============================================================================
# Range operator (moved from tensor generation)
# =============================================================================


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

        start = torch.tensor(0.0)
        limit = torch.tensor(5.0)
        delta = torch.tensor(1.0)
        expected = torch.arange(0.0, 5.0, 1.0)

        run_onnx_test(model, (start, limit, delta), expected)


# =============================================================================
# Cumulative and matrix operations
# =============================================================================


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

        x = torch.randn(2, 4)
        axis = torch.tensor(1)
        expected = torch.cumsum(x, dim=1)

        run_onnx_test(model, (x, axis), expected)


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

        x = torch.randn(3, 3)
        expected = torch.triu(x)

        run_onnx_test(model, x, expected)

    def test_tril(self):
        """Test lower triangular."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        trilu_node = onnx.helper.make_node("Trilu", ["X"], ["Y"], upper=0)

        graph = onnx.helper.make_graph([trilu_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        x = torch.randn(3, 3)
        expected = torch.tril(x)

        run_onnx_test(model, x, expected)
