# SPDX-License-Identifier: Apache-2.0
"""Tests for advanced operators."""

import torch
import onnx
from onnxscript import FLOAT, INT64, script
from onnxscript import opset22 as op

from onnx2fx.converter import convert


class TestEinsumOps:
    """Test Einsum operator."""

    def test_einsum_matmul(self):
        """Test Einsum for matrix multiplication."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 4])
        z_info = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [2, 4])

        einsum_node = onnx.helper.make_node(
            "Einsum", ["X", "Y"], ["Z"], equation="ij,jk->ik"
        )

        graph = onnx.helper.make_graph([einsum_node], "test", [x_info, y_info], [z_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        with torch.no_grad():
            result = fx_model(x, y)

        expected = torch.einsum("ij,jk->ik", x, y)
        assert torch.allclose(result, expected)

    def test_einsum_batch_matmul(self):
        """Test Einsum for batched matrix multiplication."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3, 4])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4, 5])
        z_info = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [2, 3, 5])

        einsum_node = onnx.helper.make_node(
            "Einsum", ["X", "Y"], ["Z"], equation="bij,bjk->bik"
        )

        graph = onnx.helper.make_graph([einsum_node], "test", [x_info, y_info], [z_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)

        with torch.no_grad():
            result = fx_model(x, y)

        expected = torch.einsum("bij,bjk->bik", x, y)
        assert torch.allclose(result, expected)


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
        fx_model = convert(self.sin_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.sin(x))

    def test_cos(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.cos_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.cos(x))

    def test_tan(self):
        x = torch.randn(2, 4) * 0.5  # Avoid large values
        fx_model = convert(self.tan_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.tan(x), atol=1e-5)


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
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.erf(x))

    def test_isnan(self):
        x = torch.tensor([1.0, float('nan'), 2.0, float('nan')])
        fx_model = convert(self.isnan_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.equal(result, torch.isnan(x))


class TestRangeOp:
    """Test Range operator."""

    def test_range_float(self):
        """Test Range with float values."""
        start_info = onnx.helper.make_tensor_value_info("start", onnx.TensorProto.FLOAT, [])
        limit_info = onnx.helper.make_tensor_value_info("limit", onnx.TensorProto.FLOAT, [])
        delta_info = onnx.helper.make_tensor_value_info("delta", onnx.TensorProto.FLOAT, [])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        range_node = onnx.helper.make_node("Range", ["start", "limit", "delta"], ["Y"])

        graph = onnx.helper.make_graph(
            [range_node], "test", [start_info, limit_info, delta_info], [y_info]
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        start = torch.tensor(0.0)
        limit = torch.tensor(5.0)
        delta = torch.tensor(1.0)

        with torch.no_grad():
            result = fx_model(start, limit, delta)

        expected = torch.arange(0.0, 5.0, 1.0)
        assert torch.allclose(result, expected)


class TestCumSumOp:
    """Test CumSum operator."""

    def test_cumsum(self):
        """Test cumulative sum."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
        axis_info = onnx.helper.make_tensor_value_info("axis", onnx.TensorProto.INT64, [])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        cumsum_node = onnx.helper.make_node("CumSum", ["X", "axis"], ["Y"])

        graph = onnx.helper.make_graph([cumsum_node], "test", [x_info, axis_info], [y_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(2, 4)
        axis = torch.tensor(1)

        with torch.no_grad():
            result = fx_model(x, axis)

        expected = torch.cumsum(x, dim=1)
        assert torch.allclose(result, expected)


class TestTriluOp:
    """Test Trilu operator."""

    def test_triu(self):
        """Test upper triangular."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        trilu_node = onnx.helper.make_node("Trilu", ["X"], ["Y"], upper=1)

        graph = onnx.helper.make_graph([trilu_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(3, 3)

        with torch.no_grad():
            result = fx_model(x)

        expected = torch.triu(x)
        assert torch.allclose(result, expected)

    def test_tril(self):
        """Test lower triangular."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        trilu_node = onnx.helper.make_node("Trilu", ["X"], ["Y"], upper=0)

        graph = onnx.helper.make_graph([trilu_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(3, 3)

        with torch.no_grad():
            result = fx_model(x)

        expected = torch.tril(x)
        assert torch.allclose(result, expected)


class TestResizeOp:
    """Test Resize operator."""

    def test_resize_scales(self):
        """Test Resize with scales."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
        roi_info = onnx.helper.make_tensor_value_info("roi", onnx.TensorProto.FLOAT, [0])
        scales_info = onnx.helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [4])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        resize_node = onnx.helper.make_node(
            "Resize", ["X", "roi", "scales"], ["Y"], mode="nearest"
        )

        graph = onnx.helper.make_graph(
            [resize_node], "test", [x_info, roi_info, scales_info], [y_info]
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(1, 1, 2, 2)
        roi = torch.tensor([])
        scales = torch.tensor([1.0, 1.0, 2.0, 2.0])

        with torch.no_grad():
            result = fx_model(x, roi, scales)

        assert result.shape == (1, 1, 4, 4)


class TestSpaceDepthOps:
    """Test DepthToSpace and SpaceToDepth operators."""

    def test_depth_to_space(self):
        """Test DepthToSpace."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, 2, 2])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)

        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(1, 4, 2, 2)

        with torch.no_grad():
            result = fx_model(x)

        assert result.shape == (1, 1, 4, 4)

    def test_space_to_depth(self):
        """Test SpaceToDepth."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)

        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 15)])

        fx_model = convert(model)

        x = torch.randn(1, 1, 4, 4)

        with torch.no_grad():
            result = fx_model(x)

        assert result.shape == (1, 4, 2, 2)
