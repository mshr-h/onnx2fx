# SPDX-License-Identifier: Apache-2.0
"""Tests for image/spatial transformation operators."""

import onnx
import torch

from conftest import run_onnx_test


class TestResizeOp:
    """Test Resize operator."""

    def test_resize_scales(self):
        """Test Resize with scales."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 1, 2, 2]
        )
        roi_info = onnx.helper.make_tensor_value_info(
            "roi", onnx.TensorProto.FLOAT, [0]
        )
        scales_info = onnx.helper.make_tensor_value_info(
            "scales", onnx.TensorProto.FLOAT, [4]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        resize_node = onnx.helper.make_node(
            "Resize", ["X", "roi", "scales"], ["Y"], mode="nearest"
        )

        graph = onnx.helper.make_graph(
            [resize_node], "test", [x_info, roi_info, scales_info], [y_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        x = torch.randn(1, 1, 2, 2)
        roi = torch.tensor([])
        scales = torch.tensor([1.0, 1.0, 2.0, 2.0])

        expected = torch.empty(1, 1, 4, 4)
        fx_model = run_onnx_test(
            model,
            (x, roi, scales),
            expected,
            check_shape_only=True,
        )

        assert fx_model(x, roi, scales).shape == (1, 1, 4, 4)


class TestSpaceDepthOps:
    """Test DepthToSpace and SpaceToDepth operators."""

    def test_depth_to_space(self):
        """Test DepthToSpace."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 4, 2, 2]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)

        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        x = torch.randn(1, 4, 2, 2)

        expected = torch.empty(1, 1, 4, 4)
        fx_model = run_onnx_test(model, x, expected, check_shape_only=True)
        assert fx_model(x).shape == (1, 1, 4, 4)

    def test_space_to_depth(self):
        """Test SpaceToDepth."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 1, 4, 4]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)

        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        x = torch.randn(1, 1, 4, 4)

        expected = torch.empty(1, 4, 2, 2)
        fx_model = run_onnx_test(model, x, expected, check_shape_only=True)
        assert fx_model(x).shape == (1, 4, 2, 2)
