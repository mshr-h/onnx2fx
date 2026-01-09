# SPDX-License-Identifier: Apache-2.0
"""Tests for image/spatial transformation operators."""

import onnx
import torch

from onnx2fx import convert


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

        fx_model = convert(model)

        x = torch.randn(1, 1, 2, 2)
        roi = torch.tensor([])
        scales = torch.tensor([1.0, 1.0, 2.0, 2.0])

        with torch.inference_mode():
            result = fx_model(x, roi, scales)

        assert result.shape == (1, 1, 4, 4)


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

        fx_model = convert(model)

        x = torch.randn(1, 4, 2, 2)

        with torch.inference_mode():
            result = fx_model(x)

        assert result.shape == (1, 1, 4, 4)

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

        fx_model = convert(model)

        x = torch.randn(1, 1, 4, 4)

        with torch.inference_mode():
            result = fx_model(x)

        assert result.shape == (1, 4, 2, 2)
