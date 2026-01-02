# SPDX-License-Identifier: Apache-2.0
"""Tests for Einsum operator."""

import onnx
import torch

from onnx2fx import convert


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

        graph = onnx.helper.make_graph(
            [einsum_node], "test", [x_info, y_info], [z_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        with torch.inference_mode():
            result = fx_model(x, y)

        expected = torch.einsum("ij,jk->ik", x, y)
        torch.testing.assert_close(result, expected)

    def test_einsum_batch_matmul(self):
        """Test Einsum for batched matrix multiplication."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [2, 3, 4]
        )
        y_info = onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [2, 4, 5]
        )
        z_info = onnx.helper.make_tensor_value_info(
            "Z", onnx.TensorProto.FLOAT, [2, 3, 5]
        )

        einsum_node = onnx.helper.make_node(
            "Einsum", ["X", "Y"], ["Z"], equation="bij,bjk->bik"
        )

        graph = onnx.helper.make_graph(
            [einsum_node], "test", [x_info, y_info], [z_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)

        with torch.inference_mode():
            result = fx_model(x, y)

        expected = torch.einsum("bij,bjk->bik", x, y)
        torch.testing.assert_close(result, expected)
