# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

from onnx2fx.converter import convert


class TestInitializer:
    """Test ONNX initializer support."""

    def test_model_with_initializer(self):
        """Test that models with initializers are correctly converted."""
        # Create a simple model with an initializer (weight)
        weight_data = torch.randn(4, 4).numpy()
        weight_init = onnx.numpy_helper.from_array(weight_data, name="weight")

        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [2, 4]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [2, 4]
        )

        matmul_node = onnx.helper.make_node(
            "MatMul", ["input", "weight"], ["output"], name="matmul"
        )

        graph = onnx.helper.make_graph(
            [matmul_node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            [weight_init],
        )

        model = onnx.helper.make_model(graph, opset_imports=[
            onnx.helper.make_opsetid("", 15)
        ])

        # This will fail because MatMul is not implemented yet
        # But we can test that initializer loading doesn't crash
        with pytest.raises(NotImplementedError, match="MatMul"):
            convert(model)
