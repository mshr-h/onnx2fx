# SPDX-License-Identifier: Apache-2.0

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

        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        # Convert model with initializer
        fx_model = convert(model)

        # Test that the model works correctly
        x = torch.randn(2, 4)
        with torch.inference_mode():
            result = fx_model(x)

        expected = torch.matmul(x, torch.from_numpy(weight_data))
        assert torch.allclose(result, expected, atol=1e-5)
