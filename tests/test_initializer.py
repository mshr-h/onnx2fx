# SPDX-License-Identifier: Apache-2.0
"""Tests for ONNX initializer support."""

import pytest
import torch
import onnx
from onnx import helper, TensorProto

from conftest import OPSET_MODULES, opset_id, run_onnx_test


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

        x = torch.randn(2, 4)
        expected = torch.matmul(x, torch.from_numpy(weight_data))
        run_onnx_test(model, x, expected)


class TestInitializerMultiOpset:
    """Test initializer handling across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_initializer_matmul_all_opsets(self, opset):
        """Initializer with MatMul should work across all opsets."""
        weight_data = torch.randn(4, 4).numpy()
        weight_init = onnx.numpy_helper.from_array(weight_data, name="weight")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [2, 4]
        )

        matmul_node = helper.make_node("MatMul", ["input", "weight"], ["output"])

        graph = helper.make_graph(
            [matmul_node],
            "test",
            [input_tensor],
            [output_tensor],
            [weight_init],
        )

        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        x = torch.randn(2, 4)
        expected = torch.matmul(x, torch.from_numpy(weight_data))
        run_onnx_test(model, x, expected)
