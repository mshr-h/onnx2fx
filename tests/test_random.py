# SPDX-License-Identifier: Apache-2.0
"""Tests for random generation operators."""

import torch
from onnx import TensorProto, helper

from conftest import run_onnx_test


class TestRandomOps:
    """Test random generation operators."""

    def test_random_uniform_like(self):
        """Test RandomUniformLike operator."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        random_node = helper.make_node(
            "RandomUniformLike",
            ["x"],
            ["output"],
            name="random_uniform_like",
            low=0.0,
            high=1.0,
        )

        graph = helper.make_graph(
            [random_node], "random_uniform_like_test", [x_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        x = torch.randn(2, 3)
        expected_shape = torch.zeros(2, 3)
        run_onnx_test(model, x, expected_shape, check_shape_only=True)

    def test_random_normal_like(self):
        """Test RandomNormalLike operator."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])

        random_node = helper.make_node(
            "RandomNormalLike",
            ["x"],
            ["output"],
            name="random_normal_like",
            mean=0.0,
            scale=1.0,
        )

        graph = helper.make_graph(
            [random_node], "random_normal_like_test", [x_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        x = torch.randn(3, 4)
        expected_shape = torch.zeros(3, 4)
        run_onnx_test(model, x, expected_shape, check_shape_only=True)


class TestBernoulliOp:
    """Test Bernoulli operator."""

    def test_bernoulli(self):
        """Test Bernoulli sampling."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3])

        bernoulli_node = helper.make_node(
            "Bernoulli",
            ["x"],
            ["output"],
            name="bernoulli",
        )

        graph = helper.make_graph(
            [bernoulli_node], "bernoulli_test", [x_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        # Probabilities
        x = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        expected_shape = torch.zeros(3, 3)
        run_onnx_test(model, x, expected_shape, check_shape_only=True)
