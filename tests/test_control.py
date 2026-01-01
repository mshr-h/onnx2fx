# SPDX-License-Identifier: Apache-2.0
"""Tests for control flow and multi-output operators."""

import numpy as np
import onnx
import onnxscript
import pytest
import torch
from onnx import TensorProto, helper
from onnxscript import opset21 as op

from onnx2fx import convert


class TestCompressOp:
    """Test Compress operator."""

    def test_compress_with_axis(self):
        """Test compress along an axis."""
        data_input = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4])
        cond_input = helper.make_tensor_value_info("condition", TensorProto.BOOL, [3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        compress_node = helper.make_node(
            "Compress",
            ["data", "condition"],
            ["output"],
            name="compress",
            axis=0,
        )

        graph = helper.make_graph(
            [compress_node], "compress_test", [data_input, cond_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        condition = torch.tensor([True, False, True])

        result = fx_module(data, condition)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]])

        torch.testing.assert_close(result, expected)

    def test_compress_flat(self):
        """Test compress without axis (flattened)."""
        data_input = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
        cond_input = helper.make_tensor_value_info("condition", TensorProto.BOOL, [6])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        compress_node = helper.make_node(
            "Compress",
            ["data", "condition"],
            ["output"],
            name="compress_flat",
        )

        graph = helper.make_graph(
            [compress_node], "compress_flat_test", [data_input, cond_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        condition = torch.tensor([True, False, True, False, True, False])

        result = fx_module(data, condition)
        expected = torch.tensor([1.0, 3.0, 5.0])

        torch.testing.assert_close(result, expected)


class TestBitShiftOp:
    """Test BitShift operator."""

    def test_bit_shift_left(self):
        """Test left bit shift."""
        x_input = helper.make_tensor_value_info("x", TensorProto.INT32, [3])
        y_input = helper.make_tensor_value_info("y", TensorProto.INT32, [3])
        output = helper.make_tensor_value_info("output", TensorProto.INT32, [3])

        shift_node = helper.make_node(
            "BitShift",
            ["x", "y"],
            ["output"],
            name="shift_left",
            direction="LEFT",
        )

        graph = helper.make_graph(
            [shift_node], "shift_left_test", [x_input, y_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        x = torch.tensor([1, 2, 4], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)

        result = fx_module(x, y)
        expected = torch.tensor([2, 8, 32], dtype=torch.int32)

        torch.testing.assert_close(result, expected)

    def test_bit_shift_right(self):
        """Test right bit shift."""
        x_input = helper.make_tensor_value_info("x", TensorProto.INT32, [3])
        y_input = helper.make_tensor_value_info("y", TensorProto.INT32, [3])
        output = helper.make_tensor_value_info("output", TensorProto.INT32, [3])

        shift_node = helper.make_node(
            "BitShift",
            ["x", "y"],
            ["output"],
            name="shift_right",
            direction="RIGHT",
        )

        graph = helper.make_graph(
            [shift_node], "shift_right_test", [x_input, y_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        x = torch.tensor([16, 32, 64], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)

        result = fx_module(x, y)
        expected = torch.tensor([8, 8, 8], dtype=torch.int32)

        torch.testing.assert_close(result, expected)


class TestConstantOfShapeOp:
    """Test ConstantOfShape operator."""

    def test_constant_of_shape(self):
        """Test creating constant tensor of given shape."""
        shape_input = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        # Create value tensor for fill value
        value_tensor = helper.make_tensor("value", TensorProto.FLOAT, [1], [3.14])

        const_node = helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["output"],
            name="constant_of_shape",
            value=value_tensor,
        )

        graph = helper.make_graph(
            [const_node], "constant_of_shape_test", [shape_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        shape = torch.tensor([2, 3], dtype=torch.int64)

        result = fx_module(shape)

        assert result.shape == (2, 3)
        torch.testing.assert_close(result, torch.full((2, 3), 3.14))


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

        fx_module = convert(model)

        x = torch.randn(2, 3)
        result = fx_module(x)

        assert result.shape == (2, 3)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

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

        fx_module = convert(model)

        x = torch.randn(3, 4)
        result = fx_module(x)

        assert result.shape == (3, 4)


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

        fx_module = convert(model)

        # Probabilities
        x = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])

        result = fx_module(x)

        assert result.shape == (3, 3)
        # Values should be 0 or 1
        assert ((result == 0) | (result == 1)).all()


class TestOptionalOps:
    """Test Optional operators."""

    def test_optional_with_value(self):
        """Test Optional with a value."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        optional_node = helper.make_node(
            "Optional",
            ["x"],
            ["opt"],
            name="optional",
        )

        get_node = helper.make_node(
            "OptionalGetElement",
            ["opt"],
            ["output"],
            name="get_element",
        )

        graph = helper.make_graph(
            [optional_node, get_node], "optional_test", [x_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        x = torch.randn(2, 3)
        result = fx_module(x)

        torch.testing.assert_close(result, x)

    def test_optional_has_element(self):
        """Test OptionalHasElement."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])

        optional_node = helper.make_node(
            "Optional",
            ["x"],
            ["opt"],
            name="optional",
        )

        has_node = helper.make_node(
            "OptionalHasElement",
            ["opt"],
            ["output"],
            name="has_element",
        )

        graph = helper.make_graph(
            [optional_node, has_node], "optional_has_test", [x_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        x = torch.randn(2, 3)
        result = fx_module(x)

        assert result.item() == True


class TestMultiOutputOps:
    """Test operators with multiple outputs."""

    def test_split_multi_output(self):
        """Test Split operator with multiple outputs."""
        @onnxscript.script()
        def split_model(x: onnxscript.FLOAT[6, 4]) -> tuple:
            a, b, c = op.Split(x, num_outputs=3, axis=0)
            return a, b, c

        model = split_model.to_model_proto()
        fx_module = convert(model)

        x = torch.randn(6, 4)
        result = fx_module(x)

        # Result should be a tuple of 3 tensors
        assert len(result) == 3
        assert result[0].shape == (2, 4)
        assert result[1].shape == (2, 4)
        assert result[2].shape == (2, 4)

        # Verify split is correct
        expected = torch.split(x, 2, dim=0)
        for i in range(3):
            torch.testing.assert_close(result[i], expected[i])

    def test_topk_multi_output(self):
        """Test TopK operator with multiple outputs (values and indices)."""
        x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
        k_input = helper.make_tensor_value_info("k", TensorProto.INT64, [1])
        values_output = helper.make_tensor_value_info("values", TensorProto.FLOAT, None)
        indices_output = helper.make_tensor_value_info("indices", TensorProto.INT64, None)

        topk_node = helper.make_node(
            "TopK",
            ["x", "k"],
            ["values", "indices"],
            name="topk",
            axis=-1,
        )

        graph = helper.make_graph(
            [topk_node], "topk_test", [x_input, k_input], [values_output, indices_output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        x = torch.tensor([[1.0, 4.0, 2.0, 3.0], [5.0, 2.0, 8.0, 1.0], [3.0, 6.0, 4.0, 7.0]])
        k = torch.tensor([2], dtype=torch.int64)

        values, indices = fx_module(x, k)

        expected_values, expected_indices = torch.topk(x, 2, dim=-1)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(indices, expected_indices)
