# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor manipulation operators."""

import onnxscript
import pytest
import torch
import onnx
from onnx import TensorProto, helper
from onnxscript import FLOAT, INT64, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES, opset_id, run_onnx_test


class TestTensorOps:
    """Test tensor manipulation operators."""

    @script()
    def transpose_script(x: FLOAT) -> FLOAT:
        return op.Transpose(x, perm=[1, 0])

    @script()
    def concat_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Concat(x, y, axis=0)

    @script()
    def flatten_script(x: FLOAT) -> FLOAT:
        return op.Flatten(x, axis=1)

    @script()
    def squeeze_script(x: FLOAT) -> FLOAT:
        # Create constant for axes
        axes = op.Constant(
            value=onnx.numpy_helper.from_array(torch.tensor([1]).numpy(), name="axes")
        )
        return op.Squeeze(x, axes)

    @script()
    def unsqueeze_script(x: FLOAT) -> FLOAT:
        axes = op.Constant(
            value=onnx.numpy_helper.from_array(torch.tensor([0]).numpy(), name="axes")
        )
        return op.Unsqueeze(x, axes)

    def test_transpose(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.transpose_script.to_model_proto, x, x.T)

    def test_concat(self):
        x = torch.randn(2, 4)
        y = torch.randn(3, 4)
        run_onnx_test(self.concat_script.to_model_proto, (x, y), torch.cat([x, y], dim=0))

    def test_flatten(self):
        x = torch.randn(2, 3, 4)
        fx_model = convert(self.flatten_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        assert result.shape == (2, 12)

    def test_squeeze(self):
        x = torch.randn(2, 1, 4)
        fx_model = convert(self.squeeze_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        assert result.shape == (2, 4)

    def test_unsqueeze(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.unsqueeze_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        assert result.shape == (1, 2, 4)


class TestCastOps:
    """Test cast operators."""

    @script()
    def cast_like_script(x: FLOAT, target: INT64) -> INT64:
        return op.CastLike(x, target)

    def test_cast_like(self):
        """Test CastLike converts to target tensor's dtype."""
        x = torch.tensor([1.5, 2.7, 3.2], dtype=torch.float32)
        target = torch.tensor([0, 1, 2], dtype=torch.int64)
        fx_model = convert(self.cast_like_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, target)
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_cast_like_float_to_float(self):
        """Test CastLike between float types."""

        @script()
        def cast_like_float(x: FLOAT, target: FLOAT) -> FLOAT:
            return op.CastLike(x, target)

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        target = torch.tensor([0.0], dtype=torch.float64)
        fx_model = convert(cast_like_float.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, target)
        assert result.dtype == torch.float64
        torch.testing.assert_close(result, x.to(torch.float64))


class TestReductionOps:
    """Test reduction operators."""

    @script()
    def reduce_sum_script(x: FLOAT) -> FLOAT:
        return op.ReduceSum(x, keepdims=0)

    @script()
    def reduce_mean_script(x: FLOAT) -> FLOAT:
        return op.ReduceMean(x, keepdims=0)

    @script()
    def reduce_max_script(x: FLOAT) -> FLOAT:
        return op.ReduceMax(x, keepdims=0)

    @script()
    def reduce_min_script(x: FLOAT) -> FLOAT:
        return op.ReduceMin(x, keepdims=0)

    def test_reduce_sum(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.reduce_sum_script.to_model_proto, x, x.sum())

    def test_reduce_mean(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.reduce_mean_script.to_model_proto, x, x.mean())

    def test_reduce_max(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.reduce_max_script.to_model_proto, x, x.max())

    def test_reduce_min(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.reduce_min_script.to_model_proto, x, x.min())


class TestTensorOpsMultiOpset:
    """Test tensor manipulation operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_transpose_all_opsets(self, opset):
        """Transpose should work identically across all opsets."""

        @script(default_opset=opset)
        def transpose_script(x: FLOAT) -> FLOAT:
            return opset.Transpose(x, perm=[1, 0])

        x = torch.randn(2, 4)
        run_onnx_test(transpose_script.to_model_proto, x, x.T)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_concat_all_opsets(self, opset):
        """Concat should work identically across all opsets."""

        @script(default_opset=opset)
        def concat_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Concat(x, y, axis=0)

        x = torch.randn(2, 4)
        y = torch.randn(3, 4)
        run_onnx_test(concat_script.to_model_proto, (x, y), torch.cat([x, y], dim=0))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_flatten_all_opsets(self, opset):
        """Flatten should work identically across all opsets."""

        @script(default_opset=opset)
        def flatten_script(x: FLOAT) -> FLOAT:
            return opset.Flatten(x, axis=1)

        model = flatten_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)
        assert result.shape == (2, 12)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_reshape_all_opsets(self, opset):
        """Reshape should work identically across all opsets."""

        @script(default_opset=opset)
        def reshape_script(x: FLOAT, shape: INT64) -> FLOAT:
            return opset.Reshape(x, shape)

        model = reshape_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        shape = torch.tensor([2, 12], dtype=torch.int64)
        result = fx_model(x, shape)
        assert result.shape == (2, 12)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_expand_all_opsets(self, opset):
        """Expand should work identically across all opsets."""

        @script(default_opset=opset)
        def expand_script(x: FLOAT, shape: INT64) -> FLOAT:
            return opset.Expand(x, shape)

        x = torch.randn(1, 4)
        shape = torch.tensor([3, 4], dtype=torch.int64)
        run_onnx_test(expand_script.to_model_proto, (x, shape), x.expand(3, 4))


class TestReductionOpsMultiOpset:
    """Test reduction operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_reduce_sum_all_opsets(self, opset):
        """ReduceSum should work across all opsets."""

        @script(default_opset=opset)
        def reduce_sum_script(x: FLOAT) -> FLOAT:
            return opset.ReduceSum(x, keepdims=0)

        x = torch.randn(2, 4)
        run_onnx_test(reduce_sum_script.to_model_proto, x, x.sum())

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_reduce_mean_all_opsets(self, opset):
        """ReduceMean should work across all opsets."""

        @script(default_opset=opset)
        def reduce_mean_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMean(x, keepdims=0)

        x = torch.randn(2, 4)
        run_onnx_test(reduce_mean_script.to_model_proto, x, x.mean())

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_reduce_max_all_opsets(self, opset):
        """ReduceMax should work across all opsets."""

        @script(default_opset=opset)
        def reduce_max_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMax(x, keepdims=0)

        x = torch.randn(2, 4)
        run_onnx_test(reduce_max_script.to_model_proto, x, x.max())

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_reduce_min_all_opsets(self, opset):
        """ReduceMin should work across all opsets."""

        @script(default_opset=opset)
        def reduce_min_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMin(x, keepdims=0)

        x = torch.randn(2, 4)
        run_onnx_test(reduce_min_script.to_model_proto, x, x.min())


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

        data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        )
        condition = torch.tensor([True, False, True])
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]])

        run_onnx_test(model, (data, condition), expected)

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

        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        condition = torch.tensor([True, False, True, False, True, False])
        expected = torch.tensor([1.0, 3.0, 5.0])

        run_onnx_test(model, (data, condition), expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_compress_all_opsets(self, opset):
        """Compress should work across all opsets (9+)."""
        data_input = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4])
        cond_input = helper.make_tensor_value_info("condition", TensorProto.BOOL, [3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        compress_node = helper.make_node(
            "Compress",
            ["data", "condition"],
            ["output"],
            axis=0,
        )

        graph = helper.make_graph(
            [compress_node], "test", [data_input, cond_input], [output]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        )
        condition = torch.tensor([True, False, True])
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]])

        run_onnx_test(model, (data, condition), expected)


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

        shape = torch.tensor([2, 3], dtype=torch.int64)
        expected = torch.full((2, 3), 3.14)

        run_onnx_test(model, shape, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_constant_of_shape_all_opsets(self, opset):
        """ConstantOfShape should work across all opsets (9+)."""
        shape_input = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        value_tensor = helper.make_tensor("value", TensorProto.FLOAT, [1], [3.14])
        node = helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["output"],
            value=value_tensor,
        )

        graph = helper.make_graph([node], "test", [shape_input], [output])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        shape = torch.tensor([2, 3], dtype=torch.int64)
        expected = torch.full((2, 3), 3.14)

        run_onnx_test(model, shape, expected)


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
        indices_output = helper.make_tensor_value_info(
            "indices", TensorProto.INT64, None
        )

        topk_node = helper.make_node(
            "TopK",
            ["x", "k"],
            ["values", "indices"],
            name="topk",
            axis=-1,
        )

        graph = helper.make_graph(
            [topk_node],
            "topk_test",
            [x_input, k_input],
            [values_output, indices_output],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        x = torch.tensor(
            [[1.0, 4.0, 2.0, 3.0], [5.0, 2.0, 8.0, 1.0], [3.0, 6.0, 4.0, 7.0]]
        )
        k = torch.tensor([2], dtype=torch.int64)

        expected_values, expected_indices = torch.topk(x, 2, dim=-1)

        run_onnx_test(model, (x, k), (expected_values, expected_indices))
