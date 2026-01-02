# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor manipulation operators."""

import pytest
import torch
import onnx
from onnxscript import FLOAT, INT64, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES


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
        fx_model = convert(self.transpose_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, x.T)

    def test_concat(self):
        x = torch.randn(2, 4)
        y = torch.randn(3, 4)
        fx_model = convert(self.concat_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.cat([x, y], dim=0))

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
        fx_model = convert(self.reduce_sum_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, x.sum())

    def test_reduce_mean(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_mean_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, x.mean())

    def test_reduce_max(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_max_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, x.max())

    def test_reduce_min(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_min_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, x.min())


class TestTensorOpsMultiOpset:
    """Test tensor manipulation operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_transpose_all_opsets(self, opset):
        """Transpose should work identically across all opsets."""

        @script(default_opset=opset)
        def transpose_script(x: FLOAT) -> FLOAT:
            return opset.Transpose(x, perm=[1, 0])

        model = transpose_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = x.T
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_concat_all_opsets(self, opset):
        """Concat should work identically across all opsets."""

        @script(default_opset=opset)
        def concat_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Concat(x, y, axis=0)

        model = concat_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(3, 4)
        result = fx_model(x, y)
        expected = torch.cat([x, y], dim=0)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
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

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
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

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_expand_all_opsets(self, opset):
        """Expand should work identically across all opsets."""

        @script(default_opset=opset)
        def expand_script(x: FLOAT, shape: INT64) -> FLOAT:
            return opset.Expand(x, shape)

        model = expand_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(1, 4)
        shape = torch.tensor([3, 4], dtype=torch.int64)
        result = fx_model(x, shape)
        assert result.shape == (3, 4)
        expected = x.expand(3, 4)
        torch.testing.assert_close(result, expected)


class TestReductionOpsMultiOpset:
    """Test reduction operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_reduce_sum_all_opsets(self, opset):
        """ReduceSum should work across all opsets."""

        @script(default_opset=opset)
        def reduce_sum_script(x: FLOAT) -> FLOAT:
            return opset.ReduceSum(x, keepdims=0)

        model = reduce_sum_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = x.sum()
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_reduce_mean_all_opsets(self, opset):
        """ReduceMean should work across all opsets."""

        @script(default_opset=opset)
        def reduce_mean_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMean(x, keepdims=0)

        model = reduce_mean_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = x.mean()
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_reduce_max_all_opsets(self, opset):
        """ReduceMax should work across all opsets."""

        @script(default_opset=opset)
        def reduce_max_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMax(x, keepdims=0)

        model = reduce_max_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = x.max()
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_reduce_min_all_opsets(self, opset):
        """ReduceMin should work across all opsets."""

        @script(default_opset=opset)
        def reduce_min_script(x: FLOAT) -> FLOAT:
            return opset.ReduceMin(x, keepdims=0)

        model = reduce_min_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = x.min()
        torch.testing.assert_close(result, expected)
