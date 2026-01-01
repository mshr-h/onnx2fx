# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor manipulation operators."""

import torch
import onnx
from onnxscript import FLOAT, INT64, script
from onnxscript import opset15 as op

from onnx2fx.converter import convert


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
        axes = op.Constant(value=onnx.numpy_helper.from_array(
            torch.tensor([1]).numpy(), name="axes"
        ))
        return op.Squeeze(x, axes)

    @script()
    def unsqueeze_script(x: FLOAT) -> FLOAT:
        axes = op.Constant(value=onnx.numpy_helper.from_array(
            torch.tensor([0]).numpy(), name="axes"
        ))
        return op.Unsqueeze(x, axes)

    def test_transpose(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.transpose_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, x.T)

    def test_concat(self):
        x = torch.randn(2, 4)
        y = torch.randn(3, 4)
        fx_model = convert(self.concat_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, torch.cat([x, y], dim=0))

    def test_flatten(self):
        x = torch.randn(2, 3, 4)
        fx_model = convert(self.flatten_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert result.shape == (2, 12)

    def test_squeeze(self):
        x = torch.randn(2, 1, 4)
        fx_model = convert(self.squeeze_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert result.shape == (2, 4)

    def test_unsqueeze(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.unsqueeze_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert result.shape == (1, 2, 4)


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
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, x.sum())

    def test_reduce_mean(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_mean_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, x.mean())

    def test_reduce_max(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_max_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, x.max())

    def test_reduce_min(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.reduce_min_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, x.min())
