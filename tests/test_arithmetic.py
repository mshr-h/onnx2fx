# SPDX-License-Identifier: Apache-2.0
"""Tests for arithmetic operators."""

import torch
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset22 as op

from onnx2fx.converter import convert


class TestBinaryArithmetic:
    """Test binary arithmetic operators."""

    @script()
    def sub_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Sub(x, y)

    @script()
    def mul_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Mul(x, y)

    @script()
    def div_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Div(x, y)

    @script()
    def pow_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Pow(x, y)

    @script()
    def min_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Min(x, y)

    @script()
    def max_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Max(x, y)

    def test_sub(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.sub_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, x - y)

    def test_mul(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.mul_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, x * y)

    def test_div(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4) + 0.1  # Avoid division by zero
        fx_model = convert(self.div_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, x / y)

    def test_pow(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1  # Positive values
        y = torch.randn(2, 4)
        fx_model = convert(self.pow_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, torch.pow(x, y), atol=1e-5)

    def test_min(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.min_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, torch.minimum(x, y))

    def test_max(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.max_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.allclose(result, torch.maximum(x, y))


class TestUnaryArithmetic:
    """Test unary arithmetic operators."""

    @script()
    def neg_script(x: FLOAT) -> FLOAT:
        return op.Neg(x)

    @script()
    def abs_script(x: FLOAT) -> FLOAT:
        return op.Abs(x)

    @script()
    def sqrt_script(x: FLOAT) -> FLOAT:
        return op.Sqrt(x)

    @script()
    def exp_script(x: FLOAT) -> FLOAT:
        return op.Exp(x)

    @script()
    def log_script(x: FLOAT) -> FLOAT:
        return op.Log(x)

    @script()
    def ceil_script(x: FLOAT) -> FLOAT:
        return op.Ceil(x)

    @script()
    def floor_script(x: FLOAT) -> FLOAT:
        return op.Floor(x)

    def test_neg(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.neg_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, -x)

    def test_abs(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.abs_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.abs(x))

    def test_sqrt(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        fx_model = convert(self.sqrt_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.sqrt(x))

    def test_exp(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.exp_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.exp(x))

    def test_log(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        fx_model = convert(self.log_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.log(x))

    def test_ceil(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.ceil_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.ceil(x))

    def test_floor(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.floor_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.floor(x))


class TestComparisonOps:
    """Test comparison operators."""

    @script()
    def equal_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Equal(x, y)

    @script()
    def greater_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Greater(x, y)

    @script()
    def less_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Less(x, y)

    def test_equal(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 0.0, 3.0])
        fx_model = convert(self.equal_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.equal(result, torch.eq(x, y))

    def test_greater(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.greater_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.equal(result, torch.gt(x, y))

    def test_less(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.less_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x, y)
        assert torch.equal(result, torch.lt(x, y))
