# SPDX-License-Identifier: Apache-2.0
"""Tests for arithmetic operators."""

import pytest
import torch
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES


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
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, x - y)

    def test_mul(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.mul_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, x * y)

    def test_div(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4) + 0.1  # Avoid division by zero
        fx_model = convert(self.div_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, x / y)

    def test_pow(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1  # Positive values
        y = torch.randn(2, 4)
        fx_model = convert(self.pow_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.pow(x, y), rtol=1e-5, atol=1e-5)

    def test_min(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.min_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.minimum(x, y))

    def test_max(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.max_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.maximum(x, y))


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
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, -x)

    def test_abs(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.abs_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.abs(x))

    def test_sqrt(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        fx_model = convert(self.sqrt_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.sqrt(x))

    def test_exp(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.exp_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.exp(x))

    def test_log(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        fx_model = convert(self.log_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.log(x))

    def test_ceil(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.ceil_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.ceil(x))

    def test_floor(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.floor_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.floor(x))


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
        with torch.inference_mode():
            result = fx_model(x, y)
        assert torch.equal(result, torch.eq(x, y))

    def test_greater(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.greater_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        assert torch.equal(result, torch.gt(x, y))

    def test_less(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        fx_model = convert(self.less_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        assert torch.equal(result, torch.lt(x, y))


class TestArithmeticOpsMultiOpset:
    """Test arithmetic operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_add_all_opsets(self, opset):
        """Add should work identically across all opsets."""

        @script(default_opset=opset)
        def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Add(x, y)

        model = add_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        result = fx_model(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_sub_all_opsets(self, opset):
        """Sub should work identically across all opsets."""

        @script(default_opset=opset)
        def sub_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Sub(x, y)

        model = sub_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        result = fx_model(x, y)
        expected = x - y
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_mul_all_opsets(self, opset):
        """Mul should work identically across all opsets."""

        @script(default_opset=opset)
        def mul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Mul(x, y)

        model = mul_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        result = fx_model(x, y)
        expected = x * y
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_div_all_opsets(self, opset):
        """Div should work identically across all opsets."""

        @script(default_opset=opset)
        def div_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Div(x, y)

        model = div_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(2, 4) + 0.1  # Avoid division by zero
        result = fx_model(x, y)
        expected = x / y
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_neg_all_opsets(self, opset):
        """Neg should work identically across all opsets."""

        @script(default_opset=opset)
        def neg_script(x: FLOAT) -> FLOAT:
            return opset.Neg(x)

        model = neg_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = -x
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_abs_all_opsets(self, opset):
        """Abs should work identically across all opsets."""

        @script(default_opset=opset)
        def abs_script(x: FLOAT) -> FLOAT:
            return opset.Abs(x)

        model = abs_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.abs(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_sqrt_all_opsets(self, opset):
        """Sqrt should work identically across all opsets."""

        @script(default_opset=opset)
        def sqrt_script(x: FLOAT) -> FLOAT:
            return opset.Sqrt(x)

        model = sqrt_script.to_model_proto()
        fx_model = convert(model)
        x = torch.abs(torch.randn(2, 4)) + 0.1
        result = fx_model(x)
        expected = torch.sqrt(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_exp_all_opsets(self, opset):
        """Exp should work identically across all opsets."""

        @script(default_opset=opset)
        def exp_script(x: FLOAT) -> FLOAT:
            return opset.Exp(x)

        model = exp_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.exp(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_log_all_opsets(self, opset):
        """Log should work identically across all opsets."""

        @script(default_opset=opset)
        def log_script(x: FLOAT) -> FLOAT:
            return opset.Log(x)

        model = log_script.to_model_proto()
        fx_model = convert(model)
        x = torch.abs(torch.randn(2, 4)) + 0.1
        result = fx_model(x)
        expected = torch.log(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_matmul_all_opsets(self, opset):
        """MatMul should work identically across all opsets."""

        @script(default_opset=opset)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.MatMul(x, y)

        model = matmul_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        result = fx_model(x, y)
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected)
