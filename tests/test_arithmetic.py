# SPDX-License-Identifier: Apache-2.0
"""Tests for arithmetic operators."""

import pytest
import torch
from onnx import TensorProto, helper
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from conftest import OPSET_MODULES, opset_id, run_onnx_test


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
        run_onnx_test(self.sub_script.to_model_proto, (x, y), x - y)

    def test_mul(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(self.mul_script.to_model_proto, (x, y), x * y)

    def test_div(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4) + 0.1  # Avoid division by zero
        run_onnx_test(self.div_script.to_model_proto, (x, y), x / y)

    def test_pow(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1  # Positive values
        y = torch.randn(2, 4)
        run_onnx_test(
            self.pow_script.to_model_proto, (x, y), torch.pow(x, y), rtol=1e-5, atol=1e-5
        )

    def test_min(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(self.min_script.to_model_proto, (x, y), torch.minimum(x, y))

    def test_max(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(self.max_script.to_model_proto, (x, y), torch.maximum(x, y))


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
        run_onnx_test(self.neg_script.to_model_proto, x, -x)

    def test_abs(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.abs_script.to_model_proto, x, torch.abs(x))

    def test_sqrt(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        run_onnx_test(self.sqrt_script.to_model_proto, x, torch.sqrt(x))

    def test_exp(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.exp_script.to_model_proto, x, torch.exp(x))

    def test_log(self):
        x = torch.abs(torch.randn(2, 4)) + 0.1
        run_onnx_test(self.log_script.to_model_proto, x, torch.log(x))

    def test_ceil(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.ceil_script.to_model_proto, x, torch.ceil(x))

    def test_floor(self):
        x = torch.randn(2, 4)
        run_onnx_test(self.floor_script.to_model_proto, x, torch.floor(x))


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
        run_onnx_test(self.equal_script.to_model_proto, (x, y), torch.eq(x, y))

    def test_greater(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(self.greater_script.to_model_proto, (x, y), torch.gt(x, y))

    def test_less(self):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(self.less_script.to_model_proto, (x, y), torch.lt(x, y))


class TestArithmeticOpsMultiOpset:
    """Test arithmetic operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_add_all_opsets(self, opset):
        """Add should work identically across all opsets."""

        @script(default_opset=opset)
        def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Add(x, y)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(add_script.to_model_proto, (x, y), x + y)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_sub_all_opsets(self, opset):
        """Sub should work identically across all opsets."""

        @script(default_opset=opset)
        def sub_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Sub(x, y)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(sub_script.to_model_proto, (x, y), x - y)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_mul_all_opsets(self, opset):
        """Mul should work identically across all opsets."""

        @script(default_opset=opset)
        def mul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Mul(x, y)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_onnx_test(mul_script.to_model_proto, (x, y), x * y)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_div_all_opsets(self, opset):
        """Div should work identically across all opsets."""

        @script(default_opset=opset)
        def div_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.Div(x, y)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4) + 0.1  # Avoid division by zero
        run_onnx_test(div_script.to_model_proto, (x, y), x / y)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_neg_all_opsets(self, opset):
        """Neg should work identically across all opsets."""

        @script(default_opset=opset)
        def neg_script(x: FLOAT) -> FLOAT:
            return opset.Neg(x)

        x = torch.randn(2, 4)
        run_onnx_test(neg_script.to_model_proto, x, -x)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_abs_all_opsets(self, opset):
        """Abs should work identically across all opsets."""

        @script(default_opset=opset)
        def abs_script(x: FLOAT) -> FLOAT:
            return opset.Abs(x)

        x = torch.randn(2, 4)
        run_onnx_test(abs_script.to_model_proto, x, torch.abs(x))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_sqrt_all_opsets(self, opset):
        """Sqrt should work identically across all opsets."""

        @script(default_opset=opset)
        def sqrt_script(x: FLOAT) -> FLOAT:
            return opset.Sqrt(x)

        x = torch.abs(torch.randn(2, 4)) + 0.1
        run_onnx_test(sqrt_script.to_model_proto, x, torch.sqrt(x))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_exp_all_opsets(self, opset):
        """Exp should work identically across all opsets."""

        @script(default_opset=opset)
        def exp_script(x: FLOAT) -> FLOAT:
            return opset.Exp(x)

        x = torch.randn(2, 4)
        run_onnx_test(exp_script.to_model_proto, x, torch.exp(x))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_log_all_opsets(self, opset):
        """Log should work identically across all opsets."""

        @script(default_opset=opset)
        def log_script(x: FLOAT) -> FLOAT:
            return opset.Log(x)

        x = torch.abs(torch.randn(2, 4)) + 0.1
        run_onnx_test(log_script.to_model_proto, x, torch.log(x))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_matmul_all_opsets(self, opset):
        """MatMul should work identically across all opsets."""

        @script(default_opset=opset)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.MatMul(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        run_onnx_test(matmul_script.to_model_proto, (x, y), torch.matmul(x, y))


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

        x = torch.tensor([1, 2, 4], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = torch.tensor([2, 8, 32], dtype=torch.int32)

        run_onnx_test(model, (x, y), expected)

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

        x = torch.tensor([16, 32, 64], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = torch.tensor([8, 8, 8], dtype=torch.int32)

        run_onnx_test(model, (x, y), expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_bitshift_left_all_opsets(self, opset):
        """BitShift LEFT should work across all opsets (11+)."""
        x_input = helper.make_tensor_value_info("x", TensorProto.INT32, [3])
        y_input = helper.make_tensor_value_info("y", TensorProto.INT32, [3])
        output = helper.make_tensor_value_info("output", TensorProto.INT32, [3])

        shift_node = helper.make_node(
            "BitShift",
            ["x", "y"],
            ["output"],
            direction="LEFT",
        )

        graph = helper.make_graph([shift_node], "test", [x_input, y_input], [output])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        x = torch.tensor([1, 2, 4], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = x << y

        run_onnx_test(model, (x, y), expected)
