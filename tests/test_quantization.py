# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization operators."""

import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
import torch
from onnxscript import opset13, opset14, opset15, opset16, opset17
from onnxscript import opset18, opset19, opset20, opset21, opset22, opset23

from conftest import OPSET_MODULES, opset_id, run_onnx_test


class TestQuantizeLinear:
    """Test QuantizeLinear operator."""

    def test_quantize_with_zero_point(self):
        """Test quantization with zero point (uint8)."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
        zero_point = numpy_helper.from_array(
            np.array(128, dtype=np.uint8), "zero_point"
        )

        node = helper.make_node(
            "QuantizeLinear",
            inputs=["x", "scale", "zero_point"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale, zero_point])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        test_input = torch.randn(2, 3)
        expected = torch.clamp(torch.round(test_input / 0.1) + 128, 0, 255).to(
            torch.uint8
        )
        run_onnx_test(model, test_input, expected)

    def test_quantize_without_zero_point(self):
        """Test quantization without zero point (int8)."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")

        node = helper.make_node(
            "QuantizeLinear",
            inputs=["x", "scale"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        test_input = torch.randn(2, 3)
        expected = torch.clamp(torch.round(test_input / 0.1), -128, 127).to(torch.int8)
        run_onnx_test(model, test_input, expected)


class TestDequantizeLinear:
    """Test DequantizeLinear operator."""

    def test_dequantize_with_zero_point(self):
        """Test dequantization with zero point."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
        zero_point = numpy_helper.from_array(
            np.array(128, dtype=np.uint8), "zero_point"
        )

        node = helper.make_node(
            "DequantizeLinear",
            inputs=["x", "scale", "zero_point"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale, zero_point])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        test_input = torch.randint(0, 256, (2, 3), dtype=torch.uint8)
        expected = (test_input.float() - 128) * 0.1
        run_onnx_test(model, test_input, expected)

    def test_dequantize_without_zero_point(self):
        """Test dequantization without zero point."""
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")

        node = helper.make_node(
            "DequantizeLinear",
            inputs=["x", "scale"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        test_input = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
        expected = test_input.float() * 0.1
        run_onnx_test(model, test_input, expected)


class TestDynamicQuantizeLinear:
    """Test DynamicQuantizeLinear operator."""

    def test_dynamic_quantize(self):
        """Test dynamic quantization."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])
        y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [])
        y_zero_point = helper.make_tensor_value_info(
            "y_zero_point", TensorProto.UINT8, []
        )

        node = helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
        )

        graph = helper.make_graph([node], "test", [x], [y, y_scale, y_zero_point])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        test_input = torch.randn(2, 3) * 10  # Larger range
        expected = (
            torch.empty_like(test_input, dtype=torch.uint8),
            torch.empty((), dtype=torch.float32),
            torch.empty((), dtype=torch.uint8),
        )
        fx_module = run_onnx_test(model, test_input, expected, check_shape_only=True)
        result = fx_module(test_input)

        # Result is a tuple (y, scale, zero_point)
        assert len(result) == 3
        assert result[0].dtype == torch.uint8
        assert result[0].shape == (2, 3)


class TestQLinearMatMul:
    """Test QLinearMatMul operator."""

    def test_qlinear_matmul(self):
        """Test quantized matrix multiplication."""
        a = helper.make_tensor_value_info("a", TensorProto.UINT8, [2, 3])
        b = helper.make_tensor_value_info("b", TensorProto.UINT8, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 4])

        a_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "a_scale")
        a_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "a_zp")
        b_scale = numpy_helper.from_array(np.array(0.2, dtype=np.float32), "b_scale")
        b_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "b_zp")
        y_scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp")

        node = helper.make_node(
            "QLinearMatMul",
            inputs=["a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"],
            outputs=["y"],
        )

        graph = helper.make_graph(
            [node], "test", [a, b], [y], [a_scale, a_zp, b_scale, b_zp, y_scale, y_zp]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        a_input = torch.randint(0, 256, (2, 3), dtype=torch.uint8)
        b_input = torch.randint(0, 256, (3, 4), dtype=torch.uint8)

        expected = torch.empty(2, 4, dtype=torch.uint8)
        fx_module = run_onnx_test(
            model, (a_input, b_input), expected, check_shape_only=True
        )
        result = fx_module(a_input, b_input)

        assert result.dtype == torch.uint8
        assert result.shape == (2, 4)


class TestQLinearConv:
    """Test QLinearConv operator."""

    def test_qlinear_conv2d(self):
        """Test quantized 2D convolution."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 8, 8])
        w = helper.make_tensor_value_info("w", TensorProto.UINT8, [16, 3, 3, 3])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 16, 6, 6])

        x_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "x_scale")
        x_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp")
        w_scale = numpy_helper.from_array(np.array(0.05, dtype=np.float32), "w_scale")
        w_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "w_zp")
        y_scale = numpy_helper.from_array(np.array(0.2, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp")

        node = helper.make_node(
            "QLinearConv",
            inputs=["x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"],
            outputs=["y"],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        graph = helper.make_graph(
            [node], "test", [x, w], [y], [x_scale, x_zp, w_scale, w_zp, y_scale, y_zp]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x_input = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        w_input = torch.randint(0, 256, (16, 3, 3, 3), dtype=torch.uint8)

        expected = torch.empty(1, 16, 6, 6, dtype=torch.uint8)
        fx_module = run_onnx_test(
            model, (x_input, w_input), expected, check_shape_only=True
        )
        result = fx_module(x_input, w_input)

        assert result.dtype == torch.uint8
        assert result.shape == (1, 16, 6, 6)


class TestQLinearActivations:
    """Test quantized activation operators."""

    def test_qlinear_sigmoid(self):
        """Test quantized sigmoid."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])

        x_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "x_scale")
        x_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp")
        y_scale = numpy_helper.from_array(np.array(0.004, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "y_zp")

        node = helper.make_node(
            "QLinearSigmoid",
            inputs=["x", "x_scale", "x_zp", "y_scale", "y_zp"],
            outputs=["y"],
        )

        graph = helper.make_graph(
            [node], "test", [x], [y], [x_scale, x_zp, y_scale, y_zp]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x_input = torch.randint(0, 256, (2, 3), dtype=torch.uint8)

        expected = torch.empty(2, 3, dtype=torch.uint8)
        fx_module = run_onnx_test(model, x_input, expected, check_shape_only=True)
        result = fx_module(x_input)

        assert result.dtype == torch.uint8
        assert result.shape == (2, 3)

    def test_qlinear_leaky_relu(self):
        """Test quantized leaky relu."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])

        x_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "x_scale")
        x_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp")
        y_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp")

        node = helper.make_node(
            "QLinearLeakyRelu",
            inputs=["x", "x_scale", "x_zp", "y_scale", "y_zp"],
            outputs=["y"],
            alpha=0.01,
        )

        graph = helper.make_graph(
            [node], "test", [x], [y], [x_scale, x_zp, y_scale, y_zp]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x_input = torch.randint(0, 256, (2, 3), dtype=torch.uint8)

        expected = torch.empty(2, 3, dtype=torch.uint8)
        fx_module = run_onnx_test(model, x_input, expected, check_shape_only=True)
        result = fx_module(x_input)

        assert result.dtype == torch.uint8
        assert result.shape == (2, 3)


class TestQLinearGlobalAveragePool:
    """Test QLinearGlobalAveragePool operator."""

    def test_qlinear_global_avg_pool(self):
        """Test quantized global average pooling."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 64, 7, 7])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 64, 1, 1])

        x_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "x_scale")
        x_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp")
        y_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp")

        node = helper.make_node(
            "QLinearGlobalAveragePool",
            inputs=["x", "x_scale", "x_zp", "y_scale", "y_zp"],
            outputs=["y"],
        )

        graph = helper.make_graph(
            [node], "test", [x], [y], [x_scale, x_zp, y_scale, y_zp]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x_input = torch.randint(0, 256, (1, 64, 7, 7), dtype=torch.uint8)

        expected = torch.empty(1, 64, 1, 1, dtype=torch.uint8)
        fx_module = run_onnx_test(model, x_input, expected, check_shape_only=True)
        result = fx_module(x_input)

        assert result.dtype == torch.uint8
        assert result.shape == (1, 64, 1, 1)


class TestMatMulInteger:
    """Test MatMulInteger operator."""

    def test_matmul_integer(self):
        """Test integer matrix multiplication."""
        a = helper.make_tensor_value_info("a", TensorProto.UINT8, [2, 3])
        b = helper.make_tensor_value_info("b", TensorProto.UINT8, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])

        a_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "a_zp")
        b_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "b_zp")

        node = helper.make_node(
            "MatMulInteger",
            inputs=["a", "b", "a_zp", "b_zp"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [a, b], [y], [a_zp, b_zp])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        a_input = torch.randint(0, 256, (2, 3), dtype=torch.uint8)
        b_input = torch.randint(0, 256, (3, 4), dtype=torch.uint8)

        # Verify manually
        a_int = a_input.int() - 128
        b_int = b_input.int() - 128
        expected = torch.matmul(a_int.float(), b_int.float()).int()
        run_onnx_test(model, (a_input, b_input), expected)


class TestConvInteger:
    """Test ConvInteger operator."""

    def test_conv_integer(self):
        """Test integer convolution."""
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 8, 8])
        w = helper.make_tensor_value_info("w", TensorProto.UINT8, [16, 3, 3, 3])
        y = helper.make_tensor_value_info("y", TensorProto.INT32, [1, 16, 6, 6])

        x_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp")
        w_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "w_zp")

        node = helper.make_node(
            "ConvInteger",
            inputs=["x", "w", "x_zp", "w_zp"],
            outputs=["y"],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        graph = helper.make_graph([node], "test", [x, w], [y], [x_zp, w_zp])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x_input = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        w_input = torch.randint(0, 256, (16, 3, 3, 3), dtype=torch.uint8)

        expected = torch.empty(1, 16, 6, 6, dtype=torch.int32)
        fx_module = run_onnx_test(
            model, (x_input, w_input), expected, check_shape_only=True
        )
        result = fx_module(x_input, w_input)

        assert result.dtype == torch.int32
        assert result.shape == (1, 16, 6, 6)


class TestQuantizedPipeline:
    """Test quantized model pipelines."""

    def test_quantize_dequantize_roundtrip(self):
        """Test quantize then dequantize."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

        # Use scale that covers the typical input range [-1, 1]
        scale = numpy_helper.from_array(np.array(0.01, dtype=np.float32), "scale")
        zero_point = numpy_helper.from_array(
            np.array(128, dtype=np.uint8), "zero_point"
        )

        quant_node = helper.make_node(
            "QuantizeLinear",
            inputs=["x", "scale", "zero_point"],
            outputs=["q"],
        )
        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=["q", "scale", "zero_point"],
            outputs=["y"],
        )

        graph = helper.make_graph(
            [quant_node, dequant_node], "test", [x], [y], [scale, zero_point]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        # Use input in the quantizable range: [-1.28, 1.27] with scale=0.01
        test_input = torch.clamp(torch.randn(2, 3), -1.2, 1.2)
        # Result should be close to input (within quantization error)
        run_onnx_test(model, test_input, test_input, rtol=0.1, atol=0.01)

    def test_quantized_matmul_chain(self):
        """Test chain of quantized operations."""
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])

        # Scales and zero points
        a_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "a_scale")
        a_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "a_zp")
        b_scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "b_scale")
        b_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "b_zp")
        y_scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "y_scale")
        y_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp")

        # Quantize A
        quant_a = helper.make_node(
            "QuantizeLinear",
            inputs=["a", "a_scale", "a_zp"],
            outputs=["a_q"],
        )
        # Quantize B
        quant_b = helper.make_node(
            "QuantizeLinear",
            inputs=["b", "b_scale", "b_zp"],
            outputs=["b_q"],
        )
        # QLinearMatMul
        qmm = helper.make_node(
            "QLinearMatMul",
            inputs=[
                "a_q",
                "a_scale",
                "a_zp",
                "b_q",
                "b_scale",
                "b_zp",
                "y_scale",
                "y_zp",
            ],
            outputs=["y_q"],
        )
        # Dequantize output
        dequant = helper.make_node(
            "DequantizeLinear",
            inputs=["y_q", "y_scale", "y_zp"],
            outputs=["y"],
        )

        graph = helper.make_graph(
            [quant_a, quant_b, qmm, dequant],
            "test",
            [a, b],
            [y],
            [a_scale, a_zp, b_scale, b_zp, y_scale, y_zp],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        a_input = torch.randn(2, 3)
        b_input = torch.randn(3, 4)
        # Compare with float matmul (with quantization error tolerance)
        expected = torch.matmul(a_input, b_input)
        run_onnx_test(model, (a_input, b_input), expected, rtol=0.2, atol=0.5)


class TestQuantizationMultiOpset:
    """Test quantization operators across multiple opset versions."""

    @pytest.mark.parametrize(
        "opset",
        [
            opset13,
            opset14,
            opset15,
            opset16,
            opset17,
            opset18,
            opset19,
            opset20,
            opset21,
            opset22,
            opset23,
        ],
        ids=opset_id,
    )
    def test_quantize_linear_all_opsets(self, opset):
        """QuantizeLinear should work across opsets (13+, with int output)."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")

        node = helper.make_node(
            "QuantizeLinear",
            inputs=["x", "scale"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        test_input = torch.randn(2, 3)
        expected = torch.clamp(torch.round(test_input / 0.1), -128, 127).to(torch.int8)
        run_onnx_test(model, test_input, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_dequantize_linear_all_opsets(self, opset):
        """DequantizeLinear should work across all opsets (10+)."""
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")

        node = helper.make_node(
            "DequantizeLinear",
            inputs=["x", "scale"],
            outputs=["y"],
        )

        graph = helper.make_graph([node], "test", [x], [y], [scale])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        test_input = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
        expected = test_input.float() * 0.1
        run_onnx_test(model, test_input, expected)
