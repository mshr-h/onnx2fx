# SPDX-License-Identifier: Apache-2.0
"""Tests for neural network layer operators."""

import pytest
import torch
import torch.nn.functional as F
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES


class TestMatMulOps:
    """Test matrix multiplication operators."""

    @script()
    def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.MatMul(x, y)

    def test_matmul(self):
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        fx_model = convert(self.matmul_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.matmul(x, y))

    def test_matmul_batched(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        fx_model = convert(self.matmul_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x, y)
        torch.testing.assert_close(result, torch.matmul(x, y))


class TestGemmOp:
    """Test Gemm operator."""

    def test_gemm_basic(self):
        """Test basic Gemm: Y = A * B + C"""
        a = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 3])
        b = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [3, 4])
        c = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [4])
        y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])

        gemm_node = onnx.helper.make_node(
            "Gemm", ["A", "B", "C"], ["Y"], alpha=1.0, beta=1.0, transA=0, transB=0
        )

        graph = onnx.helper.make_graph([gemm_node], "test", [a, b, c], [y])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        A = torch.randn(2, 3)
        B = torch.randn(3, 4)
        C = torch.randn(4)

        with torch.inference_mode():
            result = fx_model(A, B, C)

        expected = torch.matmul(A, B) + C
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestConvOps:
    """Test convolution operators."""

    def test_conv2d_basic(self):
        """Test basic Conv2D."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        w_info = onnx.helper.make_tensor_value_info(
            "W", onnx.TensorProto.FLOAT, [16, 3, 3, 3]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        conv_node = onnx.helper.make_node(
            "Conv",
            ["X", "W"],
            ["Y"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
        )

        graph = onnx.helper.make_graph([conv_node], "test", [x_info, w_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)

        with torch.inference_mode():
            result = fx_model(x, w)

        expected = torch.nn.functional.conv2d(x, w, padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_conv2d_with_bias(self):
        """Test Conv2D with bias."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        w_info = onnx.helper.make_tensor_value_info(
            "W", onnx.TensorProto.FLOAT, [16, 3, 3, 3]
        )
        b_info = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [16])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        conv_node = onnx.helper.make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        graph = onnx.helper.make_graph(
            [conv_node], "test", [x_info, w_info, b_info], [y_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)
        b = torch.randn(16)

        with torch.inference_mode():
            result = fx_model(x, w, b)

        expected = torch.nn.functional.conv2d(x, w, bias=b)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestPoolingOps:
    """Test pooling operators."""

    def test_max_pool2d(self):
        """Test MaxPool2D."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        pool_node = onnx.helper.make_node(
            "MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2]
        )

        graph = onnx.helper.make_graph([pool_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(1, 3, 8, 8)

        with torch.inference_mode():
            result = fx_model(x)

        expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        torch.testing.assert_close(result, expected)

    def test_average_pool2d(self):
        """Test AveragePool2D."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        pool_node = onnx.helper.make_node(
            "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2]
        )

        graph = onnx.helper.make_graph([pool_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(1, 3, 8, 8)

        with torch.inference_mode():
            result = fx_model(x)

        expected = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        torch.testing.assert_close(result, expected)

    def test_global_average_pool(self):
        """Test GlobalAveragePool."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        pool_node = onnx.helper.make_node("GlobalAveragePool", ["X"], ["Y"])

        graph = onnx.helper.make_graph([pool_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(1, 3, 8, 8)

        with torch.inference_mode():
            result = fx_model(x)

        expected = x.mean(dim=(2, 3), keepdim=True)
        torch.testing.assert_close(result, expected)


class TestNormalizationOps:
    """Test normalization operators."""

    def test_batch_norm(self):
        """Test BatchNormalization."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [2, 3, 4, 4]
        )
        scale_info = onnx.helper.make_tensor_value_info(
            "scale", onnx.TensorProto.FLOAT, [3]
        )
        bias_info = onnx.helper.make_tensor_value_info(
            "bias", onnx.TensorProto.FLOAT, [3]
        )
        mean_info = onnx.helper.make_tensor_value_info(
            "mean", onnx.TensorProto.FLOAT, [3]
        )
        var_info = onnx.helper.make_tensor_value_info(
            "var", onnx.TensorProto.FLOAT, [3]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        bn_node = onnx.helper.make_node(
            "BatchNormalization",
            ["X", "scale", "bias", "mean", "var"],
            ["Y"],
            epsilon=1e-5,
        )

        graph = onnx.helper.make_graph(
            [bn_node],
            "test",
            [x_info, scale_info, bias_info, mean_info, var_info],
            [y_info],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3, 4, 4)
        scale = torch.ones(3)
        bias = torch.zeros(3)
        mean = torch.zeros(3)
        var = torch.ones(3)

        with torch.inference_mode():
            result = fx_model(x, scale, bias, mean, var)

        expected = torch.nn.functional.batch_norm(x, mean, var, scale, bias, eps=1e-5)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_layer_norm(self):
        """Test LayerNormalization."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [2, 3, 4]
        )
        scale_info = onnx.helper.make_tensor_value_info(
            "scale", onnx.TensorProto.FLOAT, [4]
        )
        bias_info = onnx.helper.make_tensor_value_info(
            "bias", onnx.TensorProto.FLOAT, [4]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        ln_node = onnx.helper.make_node(
            "LayerNormalization", ["X", "scale", "bias"], ["Y"], axis=-1, epsilon=1e-5
        )

        graph = onnx.helper.make_graph(
            [ln_node], "test", [x_info, scale_info, bias_info], [y_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3, 4)
        scale = torch.ones(4)
        bias = torch.zeros(4)

        with torch.inference_mode():
            result = fx_model(x, scale, bias)

        expected = torch.nn.functional.layer_norm(x, [4], scale, bias, eps=1e-5)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestDropout:
    """Test dropout operator."""

    def test_dropout_inference(self):
        """Test Dropout in inference mode (identity)."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        dropout_node = onnx.helper.make_node("Dropout", ["X"], ["Y"], ratio=0.5)

        graph = onnx.helper.make_graph([dropout_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 15)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3)

        with torch.inference_mode():
            result = fx_model(x)

        # In inference mode, dropout should be identity
        torch.testing.assert_close(result, x)


class TestNNOpsMultiOpset:
    """Test neural network operators across multiple opset versions."""

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

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_matmul_batched_all_opsets(self, opset):
        """Batched MatMul should work identically across all opsets."""

        @script(default_opset=opset)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.MatMul(x, y)

        model = matmul_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        result = fx_model(x, y)
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_batch_normalization_all_opsets(self, opset):
        """BatchNormalization should work across all opsets."""
        # Use onnx.helper for BatchNorm since it has complex inputs
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [2, 3, 4, 4]
        )
        scale_info = onnx.helper.make_tensor_value_info(
            "scale", onnx.TensorProto.FLOAT, [3]
        )
        bias_info = onnx.helper.make_tensor_value_info(
            "bias", onnx.TensorProto.FLOAT, [3]
        )
        mean_info = onnx.helper.make_tensor_value_info(
            "mean", onnx.TensorProto.FLOAT, [3]
        )
        var_info = onnx.helper.make_tensor_value_info(
            "var", onnx.TensorProto.FLOAT, [3]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        bn_node = onnx.helper.make_node(
            "BatchNormalization",
            ["X", "scale", "bias", "mean", "var"],
            ["Y"],
            epsilon=1e-5,
        )

        graph = onnx.helper.make_graph(
            [bn_node],
            "test",
            [x_info, scale_info, bias_info, mean_info, var_info],
            [y_info],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", opset.version)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3, 4, 4)
        scale = torch.ones(3)
        bias = torch.zeros(3)
        mean = torch.zeros(3)
        var = torch.ones(3)

        with torch.inference_mode():
            result = fx_model(x, scale, bias, mean, var)

        expected = F.batch_norm(x, mean, var, scale, bias, training=False, eps=1e-5)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
