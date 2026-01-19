# SPDX-License-Identifier: Apache-2.0
"""Tests for neural network layer operators."""

import pytest
import torch
import torch.nn.functional as F
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES, opset_id, run_onnx_test


class TestMatMulOps:
    """Test matrix multiplication operators."""

    @script()
    def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.MatMul(x, y)

    def test_matmul(self):
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        run_onnx_test(self.matmul_script.to_model_proto, (x, y), torch.matmul(x, y))

    def test_matmul_batched(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        run_onnx_test(self.matmul_script.to_model_proto, (x, y), torch.matmul(x, y))


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

        A = torch.randn(2, 3)
        B = torch.randn(3, 4)
        C = torch.randn(4)
        expected = torch.matmul(A, B) + C

        run_onnx_test(model, (A, B, C), expected, rtol=1e-5, atol=1e-5)


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

        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)
        expected = torch.nn.functional.conv2d(x, w, padding=1)

        run_onnx_test(model, (x, w), expected, rtol=1e-5, atol=1e-5)

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

        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)
        b = torch.randn(16)
        expected = torch.nn.functional.conv2d(x, w, bias=b)

        run_onnx_test(model, (x, w, b), expected, rtol=1e-5, atol=1e-5)


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

        x = torch.randn(1, 3, 8, 8)
        expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        run_onnx_test(model, x, expected)

    def test_max_pool2d_auto_pad_same_upper(self):
        """Test MaxPool2D with auto_pad=SAME_UPPER."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 7, 7]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        pool_node = onnx.helper.make_node(
            "MaxPool",
            ["X"],
            ["Y"],
            kernel_shape=[2, 2],
            strides=[1, 1],
            auto_pad="SAME_UPPER",
        )

        graph = onnx.helper.make_graph([pool_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 11)]
        )
        model.ir_version = 6  # Compatible with ONNX Runtime

        fx_model = convert(model)

        x = torch.randn(1, 3, 7, 7)

        with torch.inference_mode():
            result = fx_model(x)

        # SAME_UPPER should preserve spatial dimensions
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

        # Compare with ONNX Runtime
        import numpy as np
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_output = ort_session.run(None, {"X": x.numpy()})[0]
        np.testing.assert_allclose(result.numpy(), ort_output, rtol=1e-5, atol=1e-5)

    def test_max_pool2d_auto_pad_same_lower(self):
        """Test MaxPool2D with auto_pad=SAME_LOWER."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3, 7, 7]
        )
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

        pool_node = onnx.helper.make_node(
            "MaxPool",
            ["X"],
            ["Y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad="SAME_LOWER",
        )

        graph = onnx.helper.make_graph([pool_node], "test", [x_info], [y_info])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 11)]
        )
        model.ir_version = 6  # Compatible with ONNX Runtime

        fx_model = convert(model)

        x = torch.randn(1, 3, 7, 7)

        with torch.inference_mode():
            result = fx_model(x)

        # Compare with ONNX Runtime
        import numpy as np
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_output = ort_session.run(None, {"X": x.numpy()})[0]
        np.testing.assert_allclose(result.numpy(), ort_output, rtol=1e-5, atol=1e-5)

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

        x = torch.randn(1, 3, 8, 8)
        expected = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        run_onnx_test(model, x, expected)

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

        x = torch.randn(1, 3, 8, 8)
        expected = x.mean(dim=(2, 3), keepdim=True)

        run_onnx_test(model, x, expected)


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

        x = torch.randn(2, 3, 4, 4)
        scale = torch.ones(3)
        bias = torch.zeros(3)
        mean = torch.zeros(3)
        var = torch.ones(3)
        expected = torch.nn.functional.batch_norm(x, mean, var, scale, bias, eps=1e-5)

        run_onnx_test(
            model, (x, scale, bias, mean, var), expected, rtol=1e-5, atol=1e-5
        )

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

        x = torch.randn(2, 3, 4)
        scale = torch.ones(4)
        bias = torch.zeros(4)
        expected = torch.nn.functional.layer_norm(x, [4], scale, bias, eps=1e-5)

        run_onnx_test(model, (x, scale, bias), expected, rtol=1e-5, atol=1e-5)


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

        x = torch.randn(2, 3)

        # In inference mode, dropout should be identity
        run_onnx_test(model, x, x)


class TestNNOpsMultiOpset:
    """Test neural network operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_matmul_all_opsets(self, opset):
        """MatMul should work identically across all opsets."""

        @script(default_opset=opset)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.MatMul(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        run_onnx_test(matmul_script.to_model_proto, (x, y), torch.matmul(x, y))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_matmul_batched_all_opsets(self, opset):
        """Batched MatMul should work identically across all opsets."""

        @script(default_opset=opset)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return opset.MatMul(x, y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        run_onnx_test(matmul_script.to_model_proto, (x, y), torch.matmul(x, y))

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
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

        x = torch.randn(2, 3, 4, 4)
        scale = torch.ones(3)
        bias = torch.zeros(3)
        mean = torch.zeros(3)
        var = torch.ones(3)
        expected = F.batch_norm(x, mean, var, scale, bias, training=False, eps=1e-5)

        run_onnx_test(
            model, (x, scale, bias, mean, var), expected, atol=1e-5, rtol=1e-5
        )
