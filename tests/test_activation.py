# SPDX-License-Identifier: Apache-2.0
"""Tests for activation operators."""

import pytest
import torch
import torch.nn.functional as F
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES


class TestActivationOps:
    """Test activation function operators."""

    @script()
    def relu_script(x: FLOAT) -> FLOAT:
        return op.Relu(x)

    @script()
    def sigmoid_script(x: FLOAT) -> FLOAT:
        return op.Sigmoid(x)

    @script()
    def tanh_script(x: FLOAT) -> FLOAT:
        return op.Tanh(x)

    @script()
    def softmax_script(x: FLOAT) -> FLOAT:
        return op.Softmax(x, axis=-1)

    @script()
    def leaky_relu_script(x: FLOAT) -> FLOAT:
        return op.LeakyRelu(x, alpha=0.1)

    @script()
    def elu_script(x: FLOAT) -> FLOAT:
        return op.Elu(x, alpha=1.0)

    @script()
    def softplus_script(x: FLOAT) -> FLOAT:
        return op.Softplus(x)

    def test_relu(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.relu_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.relu(x))

    def test_sigmoid(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.sigmoid_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.sigmoid(x))

    def test_tanh(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.tanh_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.tanh(x))

    def test_softmax(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.softmax_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.softmax(x, dim=-1))

    def test_leaky_relu(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.leaky_relu_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.nn.functional.leaky_relu(x, 0.1))

    def test_elu(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.elu_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.nn.functional.elu(x, 1.0))

    def test_softplus(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.softplus_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.nn.functional.softplus(x))


class TestActivationOpsMultiOpset:
    """Test activation operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_relu_all_opsets(self, opset):
        """Relu should work identically across all opsets."""

        @script(default_opset=opset)
        def relu_script(x: FLOAT) -> FLOAT:
            return opset.Relu(x)

        model = relu_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.relu(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_sigmoid_all_opsets(self, opset):
        """Sigmoid should work identically across all opsets."""

        @script(default_opset=opset)
        def sigmoid_script(x: FLOAT) -> FLOAT:
            return opset.Sigmoid(x)

        model = sigmoid_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.sigmoid(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_tanh_all_opsets(self, opset):
        """Tanh should work identically across all opsets."""

        @script(default_opset=opset)
        def tanh_script(x: FLOAT) -> FLOAT:
            return opset.Tanh(x)

        model = tanh_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.tanh(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_leaky_relu_all_opsets(self, opset):
        """LeakyRelu should work identically across all opsets."""

        @script(default_opset=opset)
        def leaky_relu_script(x: FLOAT) -> FLOAT:
            return opset.LeakyRelu(x, alpha=0.1)

        model = leaky_relu_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = F.leaky_relu(x, 0.1)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_elu_all_opsets(self, opset):
        """Elu should work identically across all opsets."""

        @script(default_opset=opset)
        def elu_script(x: FLOAT) -> FLOAT:
            return opset.Elu(x, alpha=1.0)

        model = elu_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = F.elu(x, 1.0)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_softplus_all_opsets(self, opset):
        """Softplus should work identically across all opsets."""

        @script(default_opset=opset)
        def softplus_script(x: FLOAT) -> FLOAT:
            return opset.Softplus(x)

        model = softplus_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = F.softplus(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_softmax_with_axis_all_opsets(self, opset):
        """Softmax with explicit axis should work across all opsets."""

        @script(default_opset=opset)
        def softmax_script(x: FLOAT) -> FLOAT:
            return opset.Softmax(x, axis=-1)

        model = softmax_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected)
