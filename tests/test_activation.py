# SPDX-License-Identifier: Apache-2.0
"""Tests for activation operators."""

import torch
from onnxscript import FLOAT, script
from onnxscript import opset15 as op

from onnx2fx.converter import convert


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
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.relu(x))

    def test_sigmoid(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.sigmoid_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.sigmoid(x))

    def test_tanh(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.tanh_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.tanh(x))

    def test_softmax(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.softmax_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.softmax(x, dim=-1))

    def test_leaky_relu(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.leaky_relu_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.nn.functional.leaky_relu(x, 0.1))

    def test_elu(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.elu_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.nn.functional.elu(x, 1.0))

    def test_softplus(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.softplus_script.to_model_proto())
        with torch.no_grad():
            result = fx_model(x)
        assert torch.allclose(result, torch.nn.functional.softplus(x))
