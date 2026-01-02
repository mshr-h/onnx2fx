# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests with real models."""

import io
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from onnx2fx import convert


def export_to_onnx(
    model: nn.Module, input_shape: tuple, opset_version: int = 21
) -> onnx.ModelProto:
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    buffer.seek(0)
    return onnx.load(buffer)


def compare_outputs(
    onnx_model: onnx.ModelProto,
    fx_module: torch.fx.GraphModule,
    test_input: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Compare outputs between ONNX Runtime and FX module."""
    # Run ONNX Runtime
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(onnx_model, f.name)
        session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: test_input.numpy()})

    # Run FX module
    fx_module.eval()
    with torch.inference_mode():
        fx_output = fx_module(test_input)

    # Compare
    if isinstance(fx_output, torch.Tensor):
        fx_output = fx_output.numpy()
    else:
        fx_output = (
            fx_output[0].numpy()
            if isinstance(fx_output[0], torch.Tensor)
            else fx_output[0]
        )

    return np.allclose(ort_outputs[0], fx_output, rtol=rtol, atol=atol)


class TestSimpleCNN:
    """Test simple CNN models."""

    def test_simple_conv_net(self):
        """Test a simple convolutional network."""

        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 10)

            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                x = self.gap(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleCNN()
        model.eval()

        input_shape = (1, 3, 32, 32)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_residual_block(self):
        """Test a residual block pattern."""

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + identity  # Skip connection
                out = self.relu(out)
                return out

        model = ResidualBlock(16)
        model.eval()

        input_shape = (1, 16, 32, 32)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestMLP:
    """Test Multi-Layer Perceptron models."""

    def test_simple_mlp(self):
        """Test a simple MLP."""

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(256, 128)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.fc3(x)
                return x

        model = SimpleMLP()
        model.eval()

        input_shape = (1, 784)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_mlp_with_dropout(self):
        """Test MLP with dropout (inference mode)."""

        class MLPWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 128)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = MLPWithDropout()
        model.eval()  # Dropout disabled in eval mode

        input_shape = (2, 256)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestTransformerComponents:
    """Test Transformer components."""

    def test_multi_head_attention(self):
        """Test MultiheadAttention layer."""

        class SimpleAttention(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True
                )

            def forward(self, x):
                attn_output, _ = self.attn(x, x, x)
                return attn_output

        model = SimpleAttention(embed_dim=64, num_heads=4)
        model.eval()

        input_shape = (2, 10, 64)  # batch, seq_len, embed_dim
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-4)

    def test_transformer_encoder_layer(self):
        """Test TransformerEncoderLayer."""

        class SimpleTransformer(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    batch_first=True,
                    dim_feedforward=128,
                )

            def forward(self, x):
                return self.encoder_layer(x)

        model = SimpleTransformer(d_model=64, nhead=4)
        model.eval()

        input_shape = (2, 10, 64)  # batch, seq_len, d_model
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-4)


class TestNormalizationLayers:
    """Test various normalization layers."""

    def test_batch_norm(self):
        """Test BatchNorm2d."""

        class BNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(16)

            def forward(self, x):
                return self.bn(x)

        model = BNModel()
        model.eval()

        input_shape = (2, 16, 8, 8)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_layer_norm(self):
        """Test LayerNorm."""

        class LNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm([64])

            def forward(self, x):
                return self.ln(x)

        model = LNModel()
        model.eval()

        input_shape = (2, 10, 64)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_instance_norm(self):
        """Test InstanceNorm2d."""

        class INModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.instance_norm = nn.InstanceNorm2d(16)

            def forward(self, x):
                return self.instance_norm(x)

        model = INModel()
        model.eval()

        input_shape = (2, 16, 8, 8)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestPoolingLayers:
    """Test pooling layers."""

    def test_max_pool(self):
        """Test MaxPool2d."""

        class MaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                return self.pool(x)

        model = MaxPoolModel()
        model.eval()

        input_shape = (1, 3, 16, 16)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected)

    def test_avg_pool(self):
        """Test AvgPool2d."""

        class AvgPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                return self.pool(x)

        model = AvgPoolModel()
        model.eval()

        input_shape = (1, 3, 16, 16)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected)

    def test_global_avg_pool(self):
        """Test Global Average Pooling."""

        class GAPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gap = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                return self.gap(x)

        model = GAPModel()
        model.eval()

        input_shape = (2, 64, 7, 7)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestActivations:
    """Test various activation functions in full models."""

    def test_model_with_various_activations(self):
        """Test model with various activations."""

        class ActivationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 64)
                self.fc4 = nn.Linear(64, 10)
                self.relu = nn.ReLU()
                self.gelu = nn.GELU()
                self.silu = nn.SiLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.gelu(self.fc2(x))
                x = self.silu(self.fc3(x))
                x = self.fc4(x)
                return x

        model = ActivationModel()
        model.eval()

        input_shape = (2, 64)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestComplexModels:
    """Test more complex model architectures."""

    def test_unet_block(self):
        """Test a U-Net-like encoder-decoder block."""

        class UNetBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                # Encoder
                self.enc_conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                self.enc_bn1 = nn.BatchNorm2d(out_ch)
                self.enc_conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                self.enc_bn2 = nn.BatchNorm2d(out_ch)
                self.pool = nn.MaxPool2d(2)

                # Decoder
                self.up = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
                self.dec_conv1 = nn.Conv2d(out_ch * 2, out_ch, 3, padding=1)
                self.dec_bn1 = nn.BatchNorm2d(out_ch)
                self.dec_conv2 = nn.Conv2d(out_ch, in_ch, 3, padding=1)

                self.relu = nn.ReLU()

            def forward(self, x):
                # Encoder
                enc = self.relu(self.enc_bn1(self.enc_conv1(x)))
                enc = self.relu(self.enc_bn2(self.enc_conv2(enc)))
                down = self.pool(enc)

                # Decoder
                up = self.up(down)
                concat = torch.cat([up, enc], dim=1)
                dec = self.relu(self.dec_bn1(self.dec_conv1(concat)))
                out = self.dec_conv2(dec)
                return out

        model = UNetBlock(3, 16)
        model.eval()

        input_shape = (1, 3, 32, 32)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_inception_module(self):
        """Test an Inception-like module."""

        class InceptionModule(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                # Branch 1: 1x1 conv
                self.branch1 = nn.Conv2d(in_channels, 16, 1)

                # Branch 2: 1x1 -> 3x3
                self.branch2_1 = nn.Conv2d(in_channels, 16, 1)
                self.branch2_2 = nn.Conv2d(16, 24, 3, padding=1)

                # Branch 3: 1x1 -> 5x5
                self.branch3_1 = nn.Conv2d(in_channels, 16, 1)
                self.branch3_2 = nn.Conv2d(16, 24, 5, padding=2)

                # Branch 4: pool -> 1x1
                self.branch4_pool = nn.MaxPool2d(3, stride=1, padding=1)
                self.branch4_conv = nn.Conv2d(in_channels, 16, 1)

                self.relu = nn.ReLU()

            def forward(self, x):
                b1 = self.relu(self.branch1(x))
                b2 = self.relu(self.branch2_2(self.relu(self.branch2_1(x))))
                b3 = self.relu(self.branch3_2(self.relu(self.branch3_1(x))))
                b4 = self.relu(self.branch4_conv(self.branch4_pool(x)))
                return torch.cat([b1, b2, b3, b4], dim=1)

        model = InceptionModule(32)
        model.eval()

        input_shape = (1, 32, 14, 14)
        onnx_model = export_to_onnx(model, input_shape)

        fx_module = convert(onnx_model)

        test_input = torch.randn(*input_shape)

        with torch.inference_mode():
            expected = model(test_input)
            result = fx_module(test_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
