# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic shape support."""

import io


import torch
import torch.nn as nn
import onnx

from conftest import run_onnx_test, convert_onnx_model


def export_to_onnx_dynamic(
    model: nn.Module,
    input_shape: tuple,
    dynamic_axes: dict,
) -> onnx.ModelProto:
    """Export PyTorch model to ONNX with dynamic axes."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    buffer.seek(0)
    return onnx.load(buffer)


class TestDynamicBatchSize:
    """Test models with dynamic batch size."""

    def test_simple_linear_dynamic_batch(self):
        """Test simple linear model with dynamic batch size."""

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc(x)

        model = SimpleLinear()
        model.eval()

        # Export with dynamic batch
        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 64),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            test_input = torch.randn(batch_size, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )

    def test_conv_dynamic_batch(self):
        """Test CNN with dynamic batch size."""

        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.relu(self.bn(self.conv(x)))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleCNN()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 3, 32, 32),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 3, 32, 32)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )


class TestDynamicSequenceLength:
    """Test models with dynamic sequence length."""

    def test_rnn_dynamic_sequence(self):
        """Test model with dynamic sequence length using matmul."""

        class SimpleSequenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(64, 64)
                self.norm = nn.LayerNorm(64)

            def forward(self, x):
                # x: [batch, seq, hidden]
                x = self.proj(x)
                x = self.norm(x)
                return x

        model = SimpleSequenceModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different sequence lengths
        for seq_len in [5, 10, 20, 50]:
            test_input = torch.randn(2, seq_len, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )

    def test_attention_dynamic_sequence(self):
        """Test attention with dynamic sequence length."""

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

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different sequence lengths
        for seq_len in [5, 10, 20]:
            test_input = torch.randn(2, seq_len, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-4,
            )


class TestDynamicImageSize:
    """Test models with dynamic image size."""

    def test_fully_conv_dynamic_size(self):
        """Test fully convolutional network with dynamic image size."""

        class FullyConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 10, 1)  # 1x1 conv for output

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.conv3(x)
                return x

        model = FullyConv()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(1, 3, 64, 64),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different image sizes
        for h, w in [(32, 32), (64, 64), (128, 96)]:
            test_input = torch.randn(1, 3, h, w)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )


class TestDynamicReshape:
    """Test reshape operations with dynamic dimensions."""

    def test_flatten_dynamic_batch(self):
        """Test flattening with dynamic batch size."""

        class FlattenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 8 * 8, 10)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)  # Dynamic flatten
                return self.fc(x)

        model = FlattenModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 3, 8, 8),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 3, 8, 8)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )

    def test_reshape_preserve_batch(self):
        """Test reshape that preserves batch dimension."""

        class ReshapeModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # x: [batch, 64] -> [batch, 8, 8]
                batch = x.size(0)
                return x.view(batch, 8, 8)

        model = ReshapeModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 64),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-5,
                atol=1e-6,
            )


class TestDynamicCat:
    """Test concatenation with dynamic shapes."""

    def test_cat_dynamic_batch(self):
        """Test concatenation with dynamic batch size."""

        class CatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 16)
                self.fc2 = nn.Linear(32, 16)

            def forward(self, x):
                y1 = self.fc1(x)
                y2 = self.fc2(x)
                return torch.cat([y1, y2], dim=1)

        model = CatModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 32),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 32)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )


class TestDynamicReduce:
    """Test reduction operations with dynamic shapes."""

    def test_mean_dynamic_batch(self):
        """Test mean reduction with dynamic batch."""

        class MeanModel(nn.Module):
            def forward(self, x):
                # Global average over spatial dims
                return x.mean(dim=[2, 3], keepdim=True)

        model = MeanModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 64, 7, 7),
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        fx_module = convert_onnx_model(onnx_model)

        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 64, 7, 7)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )

    def test_sum_dynamic_sequence(self):
        """Test sum reduction with dynamic sequence length."""

        class SumModel(nn.Module):
            def forward(self, x):
                return x.sum(dim=1)

        model = SumModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        for seq_len in [5, 10, 20]:
            test_input = torch.randn(2, seq_len, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-4,
                atol=1e-5,
            )


class TestDynamicBroadcast:
    """Test broadcast operations with dynamic shapes."""

    def test_add_broadcast_dynamic(self):
        """Test addition with broadcasting and dynamic batch."""

        class BroadcastModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(64))

            def forward(self, x):
                # x: [batch, seq, 64], bias: [64]
                return x + self.bias

        model = BroadcastModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        for batch_size in [1, 4]:
            for seq_len in [5, 10, 20]:
                test_input = torch.randn(batch_size, seq_len, 64)
                run_onnx_test(
                    fx_module,
                    test_input,
                    lambda x: model(x),
                    rtol=1e-5,
                    atol=1e-6,
                )


class TestDynamicSlice:
    """Test slice operations with dynamic shapes."""

    def test_slice_dynamic_dim(self):
        """Test slicing with dynamic dimension."""

        class SliceModel(nn.Module):
            def forward(self, x):
                # Take first half of sequence
                seq_len = x.size(1)
                return x[:, : seq_len // 2, :]

        model = SliceModel()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "half_seq"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        for seq_len in [10, 20, 40]:
            test_input = torch.randn(2, seq_len, 64)
            run_onnx_test(
                fx_module,
                test_input,
                lambda x: model(x),
                rtol=1e-5,
                atol=1e-6,
            )


class TestComplexDynamicModels:
    """Test complex models with multiple dynamic dimensions."""

    def test_transformer_block_dynamic(self):
        """Test transformer block with dynamic batch and sequence."""

        class TransformerBlock(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                x = self.norm2(x + self.ffn(x))
                return x

        model = TransformerBlock(d_model=64, nhead=4)
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(2, 10, 64),
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        for batch_size in [1, 4]:
            for seq_len in [5, 10, 20]:
                test_input = torch.randn(batch_size, seq_len, 64)
                run_onnx_test(
                    fx_module,
                    test_input,
                    lambda x: model(x),
                    rtol=1e-3,
                    atol=1e-4,
                )

    def test_encoder_decoder_dynamic(self):
        """Test encoder-decoder like model with dynamic dimensions."""

        class EncoderDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(
                        64, 32, 3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                )

            def forward(self, x):
                enc = self.encoder(x)
                dec = self.decoder(enc)
                return dec

        model = EncoderDecoder()
        model.eval()

        onnx_model = export_to_onnx_dynamic(
            model,
            input_shape=(1, 3, 64, 64),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
        fx_module = convert_onnx_model(onnx_model)

        for batch_size in [1, 2]:
            for size in [(64, 64), (128, 128)]:
                test_input = torch.randn(batch_size, 3, *size)
                run_onnx_test(
                    fx_module,
                    test_input,
                    lambda x: model(x),
                    rtol=1e-4,
                    atol=1e-5,
                )
