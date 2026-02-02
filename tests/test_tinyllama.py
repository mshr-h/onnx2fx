# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""Tests for TinyLlama and LLaMA-like model support.

This module tests the operators commonly used in LLaMA-family models,
including TinyLlama, LLaMA, LLaMA2, Mistral, etc.

Key operators in LLaMA architecture:
- Embedding lookup (Gather)
- RMSNorm (ReduceMean, Pow, Sqrt, Div, Mul)
- Rotary Position Embeddings (Cos, Sin, Slice, Concat, Mul)
- SwiGLU activation (Sigmoid, Mul)
- Attention (MatMul, Softmax, Transpose, Reshape)
- Linear layers (MatMul, Add)
"""

import numpy as np
import onnxruntime as ort
import pytest
import tempfile
import torch
from onnxscript import FLOAT, INT64, opset14 as op
from onnxscript import script

import onnx
from conftest import run_onnx_test, convert_onnx_model


def run_and_compare(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Run model through ONNX Runtime and FX, compare outputs."""
    # Run ONNX Runtime
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])

    ort_outputs = session.run(None, inputs)

    # Run FX module - use ONNX model input order
    input_names = [inp.name for inp in model.graph.input]
    torch_inputs = tuple(torch.from_numpy(inputs[name]) for name in input_names)

    # Normalize expected structure to match FX outputs
    if len(ort_outputs) == 1:
        expected = torch.from_numpy(ort_outputs[0])
    else:
        expected = tuple(torch.from_numpy(out) for out in ort_outputs)

    run_onnx_test(
        model,
        torch_inputs[0] if len(torch_inputs) == 1 else torch_inputs,
        expected,
        rtol=rtol,
        atol=atol,
    )


class TestRMSNorm:
    """Test RMSNorm (Root Mean Square Layer Normalization) used in LLaMA."""

    def test_rmsnorm_basic(self):
        """Test basic RMSNorm implementation."""

        @script()
        def rmsnorm_model(
            x: FLOAT["B", "S", "H"], weight: FLOAT["H"]
        ) -> FLOAT["B", "S", "H"]:
            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            eps = op.Constant(
                value=onnx.numpy_helper.from_array(
                    np.array(1e-6, dtype=np.float32), name="eps"
                )
            )
            x_sq = op.Mul(x, x)
            mean_sq = op.ReduceMean(x_sq, axes=[-1], keepdims=True)
            mean_sq_eps = op.Add(mean_sq, eps)
            rms = op.Sqrt(mean_sq_eps)
            x_norm = op.Div(x, rms)
            return op.Mul(x_norm, weight)

        model = rmsnorm_model.to_model_proto()

        # Test inputs
        batch, seq, hidden = 2, 16, 64
        x = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight = np.random.randn(hidden).astype(np.float32)

        run_and_compare(model, {"x": x, "weight": weight})


class TestSwiGLU:
    """Test SwiGLU activation (Swish-Gated Linear Unit) used in LLaMA."""

    def test_swiglu_basic(self):
        """Test SwiGLU: x * sigmoid(x) * gate."""

        @script()
        def swiglu_model(
            x: FLOAT["B", "S", "H"], gate: FLOAT["B", "S", "H"]
        ) -> FLOAT["B", "S", "H"]:
            # SwiGLU: silu(x) * gate = x * sigmoid(x) * gate
            sigmoid_x = op.Sigmoid(x)
            silu_x = op.Mul(x, sigmoid_x)
            return op.Mul(silu_x, gate)

        model = swiglu_model.to_model_proto()

        batch, seq, hidden = 2, 16, 64
        x = np.random.randn(batch, seq, hidden).astype(np.float32)
        gate = np.random.randn(batch, seq, hidden).astype(np.float32)

        run_and_compare(model, {"x": x, "gate": gate})


class TestRotaryEmbedding:
    """Test Rotary Position Embeddings (RoPE) used in LLaMA."""

    def test_rope_basic(self):
        """Test basic RoPE application."""

        @script()
        def rope_model(
            x: FLOAT["B", "H", "S", "D"],  # [batch, heads, seq, head_dim]
            cos: FLOAT["S", "D2"],  # [seq, head_dim/2]
            sin: FLOAT["S", "D2"],  # [seq, head_dim/2]
        ) -> FLOAT["B", "H", "S", "D"]:
            # Split x into first and second half
            # x1 = x[..., :head_dim/2], x2 = x[..., head_dim/2:]
            # result = concat(x1*cos - x2*sin, x1*sin + x2*cos)

            # Get shapes
            shape = op.Shape(x)
            head_dim = op.Gather(
                shape,
                op.Constant(
                    value=onnx.numpy_helper.from_array(
                        np.array(3, dtype=np.int64), name="idx3"
                    )
                ),
                axis=0,
            )
            half_dim = op.Div(
                head_dim,
                op.Constant(
                    value=onnx.numpy_helper.from_array(
                        np.array(2, dtype=np.int64), name="two"
                    )
                ),
            )

            # Slice x into two halves
            starts_0 = op.Constant(
                value=onnx.numpy_helper.from_array(
                    np.array([0], dtype=np.int64), name="starts_0"
                )
            )
            ends_half = op.Reshape(
                half_dim,
                op.Constant(
                    value=onnx.numpy_helper.from_array(
                        np.array([1], dtype=np.int64), name="shape1"
                    )
                ),
            )
            axes_3 = op.Constant(
                value=onnx.numpy_helper.from_array(
                    np.array([3], dtype=np.int64), name="axes_3"
                )
            )

            x1 = op.Slice(x, starts_0, ends_half, axes_3)
            x2 = op.Slice(
                x,
                ends_half,
                op.Reshape(
                    head_dim,
                    op.Constant(
                        value=onnx.numpy_helper.from_array(
                            np.array([1], dtype=np.int64), name="shape1_2"
                        )
                    ),
                ),
                axes_3,
            )

            # Expand cos/sin for broadcast: [S, D2] -> [1, 1, S, D2]
            cos_exp = op.Unsqueeze(op.Unsqueeze(cos, axes=[0]), axes=[0])
            sin_exp = op.Unsqueeze(op.Unsqueeze(sin, axes=[0]), axes=[0])

            # Apply rotation
            x1_cos = op.Mul(x1, cos_exp)
            x2_sin = op.Mul(x2, sin_exp)
            x1_sin = op.Mul(x1, sin_exp)
            x2_cos = op.Mul(x2, cos_exp)

            rot1 = op.Sub(x1_cos, x2_sin)
            rot2 = op.Add(x1_sin, x2_cos)

            return op.Concat(rot1, rot2, axis=-1)

        model = rope_model.to_model_proto()

        batch, heads, seq, head_dim = 2, 8, 16, 64
        x = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
        cos = np.cos(
            np.arange(seq)[:, None]
            / 10000 ** (np.arange(head_dim // 2) / (head_dim // 2))
        ).astype(np.float32)
        sin = np.sin(
            np.arange(seq)[:, None]
            / 10000 ** (np.arange(head_dim // 2) / (head_dim // 2))
        ).astype(np.float32)

        run_and_compare(model, {"x": x, "cos": cos, "sin": sin})


class TestCausalMask:
    """Test causal attention mask generation used in LLaMA."""

    def test_causal_mask_trilu(self):
        """Test causal mask using Trilu operator."""

        @script()
        def causal_mask_model(x: FLOAT["S", "S"]) -> FLOAT["S", "S"]:
            # Create upper triangular mask and negate for causal
            ones = op.ConstantOfShape(
                op.Shape(x),
                value=onnx.numpy_helper.from_array(
                    np.array([1.0], dtype=np.float32), name="one"
                ),
            )
            # Trilu with upper=0 gives lower triangular (including diagonal)
            mask = op.Trilu(ones, upper=0)
            return mask

        model = causal_mask_model.to_model_proto()

        seq = 8
        x = np.ones((seq, seq), dtype=np.float32)

        run_and_compare(model, {"x": x})


class TestAttentionPattern:
    """Test scaled dot-product attention pattern used in LLaMA."""

    def test_sdpa_basic(self):
        """Test basic scaled dot-product attention."""

        @script()
        def sdpa_model(
            q: FLOAT["B", "H", "S", "D"],
            k: FLOAT["B", "H", "S", "D"],
            v: FLOAT["B", "H", "S", "D"],
        ) -> FLOAT["B", "H", "S", "D"]:
            # Attention = softmax(Q @ K^T / sqrt(d)) @ V
            scale = op.Constant(
                value=onnx.numpy_helper.from_array(
                    np.array(0.125, dtype=np.float32),
                    name="scale",  # 1/sqrt(64)
                )
            )

            # K^T: [B, H, S, D] -> [B, H, D, S]
            k_t = op.Transpose(k, perm=[0, 1, 3, 2])

            # Q @ K^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
            scores = op.MatMul(q, k_t)

            # Scale
            scores_scaled = op.Mul(scores, scale)

            # Softmax
            attn = op.Softmax(scores_scaled, axis=-1)

            # @ V
            return op.MatMul(attn, v)

        model = sdpa_model.to_model_proto()

        batch, heads, seq, head_dim = 2, 8, 16, 64
        q = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
        k = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)

        run_and_compare(model, {"q": q, "k": k, "v": v})


class TestEmbeddingLookup:
    """Test embedding lookup used in LLaMA."""

    def test_gather_embedding(self):
        """Test token embedding lookup via Gather."""

        @script()
        def embedding_model(
            input_ids: INT64["B", "S"],
            embed_weights: FLOAT["V", "H"],
        ) -> FLOAT["B", "S", "H"]:
            return op.Gather(embed_weights, input_ids, axis=0)

        model = embedding_model.to_model_proto()

        batch, seq, vocab, hidden = 2, 16, 1000, 64
        input_ids = np.random.randint(0, vocab, (batch, seq)).astype(np.int64)
        embed_weights = np.random.randn(vocab, hidden).astype(np.float32)

        run_and_compare(model, {"input_ids": input_ids, "embed_weights": embed_weights})


class TestMLPPattern:
    """Test MLP (Feed-Forward) pattern used in LLaMA."""

    def test_gated_mlp(self):
        """Test gated MLP: down(silu(gate(x)) * up(x))."""

        @script()
        def gated_mlp_model(
            x: FLOAT["B", "S", "H"],
            gate_weight: FLOAT["I", "H"],
            up_weight: FLOAT["I", "H"],
            down_weight: FLOAT["H", "I"],
        ) -> FLOAT["B", "S", "H"]:
            # gate = x @ gate_weight.T
            gate_weight_t = op.Transpose(gate_weight, perm=[1, 0])
            gate = op.MatMul(x, gate_weight_t)

            # up = x @ up_weight.T
            up_weight_t = op.Transpose(up_weight, perm=[1, 0])
            up = op.MatMul(x, up_weight_t)

            # silu(gate) * up
            gate_sigmoid = op.Sigmoid(gate)
            gate_silu = op.Mul(gate, gate_sigmoid)
            hidden = op.Mul(gate_silu, up)

            # down projection
            down_weight_t = op.Transpose(down_weight, perm=[1, 0])
            return op.MatMul(hidden, down_weight_t)

        model = gated_mlp_model.to_model_proto()

        batch, seq, hidden, intermediate = 2, 16, 64, 256
        x = np.random.randn(batch, seq, hidden).astype(np.float32)
        gate_weight = np.random.randn(intermediate, hidden).astype(np.float32) * 0.02
        up_weight = np.random.randn(intermediate, hidden).astype(np.float32) * 0.02
        down_weight = np.random.randn(hidden, intermediate).astype(np.float32) * 0.02

        run_and_compare(
            model,
            {
                "x": x,
                "gate_weight": gate_weight,
                "up_weight": up_weight,
                "down_weight": down_weight,
            },
        )


class TestTinyLlamaE2E:
    """End-to-end tests for TinyLlama model.

    These tests require downloading the model from HuggingFace and are marked slow.
    Run with: pytest -m slow
    """

    @pytest.fixture
    def tinyllama_model_path(self):
        """Download and return path to TinyLlama ONNX model."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            pytest.skip("huggingface_hub not installed")

        try:
            # Download model and data files
            model_path = hf_hub_download(
                repo_id="onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
                filename="onnx/model_fp16.onnx",
            )
            # Also need the data file
            hf_hub_download(
                repo_id="onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
                filename="onnx/model_fp16.onnx_data",
            )
            return model_path
        except Exception as e:
            pytest.skip(f"Failed to download model: {e}")

    @pytest.mark.slow
    def test_conversion_success(self, tinyllama_model_path):
        """Test that TinyLlama model converts without errors."""
        model = onnx.load(tinyllama_model_path)
        fx_module = convert_onnx_model(model)

        assert fx_module is not None
        assert isinstance(fx_module, torch.fx.GraphModule)

        # Check basic structure
        nodes = list(fx_module.graph.nodes)
        assert len(nodes) > 0

        # Check for expected node types
        node_ops = [n.op for n in nodes]
        assert "placeholder" in node_ops  # inputs
        assert "output" in node_ops  # output
        assert "call_function" in node_ops  # operations

    @pytest.mark.slow
    def test_inference_basic(self, tinyllama_model_path):
        """Test basic inference on converted TinyLlama model."""
        model = onnx.load(tinyllama_model_path)
        fx_module = convert_onnx_model(model)
        fx_module.eval()

        # TinyLlama model parameters
        batch_size = 1
        seq_len = 1  # Single token generation (decode phase)
        past_seq_len = 5  # Previous tokens in cache
        num_layers = 22
        num_kv_heads = 4
        head_dim = 64

        # Create inputs for decode phase
        inputs = {}
        inputs["input_ids"] = torch.randint(
            0, 32000, (batch_size, seq_len), dtype=torch.int64
        )
        inputs["attention_mask"] = torch.ones(
            batch_size, past_seq_len + seq_len, dtype=torch.int64
        )
        inputs["position_ids"] = torch.tensor([[past_seq_len]], dtype=torch.int64)

        # Past key values for each layer (float32 as expected by model)
        for i in range(num_layers):
            inputs[f"past_key_values.{i}.key"] = torch.randn(
                batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32
            )
            inputs[f"past_key_values.{i}.value"] = torch.randn(
                batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32
            )

        # Convert to kwargs format (replace . with _)
        fx_inputs = {}
        for name, tensor in inputs.items():
            safe_name = name.replace(".", "_").replace("/", "_").replace("-", "_")
            fx_inputs[safe_name] = tensor

        # Expected shapes: logits + 2 * num_layers present key/values
        present_seq_len = past_seq_len + seq_len
        expected_outputs = [torch.empty(batch_size, seq_len, 32000)] + [
            torch.empty(batch_size, num_kv_heads, present_seq_len, head_dim)
            for _ in range(num_layers * 2)
        ]

        run_onnx_test(
            fx_module,
            fx_inputs,
            tuple(expected_outputs),
            check_shape_only=True,
        )
