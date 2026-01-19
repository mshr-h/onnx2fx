# SPDX-License-Identifier: Apache-2.0
"""Tests for attention and transformer related operators."""

import onnxscript
import pytest
import torch
from onnx import TensorProto, helper
from onnxscript import opset13, opset14, opset15
from onnxscript import opset16, opset17, opset18, opset19, opset20
from onnxscript import opset21, opset22, opset23
from onnxscript import opset23 as op

from onnx2fx import convert
from conftest import OPSET_MODULES, opset_id, run_onnx_test


class TestLogSoftmaxOp:
    """Test LogSoftmax operator."""

    def test_log_softmax(self):
        @onnxscript.script()
        def log_softmax_model(
            x: onnxscript.FLOAT[2, 3, 4],
        ) -> onnxscript.FLOAT[2, 3, 4]:
            return op.LogSoftmax(x, axis=-1)

        x = torch.randn(2, 3, 4)
        expected = torch.nn.functional.log_softmax(x, dim=-1)
        run_onnx_test(log_softmax_model.to_model_proto, x, expected)


class TestHardmaxOp:
    """Test Hardmax operator."""

    def test_hardmax(self):
        @onnxscript.script()
        def hardmax_model(x: onnxscript.FLOAT[2, 5]) -> onnxscript.FLOAT[2, 5]:
            return op.Hardmax(x, axis=-1)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
        run_onnx_test(hardmax_model.to_model_proto, x, expected)


class TestSequenceOps:
    """Test sequence operators."""

    def test_sequence_construct_and_at(self):
        # Create model using raw ONNX nodes
        from onnx import TensorProto, helper

        # SequenceConstruct + SequenceAt
        a_input = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
        b_input = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
        pos_input = helper.make_tensor_value_info("pos", TensorProto.INT64, [])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        seq_node = helper.make_node(
            "SequenceConstruct", ["a", "b"], ["seq"], name="seq_construct"
        )
        at_node = helper.make_node(
            "SequenceAt", ["seq", "pos"], ["output"], name="seq_at"
        )

        graph = helper.make_graph(
            [seq_node, at_node], "seq_test", [a_input, b_input, pos_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        pos = torch.tensor(1, dtype=torch.int64)

        result = fx_module(a, b, pos)
        torch.testing.assert_close(result, b)

    def test_sequence_length(self):
        from onnx import TensorProto, helper

        a_input = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
        b_input = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
        c_input = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("length", TensorProto.INT64, [])

        seq_node = helper.make_node(
            "SequenceConstruct", ["a", "b", "c"], ["seq"], name="seq_construct"
        )
        len_node = helper.make_node(
            "SequenceLength", ["seq"], ["length"], name="seq_len"
        )

        graph = helper.make_graph(
            [seq_node, len_node], "seq_len_test", [a_input, b_input, c_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)

        result = fx_module(a, b, c)
        assert result.item() == 3


class TestConcatFromSequence:
    """Test ConcatFromSequence operator."""

    def test_concat_from_sequence(self):
        from onnx import TensorProto, helper

        a_input = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
        b_input = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 3])

        seq_node = helper.make_node(
            "SequenceConstruct", ["a", "b"], ["seq"], name="seq_construct"
        )
        concat_node = helper.make_node(
            "ConcatFromSequence", ["seq"], ["output"], name="concat", axis=0
        )

        graph = helper.make_graph(
            [seq_node, concat_node], "concat_seq_test", [a_input, b_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        expected = torch.cat([a, b], dim=0)
        run_onnx_test(model, (a, b), expected)

    def test_concat_from_sequence_new_axis(self):
        from onnx import TensorProto, helper

        a_input = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
        b_input = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2, 3])

        seq_node = helper.make_node(
            "SequenceConstruct", ["a", "b"], ["seq"], name="seq_construct"
        )
        concat_node = helper.make_node(
            "ConcatFromSequence", ["seq"], ["output"], name="concat", axis=0, new_axis=1
        )

        graph = helper.make_graph(
            [seq_node, concat_node], "stack_seq_test", [a_input, b_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        expected = torch.stack([a, b], dim=0)
        run_onnx_test(model, (a, b), expected)


class TestGatherScatterND:
    """Test GatherND and ScatterND operators."""

    def test_gather_nd_simple(self):
        @onnxscript.script()
        def gather_nd_model(
            data: onnxscript.FLOAT[2, 2],
            indices: onnxscript.INT64[2, 2],
        ) -> onnxscript.FLOAT[2]:
            return op.GatherND(data, indices)

        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
        expected = torch.tensor([0.0, 3.0])
        run_onnx_test(gather_nd_model.to_model_proto, (data, indices), expected)


class TestLossOps:
    """Test loss function operators."""

    def test_softmax_cross_entropy_loss(self):
        from onnx import TensorProto, helper

        scores_input = helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, [3, 5]
        )
        labels_input = helper.make_tensor_value_info("labels", TensorProto.INT64, [3])
        output = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])

        loss_node = helper.make_node(
            "SoftmaxCrossEntropyLoss",
            ["scores", "labels"],
            ["loss"],
            name="loss",
            reduction="mean",
        )

        graph = helper.make_graph(
            [loss_node], "loss_test", [scores_input, labels_input], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        scores = torch.randn(3, 5)
        labels = torch.tensor([1, 0, 4], dtype=torch.int64)
        expected = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        run_onnx_test(model, (scores, labels), expected)

    def test_nll_loss(self):
        from onnx import TensorProto, helper

        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 5])
        target_info = helper.make_tensor_value_info("target", TensorProto.INT64, [3])
        output = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])

        loss_node = helper.make_node(
            "NegativeLogLikelihoodLoss",
            ["input", "target"],
            ["loss"],
            name="nll_loss",
            reduction="mean",
        )

        graph = helper.make_graph(
            [loss_node], "nll_test", [input_info, target_info], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        # Input should be log-probabilities for NLL loss
        input_tensor = torch.nn.functional.log_softmax(torch.randn(3, 5), dim=-1)
        target = torch.tensor([1, 0, 4], dtype=torch.int64)
        expected = torch.nn.functional.nll_loss(input_tensor, target, reduction="mean")
        run_onnx_test(model, (input_tensor, target), expected)


class TestAttentionOp:
    """Test Attention operator (Microsoft custom domain)."""

    def test_attention_basic(self):
        """Test basic attention without mask."""
        from onnx import TensorProto, helper

        batch_size = 2
        seq_len = 4
        hidden_size = 6

        # Attention inputs: input, weight, bias
        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        weight_info = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, [hidden_size, 3 * hidden_size]
        )
        bias_info = helper.make_tensor_value_info(
            "bias", TensorProto.FLOAT, [3 * hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        attention_node = helper.make_node(
            "Attention",
            ["input", "weight", "bias"],
            ["output"],
            name="attention",
            domain="com.microsoft",
            num_heads=2,
        )

        graph = helper.make_graph(
            [attention_node],
            "attention_test",
            [input_info, weight_info, bias_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        # Create test inputs
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)
        bias = torch.randn(3 * hidden_size)

        # Compute expected using scaled_dot_product_attention
        qkv = torch.matmul(input_tensor, weight) + bias
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        run_onnx_test(model, (input_tensor, weight, bias), expected)

    def test_attention_without_bias(self):
        """Test attention without bias."""
        from onnx import TensorProto, helper

        batch_size = 2
        seq_len = 4
        hidden_size = 8

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        weight_info = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, [hidden_size, 3 * hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        # Empty string for optional bias input
        attention_node = helper.make_node(
            "Attention",
            ["input", "weight", ""],
            ["output"],
            name="attention_no_bias",
            domain="com.microsoft",
            num_heads=2,
        )

        graph = helper.make_graph(
            [attention_node],
            "attention_no_bias_test",
            [input_info, weight_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)

        # Compute expected using scaled_dot_product_attention
        qkv = torch.matmul(input_tensor, weight)
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        run_onnx_test(model, (input_tensor, weight), expected)

    def test_attention_unidirectional(self):
        """Test causal (unidirectional) attention."""
        from onnx import TensorProto, helper

        batch_size = 1
        seq_len = 4
        hidden_size = 6

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        weight_info = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, [hidden_size, 3 * hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        attention_node = helper.make_node(
            "Attention",
            ["input", "weight", ""],
            ["output"],
            name="causal_attention",
            domain="com.microsoft",
            num_heads=1,
            unidirectional=1,
        )

        graph = helper.make_graph(
            [attention_node],
            "causal_attention_test",
            [input_info, weight_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)

        # Compute expected using scaled_dot_product_attention with is_causal=True
        qkv = torch.matmul(input_tensor, weight)
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        run_onnx_test(model, (input_tensor, weight), expected)


class TestSkipLayerNormalization:
    """Test SkipLayerNormalization operator."""

    def test_skip_layer_norm(self):
        """Test skip connection + layer normalization."""
        from onnx import TensorProto, helper

        batch_size = 2
        seq_len = 4
        hidden_size = 8

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        skip_info = helper.make_tensor_value_info(
            "skip", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        gamma_info = helper.make_tensor_value_info(
            "gamma", TensorProto.FLOAT, [hidden_size]
        )
        beta_info = helper.make_tensor_value_info(
            "beta", TensorProto.FLOAT, [hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        skip_ln_node = helper.make_node(
            "SkipLayerNormalization",
            ["input", "skip", "gamma", "beta"],
            ["output"],
            name="skip_layer_norm",
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [skip_ln_node],
            "skip_ln_test",
            [input_info, skip_info, gamma_info, beta_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        skip_tensor = torch.randn(batch_size, seq_len, hidden_size)
        gamma = torch.randn(hidden_size)
        beta = torch.randn(hidden_size)

        # Manual computation
        hidden = input_tensor + skip_tensor
        expected = torch.nn.functional.layer_norm(
            hidden, (hidden_size,), weight=gamma, bias=beta, eps=1e-5
        )
        run_onnx_test(model, (input_tensor, skip_tensor, gamma, beta), expected)


class TestSimplifiedLayerNormalization:
    """Test SimplifiedLayerNormalization (RMSNorm) operator."""

    def test_simplified_layer_norm(self):
        """Test SimplifiedLayerNormalization computes RMSNorm correctly."""
        from onnx import TensorProto, helper

        batch_size, seq_len, hidden_size = 2, 4, 8

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        scale_info = helper.make_tensor_value_info(
            "scale", TensorProto.FLOAT, [hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        node = helper.make_node(
            "SimplifiedLayerNormalization",
            ["input", "scale"],
            ["output"],
            name="simplified_ln",
            axis=-1,
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [node],
            "simplified_ln_test",
            [input_info, scale_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        x = torch.randn(batch_size, seq_len, hidden_size)
        scale = torch.randn(hidden_size)

        # Manual RMSNorm computation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + 1e-5)
        expected = x_normalized * scale
        run_onnx_test(model, (x, scale), expected)

    def test_simplified_layer_norm_fp16(self):
        """Test SimplifiedLayerNormalization with FP16."""
        from onnx import TensorProto, helper

        batch_size, seq_len, hidden_size = 2, 4, 16

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT16, [batch_size, seq_len, hidden_size]
        )
        scale_info = helper.make_tensor_value_info(
            "scale", TensorProto.FLOAT16, [hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, [batch_size, seq_len, hidden_size]
        )

        node = helper.make_node(
            "SimplifiedLayerNormalization",
            ["input", "scale"],
            ["output"],
            name="simplified_ln",
            axis=-1,
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [node],
            "simplified_ln_test",
            [input_info, scale_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        scale = torch.randn(hidden_size, dtype=torch.float16)

        # Manual RMSNorm computation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + 1e-5)
        expected = x_normalized * scale
        run_onnx_test(model, (x, scale), expected, rtol=1e-2, atol=1e-2)


class TestGroupQueryAttention:
    """Test GroupQueryAttention operator (used in LLaMA, Mistral, etc.)."""

    def test_group_query_attention_basic(self):
        """Test basic GQA without past key-values."""
        from onnx import TensorProto, helper

        batch_size = 1
        seq_len = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        hidden_size = num_heads * head_dim
        kv_hidden_size = num_kv_heads * head_dim

        query_info = helper.make_tensor_value_info(
            "query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        key_info = helper.make_tensor_value_info(
            "key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]
        )
        value_info = helper.make_tensor_value_info(
            "value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        node = helper.make_node(
            "GroupQueryAttention",
            ["query", "key", "value"],
            ["output", "present_key", "present_value"],
            name="gqa",
            num_heads=num_heads,
            kv_num_heads=num_kv_heads,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "gqa_test",
            [query_info, key_info, value_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, kv_hidden_size)
        value = torch.randn(batch_size, seq_len, kv_hidden_size)

        result = fx_module(query, key, value)

        # Check output shape
        assert result.shape == (batch_size, seq_len, hidden_size)


class TestRotaryEmbedding:
    """Test RotaryEmbedding operator (com.microsoft domain)."""

    def test_rotary_embedding_3d_non_interleaved(self):
        """Test RotaryEmbedding with 3D input and non-interleaved format."""
        batch_size = 2
        seq_len = 4
        num_heads = 4
        head_size = 8
        hidden_size = num_heads * head_size
        max_seq_len = 16
        rotary_dim = head_size  # Full rotation

        # Create input tensor value infos
        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [1]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            num_heads=num_heads,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        # Create test inputs
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.tensor([0], dtype=torch.int64)

        # Generate cos/sin cache
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        # Check output shape matches input
        assert result.shape == input_tensor.shape

    def test_rotary_embedding_4d_non_interleaved(self):
        """Test RotaryEmbedding with 4D input (batch, heads, seq, head_size)."""
        batch_size = 2
        num_heads = 4
        seq_len = 4
        head_size = 8
        max_seq_len = 16
        rotary_dim = head_size

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [seq_len]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_size)
        position_ids = torch.arange(seq_len, dtype=torch.int64)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        assert result.shape == input_tensor.shape

    def test_rotary_embedding_interleaved(self):
        """Test RotaryEmbedding with interleaved format (GPT-NeoX style)."""
        batch_size = 2
        num_heads = 4
        seq_len = 4
        head_size = 8
        max_seq_len = 16
        rotary_dim = head_size

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [seq_len]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=1,  # Interleaved format
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_size)
        position_ids = torch.arange(seq_len, dtype=torch.int64)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        assert result.shape == input_tensor.shape

    def test_rotary_embedding_with_scale(self):
        """Test RotaryEmbedding with custom scale."""
        batch_size = 2
        num_heads = 4
        seq_len = 4
        head_size = 8
        max_seq_len = 16
        rotary_dim = head_size
        scale = 0.5

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [seq_len]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            scale=scale,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_size)
        position_ids = torch.arange(seq_len, dtype=torch.int64)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        assert result.shape == input_tensor.shape

    def test_rotary_embedding_partial_rotation(self):
        """Test RotaryEmbedding with partial rotation (rotary_dim < head_size)."""
        batch_size = 2
        num_heads = 4
        seq_len = 4
        head_size = 16
        max_seq_len = 16
        rotary_dim = 8  # Only rotate first 8 dimensions

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [seq_len]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            rotary_embedding_dim=rotary_dim,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_size)
        position_ids = torch.arange(seq_len, dtype=torch.int64)

        # cos/sin cache matches rotary_dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        # Check output shape
        assert result.shape == input_tensor.shape

        # The pass-through part should be unchanged
        torch.testing.assert_close(
            result[..., rotary_dim:],
            input_tensor[..., rotary_dim:],
        )

    def test_rotary_embedding_2d_position_ids(self):
        """Test RotaryEmbedding with 2D position_ids (batch, seq)."""
        batch_size = 2
        num_heads = 4
        seq_len = 4
        head_size = 8
        max_seq_len = 16
        rotary_dim = head_size

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [batch_size, seq_len]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_dim // 2]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )

        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_size)
        # 2D position_ids with same positions for each batch
        position_ids = (
            torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
        )

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim // 2).float() / (rotary_dim // 2))
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        assert result.shape == input_tensor.shape

    def test_rotary_embedding_num_heads_zero_3d(self):
        """Test RotaryEmbedding with num_heads=0 and 3D input (LFM2.5 style).

        When num_heads=0, the implementation should infer head_size from cos_cache
        dimension and calculate num_heads as hidden_size / head_size.
        """
        batch_size = 2
        seq_len = 4
        num_heads = 32  # Will be inferred, not passed to the operator
        head_size = 64
        hidden_size = num_heads * head_size  # 2048
        max_seq_len = 16
        rotary_half_dim = 32  # cos_cache.shape[-1]

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        position_ids_info = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [1]
        )
        cos_cache_info = helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, [max_seq_len, rotary_half_dim]
        )
        sin_cache_info = helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, [max_seq_len, rotary_half_dim]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        # num_heads=0 means infer from cos_cache
        node = helper.make_node(
            "RotaryEmbedding",
            ["input", "position_ids", "cos_cache", "sin_cache"],
            ["output"],
            name="rotary_emb",
            interleaved=0,
            num_heads=0,  # Key: num_heads=0
            rotary_embedding_dim=0,  # Key: rotary_embedding_dim=0
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "rotary_embedding_test",
            [input_info, position_ids_info, cos_cache_info, sin_cache_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        fx_module = convert(model)

        # Create test inputs
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.tensor([0], dtype=torch.int64)

        # Generate cos/sin cache with rotary_half_dim=32 (so rotary_dim=64)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_half_dim).float() / rotary_half_dim)
        )
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        result = fx_module(input_tensor, position_ids, cos_cache, sin_cache)

        # Check output shape matches input
        assert result.shape == input_tensor.shape

        # Verify that rotation was applied correctly by checking it's different from input
        # (rotation should change the values)
        assert not torch.allclose(result, input_tensor)


class TestAttentionOpsMultiOpset:
    """Test attention-related operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_hardmax_all_opsets(self, opset):
        """Hardmax should work across all opsets (11+)."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])

        hardmax_node = helper.make_node("Hardmax", ["X"], ["Y"], axis=-1)

        graph = helper.make_graph([hardmax_node], "test", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
        run_onnx_test(model, x, expected)

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
    def test_log_softmax_all_opsets(self, opset):
        """LogSoftmax should work across opsets 13+ (axis semantics changed)."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])

        node = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=-1)

        graph = helper.make_graph([node], "test", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        x = torch.randn(2, 3, 4)
        expected = torch.nn.functional.log_softmax(x, dim=-1)
        run_onnx_test(model, x, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
    def test_gather_nd_all_opsets(self, opset):
        """GatherND should work across all opsets (11+)."""
        data_info = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2])
        indices_info = helper.make_tensor_value_info(
            "indices", TensorProto.INT64, [2, 2]
        )
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        gather_node = helper.make_node("GatherND", ["data", "indices"], ["output"])

        graph = helper.make_graph(
            [gather_node], "test", [data_info, indices_info], [output_info]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
        expected = torch.tensor([0.0, 3.0])
        run_onnx_test(model, (data, indices), expected)


# =============================================================================
# Microsoft Domain (com.microsoft) Tests
# =============================================================================


class TestSkipLayerNormalizationMicrosoft:
    """Test SkipLayerNormalization with com.microsoft domain."""

    def test_skip_layer_norm_microsoft_domain(self):
        """Test skip connection + layer normalization with com.microsoft domain."""
        from onnx import TensorProto, helper

        batch_size = 2
        seq_len = 4
        hidden_size = 8

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        skip_info = helper.make_tensor_value_info(
            "skip", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        gamma_info = helper.make_tensor_value_info(
            "gamma", TensorProto.FLOAT, [hidden_size]
        )
        beta_info = helper.make_tensor_value_info(
            "beta", TensorProto.FLOAT, [hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        skip_ln_node = helper.make_node(
            "SkipLayerNormalization",
            ["input", "skip", "gamma", "beta"],
            ["output"],
            name="skip_layer_norm",
            domain="com.microsoft",
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [skip_ln_node],
            "skip_ln_msft_test",
            [input_info, skip_info, gamma_info, beta_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        skip_tensor = torch.randn(batch_size, seq_len, hidden_size)
        gamma = torch.randn(hidden_size)
        beta = torch.randn(hidden_size)

        # Manual computation
        hidden = input_tensor + skip_tensor
        expected = torch.nn.functional.layer_norm(
            hidden, (hidden_size,), weight=gamma, bias=beta, eps=1e-5
        )
        run_onnx_test(model, (input_tensor, skip_tensor, gamma, beta), expected)


class TestSimplifiedLayerNormalizationMicrosoft:
    """Test SimplifiedLayerNormalization with com.microsoft domain."""

    def test_simplified_layer_norm_microsoft_domain(self):
        """Test SimplifiedLayerNormalization (RMSNorm) with com.microsoft domain."""
        from onnx import TensorProto, helper

        batch_size, seq_len, hidden_size = 2, 4, 8

        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
        scale_info = helper.make_tensor_value_info(
            "scale", TensorProto.FLOAT, [hidden_size]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )

        node = helper.make_node(
            "SimplifiedLayerNormalization",
            ["input", "scale"],
            ["output"],
            name="simplified_ln",
            domain="com.microsoft",
            axis=-1,
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [node],
            "simplified_ln_msft_test",
            [input_info, scale_info],
            [output_info],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        x = torch.randn(batch_size, seq_len, hidden_size)
        scale = torch.randn(hidden_size)

        # Manual RMSNorm computation
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
        expected = (x / rms) * scale
        run_onnx_test(model, (x, scale), expected)
