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
from conftest import OPSET_MODULES


class TestLogSoftmaxOp:
    """Test LogSoftmax operator."""

    def test_log_softmax(self):
        @onnxscript.script()
        def log_softmax_model(
            x: onnxscript.FLOAT[2, 3, 4],
        ) -> onnxscript.FLOAT[2, 3, 4]:
            return op.LogSoftmax(x, axis=-1)

        model = log_softmax_model.to_model_proto()
        fx_module = convert(model)

        x = torch.randn(2, 3, 4)
        expected = torch.nn.functional.log_softmax(x, dim=-1)

        result = fx_module(x)
        torch.testing.assert_close(result, expected)


class TestHardmaxOp:
    """Test Hardmax operator."""

    def test_hardmax(self):
        @onnxscript.script()
        def hardmax_model(x: onnxscript.FLOAT[2, 5]) -> onnxscript.FLOAT[2, 5]:
            return op.Hardmax(x, axis=-1)

        model = hardmax_model.to_model_proto()
        fx_module = convert(model)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

        result = fx_module(x)
        torch.testing.assert_close(result, expected)


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

        fx_module = convert(model)

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)

        result = fx_module(a, b)
        expected = torch.cat([a, b], dim=0)
        torch.testing.assert_close(result, expected)

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

        fx_module = convert(model)

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)

        result = fx_module(a, b)
        expected = torch.stack([a, b], dim=0)
        torch.testing.assert_close(result, expected)


class TestGatherScatterND:
    """Test GatherND and ScatterND operators."""

    def test_gather_nd_simple(self):
        @onnxscript.script()
        def gather_nd_model(
            data: onnxscript.FLOAT[2, 2],
            indices: onnxscript.INT64[2, 2],
        ) -> onnxscript.FLOAT[2]:
            return op.GatherND(data, indices)

        model = gather_nd_model.to_model_proto()
        fx_module = convert(model)

        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)

        result = fx_module(data, indices)
        expected = torch.tensor([0.0, 3.0])
        torch.testing.assert_close(result, expected)


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

        fx_module = convert(model)

        scores = torch.randn(3, 5)
        labels = torch.tensor([1, 0, 4], dtype=torch.int64)

        result = fx_module(scores, labels)
        expected = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        torch.testing.assert_close(result, expected)

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

        fx_module = convert(model)

        # Input should be log-probabilities for NLL loss
        input_tensor = torch.nn.functional.log_softmax(torch.randn(3, 5), dim=-1)
        target = torch.tensor([1, 0, 4], dtype=torch.int64)

        result = fx_module(input_tensor, target)
        expected = torch.nn.functional.nll_loss(input_tensor, target, reduction="mean")
        torch.testing.assert_close(result, expected)


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
            num_heads=2,
        )

        graph = helper.make_graph(
            [attention_node],
            "attention_test",
            [input_info, weight_info, bias_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        # Create test inputs
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)
        bias = torch.randn(3 * hidden_size)

        # Run the converted model
        result = fx_module(input_tensor, weight, bias)

        # Compute expected using scaled_dot_product_attention
        qkv = torch.matmul(input_tensor, weight) + bias
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.testing.assert_close(result, expected)

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
            num_heads=2,
        )

        graph = helper.make_graph(
            [attention_node],
            "attention_no_bias_test",
            [input_info, weight_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)

        result = fx_module(input_tensor, weight)

        # Compute expected using scaled_dot_product_attention
        qkv = torch.matmul(input_tensor, weight)
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.testing.assert_close(result, expected)

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
            num_heads=1,
            unidirectional=1,
        )

        graph = helper.make_graph(
            [attention_node],
            "causal_attention_test",
            [input_info, weight_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size, 3 * hidden_size)

        result = fx_module(input_tensor, weight)

        # Compute expected using scaled_dot_product_attention with is_causal=True
        qkv = torch.matmul(input_tensor, weight)
        q, k, v = qkv.chunk(3, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        torch.testing.assert_close(result, expected)


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

        fx_module = convert(model)

        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        skip_tensor = torch.randn(batch_size, seq_len, hidden_size)
        gamma = torch.randn(hidden_size)
        beta = torch.randn(hidden_size)

        result = fx_module(input_tensor, skip_tensor, gamma, beta)

        # Manual computation
        hidden = input_tensor + skip_tensor
        expected = torch.nn.functional.layer_norm(
            hidden, (hidden_size,), weight=gamma, bias=beta, eps=1e-5
        )

        torch.testing.assert_close(result, expected)


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

        fx_module = convert(model)

        x = torch.randn(batch_size, seq_len, hidden_size)
        scale = torch.randn(hidden_size)

        result = fx_module(x, scale)

        # Manual RMSNorm computation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + 1e-5)
        expected = x_normalized * scale

        torch.testing.assert_close(result, expected)

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

        fx_module = convert(model)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        scale = torch.randn(hidden_size, dtype=torch.float16)

        result = fx_module(x, scale)

        # Manual RMSNorm computation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + 1e-5)
        expected = x_normalized * scale

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


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


class TestAttentionOpsMultiOpset:
    """Test attention-related operators across multiple opset versions."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_hardmax_all_opsets(self, opset):
        """Hardmax should work across all opsets (11+)."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])

        hardmax_node = helper.make_node("Hardmax", ["X"], ["Y"], axis=-1)

        graph = helper.make_graph([hardmax_node], "test", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_module = convert(model)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

        result = fx_module(x)
        torch.testing.assert_close(result, expected)

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
        ids=lambda x: f"opset{x.version}",
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

        fx_module = convert(model)

        x = torch.randn(2, 3, 4)
        expected = torch.nn.functional.log_softmax(x, dim=-1)

        result = fx_module(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
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

        fx_module = convert(model)

        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)

        result = fx_module(data, indices)
        expected = torch.tensor([0.0, 3.0])
        torch.testing.assert_close(result, expected)
