# SPDX-License-Identifier: Apache-2.0
"""Attention and Transformer related operators."""

from typing import TYPE_CHECKING

import onnx
import torch

from ..op_registry import register
from ..utils.attributes import get_attribute

if TYPE_CHECKING:
    from ..graph_builder import GraphBuilder


# =============================================================================
# Softmax variants
# =============================================================================


@register("LogSoftmax")
def log_softmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Log of softmax."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)
    return builder.call_function(
        torch.nn.functional.log_softmax, args=(x,), kwargs={"dim": axis}
    )


@register("Hardmax")
def hardmax(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Hardmax - one-hot encoding of argmax."""
    x = builder.get_value(node.input[0])
    axis = get_attribute(node, "axis", -1)

    def _hardmax(t: torch.Tensor, ax: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.argmax(t, dim=ax), num_classes=t.shape[ax]
        ).to(t.dtype)

    return builder.call_function(_hardmax, args=(x, axis))


# =============================================================================
# Gather/Scatter ND operators
# =============================================================================


@register("GatherND")
def gather_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Gather elements from data using indices."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])

    batch_dims = get_attribute(node, "batch_dims", 0)

    def _gather_nd(d: torch.Tensor, idx: torch.Tensor, bd: int) -> torch.Tensor:
        idx = idx.long()
        idx_shape = idx.shape[:-1]
        last_dim = idx.shape[-1]
        data_shape = d.shape[last_dim:]

        result_shape = idx_shape + data_shape
        result = torch.zeros(result_shape, dtype=d.dtype, device=d.device)

        # Flatten for iteration
        flat_idx = idx.reshape(-1, last_dim)
        flat_result = (
            result.reshape(-1, *data_shape) if data_shape else result.reshape(-1)
        )

        for i in range(flat_idx.shape[0]):
            data_idx = tuple(flat_idx[i].tolist())
            flat_result[i] = d[data_idx]

        return result

    return builder.call_function(_gather_nd, args=(data, indices, batch_dims))


@register("ScatterND")
def scatter_nd(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Scatter updates into data at indices."""
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    updates = builder.get_value(node.input[2])

    _reduction = get_attribute(node, "reduction", "none")

    def _scatter_nd(
        d: torch.Tensor, idx: torch.Tensor, upd: torch.Tensor
    ) -> torch.Tensor:
        output = d.clone()
        idx = idx.long()

        idx_shape = idx.shape[:-1]
        last_dim = idx.shape[-1]

        flat_idx = idx.reshape(-1, last_dim)
        flat_upd = upd.reshape(-1, *upd.shape[len(idx_shape) :])

        for i in range(flat_idx.shape[0]):
            data_idx = tuple(flat_idx[i].tolist())
            output[data_idx] = flat_upd[i]

        return output

    return builder.call_function(_scatter_nd, args=(data, indices, updates))


# =============================================================================
# Embedding and LayerNorm variants
# =============================================================================


@register("SkipLayerNormalization")
def skip_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Skip connection + LayerNorm (common in transformers)."""
    x = builder.get_value(node.input[0])
    skip = builder.get_value(node.input[1])
    gamma = builder.get_value(node.input[2])
    beta = (
        builder.get_value(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else None
    )
    bias = (
        builder.get_value(node.input[4])
        if len(node.input) > 4 and node.input[4]
        else None
    )

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _skip_layer_norm(
        inp: torch.Tensor,
        sk: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor | None,
        bi: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        hidden = inp + sk
        if bi is not None:
            hidden = hidden + bi
        return torch.nn.functional.layer_norm(
            hidden, hidden.shape[-1:], weight=g, bias=b, eps=eps
        )

    return builder.call_function(
        _skip_layer_norm, args=(x, skip, gamma, beta, bias, epsilon)
    )


@register("EmbedLayerNormalization")
def embed_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Embedding + LayerNorm (common in BERT-like models)."""
    input_ids = builder.get_value(node.input[0])
    segment_ids = (
        builder.get_value(node.input[1])
        if len(node.input) > 1 and node.input[1]
        else None
    )
    word_embedding = builder.get_value(node.input[2])
    position_embedding = builder.get_value(node.input[3])
    segment_embedding = (
        builder.get_value(node.input[4])
        if len(node.input) > 4 and node.input[4]
        else None
    )
    gamma = builder.get_value(node.input[5]) if len(node.input) > 5 else None
    beta = builder.get_value(node.input[6]) if len(node.input) > 6 else None

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _embed_layer_norm(
        ids: torch.Tensor,
        seg_ids: torch.Tensor | None,
        word_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        seg_emb: torch.Tensor | None,
        g: torch.Tensor | None,
        b: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        # Word embedding lookup
        word_embed = torch.nn.functional.embedding(ids, word_emb)

        # Position embedding (assume sequential positions)
        seq_len = ids.shape[1]
        pos_embed = pos_emb[:seq_len].unsqueeze(0).expand(ids.shape[0], -1, -1)

        hidden = word_embed + pos_embed

        # Add segment embedding if present
        if seg_emb is not None and seg_ids is not None:
            seg_embed = torch.nn.functional.embedding(seg_ids, seg_emb)
            hidden = hidden + seg_embed

        # Layer normalization
        if g is not None:
            hidden = torch.nn.functional.layer_norm(
                hidden, hidden.shape[-1:], weight=g, bias=b, eps=eps
            )

        return hidden

    return builder.call_function(
        _embed_layer_norm,
        args=(
            input_ids,
            segment_ids,
            word_embedding,
            position_embedding,
            segment_embedding,
            gamma,
            beta,
            epsilon,
        ),
    )


# =============================================================================
# Attention operator (custom Microsoft domain)
# =============================================================================


@register("Attention")
def attention(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Attention operator.

    Supports two patterns:
    1. Decomposed SDPA (ViT, etc.): Inputs are Q, K, V tensors directly
       - 3 inputs: query, key, value
       - Attributes: is_causal, softcap, qk_matmul_output_mode

    2. Fused SDPA (com.microsoft domain): Inputs are input, weight, bias
       - 5+ inputs: input, weight, bias (optional), mask_index (optional), past (optional)
       - Attributes: num_heads, unidirectional
    """
    num_inputs = len(node.input)

    # Check if this is Decomposed SDPA pattern (Q, K, V directly)
    # Heuristic: If we have exactly 3 inputs and is_causal attribute exists
    is_causal_attr = get_attribute(node, "is_causal", None)
    softcap = get_attribute(node, "softcap", 0.0)

    if num_inputs == 3 and is_causal_attr is not None:
        # Decomposed SDPA: Q, K, V are passed directly
        query = builder.get_value(node.input[0])
        key = builder.get_value(node.input[1])
        value = builder.get_value(node.input[2])

        is_causal = is_causal_attr

        def _decomposed_sdpa(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            is_causal: int,
            scale: float,
        ) -> torch.Tensor:
            # Use scaled_dot_product_attention directly with Q, K, V
            if is_causal:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return output

        return builder.call_function(
            _decomposed_sdpa,
            args=(query, key, value, is_causal, softcap),
        )

    # Fused SDPA pattern (Microsoft domain style)
    input_node = builder.get_value(node.input[0])
    weight = builder.get_value(node.input[1])
    bias = (
        builder.get_value(node.input[2])
        if len(node.input) > 2 and node.input[2]
        else None
    )
    mask_index = (
        builder.get_value(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else None
    )

    num_heads = get_attribute(node, "num_heads", 1)
    unidirectional = get_attribute(node, "unidirectional", 0)

    def _attention_sdpa(
        inp: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        mask: torch.Tensor | None,
        n_heads: int,
        is_causal: int,
    ) -> torch.Tensor:
        # QKV projection: input @ weight + bias
        qkv = torch.matmul(inp, w)
        if b is not None:
            qkv = qkv + b

        # Split QKV into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Use scaled_dot_product_attention
        # Note: is_causal cannot be used together with attn_mask
        if mask is not None:
            # Convert mask to attention mask format (additive mask)
            # SDPA expects: 0 = attend, -inf = don't attend
            attn_mask = torch.where(mask == 0, float("-inf"), 0.0)
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        elif is_causal:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        return output

    return builder.call_function(
        _attention_sdpa,
        args=(input_node, weight, bias, mask_index, num_heads, unidirectional),
    )


# =============================================================================
# Simplified LayerNormalization variants (ONNX Runtime contrib ops)
# =============================================================================


@register("SimplifiedLayerNormalization")
@register("SimplifiedLayerNormalization", domain="com.microsoft")
def simplified_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Simplified Layer Normalization (RMSNorm).

    This is LayerNormalization without bias and mean subtraction.
    Formula: output = x / sqrt(mean(x^2) + epsilon) * scale
    """
    x = builder.get_value(node.input[0])
    scale = builder.get_value(node.input[1])

    axis = get_attribute(node, "axis", -1)
    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _simplified_layer_norm(x, scale, axis, epsilon):
        # Simplified LayerNorm (RMSNorm)
        # output = x * rsqrt(mean(x^2) + epsilon) * scale
        if axis < 0:
            axis_pos = x.dim() + axis
        else:
            axis_pos = axis

        # Keep dims for broadcasting
        dims = list(range(axis_pos, x.dim()))

        # Compute RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + epsilon)

        return x_normalized * scale

    return builder.call_function(_simplified_layer_norm, args=(x, scale, axis, epsilon))


@register("SkipSimplifiedLayerNormalization")
@register("SkipSimplifiedLayerNormalization", domain="com.microsoft")
def skip_simplified_layer_normalization(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Skip connection + Simplified Layer Normalization (RMSNorm)."""
    x = builder.get_value(node.input[0])
    skip = builder.get_value(node.input[1])
    scale = builder.get_value(node.input[2])
    bias = (
        builder.get_value(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else None
    )

    epsilon = get_attribute(node, "epsilon", 1e-5)

    def _skip_simplified_layer_norm(x, skip, scale, bias, epsilon):
        # Add skip connection
        hidden = x + skip
        if bias is not None:
            hidden = hidden + bias

        # Simplified LayerNorm (RMSNorm)
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        hidden_normalized = hidden * torch.rsqrt(variance + epsilon)

        return hidden_normalized * scale

    return builder.call_function(
        _skip_simplified_layer_norm, args=(x, skip, scale, bias, epsilon)
    )


@register("GroupQueryAttention")
@register("GroupQueryAttention", domain="com.microsoft")
def group_query_attention(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """Group Query Attention (GQA) - used in LLaMA, Mistral, etc.

    Inputs:
        - query: [batch, seq_len, num_heads * head_size]
        - key: [batch, kv_seq_len, num_kv_heads * head_size]
        - value: [batch, kv_seq_len, num_kv_heads * head_size]
        - past_key (optional): [batch, num_kv_heads, past_seq_len, head_size]
        - past_value (optional): [batch, num_kv_heads, past_seq_len, head_size]
        - seqlens_k (optional): cumulative sequence lengths for keys
        - total_sequence_length (optional): total sequence length
        - cos_cache (optional): [max_seq_len, head_size / 2] or [max_seq_len, head_size]
        - sin_cache (optional): [max_seq_len, head_size / 2] or [max_seq_len, head_size]

    Attributes:
        - num_heads: number of attention heads
        - kv_num_heads: number of key-value heads (for GQA)
        - scale: scaling factor (default: 1/sqrt(head_size))
        - local_window_size: for sliding window attention
        - do_rotary: whether to apply rotary position embeddings
        - rotary_interleaved: whether rotary is interleaved (GPT-NeoX style vs LLaMA)

    Outputs:
        - output: [batch, seq_len, num_heads * head_size]
        - present_key: [batch, num_kv_heads, total_seq_len, head_size]
        - present_value: [batch, num_kv_heads, total_seq_len, head_size]
    """
    # Get required inputs
    query = builder.get_value(node.input[0])
    key = builder.get_value(node.input[1])
    value = builder.get_value(node.input[2])

    def get_optional_input(idx: int) -> torch.fx.Node | None:
        return (
            builder.get_value(node.input[idx])
            if len(node.input) > idx and node.input[idx]
            else None
        )

    # Get optional inputs
    past_key = get_optional_input(3)
    past_value = get_optional_input(4)
    seqlens_k = get_optional_input(5)
    total_seq_len = get_optional_input(6)
    cos_cache = get_optional_input(7)
    sin_cache = get_optional_input(8)

    # Get attributes
    num_heads = get_attribute(node, "num_heads", 1)
    kv_num_heads = get_attribute(node, "kv_num_heads", num_heads)
    scale = get_attribute(node, "scale", None)
    local_window_size = get_attribute(node, "local_window_size", -1)
    do_rotary = get_attribute(node, "do_rotary", 0)
    rotary_interleaved = get_attribute(node, "rotary_interleaved", 0)

    def _group_query_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        seqlens_k: torch.Tensor | None,
        total_seq_len: torch.Tensor | None,
        cos_cache: torch.Tensor | None,
        sin_cache: torch.Tensor | None,
        n_heads: int,
        n_kv_heads: int,
        attn_scale: float | None,
        window_size: int,
        do_rotary: int,
        rotary_interleaved: int,
    ):
        batch_size, seq_len, _ = q.shape
        head_size = q.shape[-1] // n_heads
        kv_head_size = k.shape[-1] // n_kv_heads

        # Reshape Q, K, V to [batch, num_heads, seq_len, head_size]
        q = q.view(batch_size, seq_len, n_heads, head_size).transpose(1, 2)
        k = k.view(batch_size, -1, n_kv_heads, kv_head_size).transpose(1, 2)
        v = v.view(batch_size, -1, n_kv_heads, kv_head_size).transpose(1, 2)

        # Calculate position offset from past cache
        past_seq_len = 0
        if past_k is not None and past_k.numel() > 0:
            past_seq_len = past_k.shape[2]

        # Apply rotary position embeddings if enabled
        if do_rotary and cos_cache is not None and sin_cache is not None:
            # Get the position indices
            positions = torch.arange(
                past_seq_len, past_seq_len + seq_len, device=q.device
            )

            # Get cos/sin values for current positions
            cos = cos_cache[positions]  # [seq_len, rotary_dim]
            sin = sin_cache[positions]  # [seq_len, rotary_dim]

            # Expand for batch and heads: [1, 1, seq_len, rotary_dim]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            rotary_dim = cos.shape[-1]

            if rotary_interleaved:
                # GPT-NeoX style: [x0, x1, x2, x3, ...] -> rotate pairs
                q_rot = q[..., :rotary_dim]
                q_pass = q[..., rotary_dim:]
                k_rot = k[..., :rotary_dim]
                k_pass = k[..., rotary_dim:]

                # Apply rotation
                q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
                k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]

                cos_half = cos[..., ::2]
                sin_half = sin[..., ::2]

                q_rot_new = torch.stack(
                    [q1 * cos_half - q2 * sin_half, q1 * sin_half + q2 * cos_half],
                    dim=-1,
                ).flatten(-2)
                k_rot_new = torch.stack(
                    [k1 * cos_half - k2 * sin_half, k1 * sin_half + k2 * cos_half],
                    dim=-1,
                ).flatten(-2)

                q = torch.cat([q_rot_new, q_pass], dim=-1)
                k = torch.cat([k_rot_new, k_pass], dim=-1)
            else:
                # LLaMA style: cos/sin are [seq, rotary_dim]
                # rotary_dim is half the head_size in this format
                # q/k first rotary_dim*2 elements are rotated:
                # q1 = q[..., :rotary_dim], q2 = q[..., rotary_dim:rotary_dim*2]
                # result = (q1*cos - q2*sin, q1*sin + q2*cos)

                rotary_full = rotary_dim * 2  # total dims that get rotated
                q_rot = q[..., :rotary_full]
                q_pass = q[..., rotary_full:]
                k_rot = k[..., :rotary_full]
                k_pass = k[..., rotary_full:]

                # Split into first half and second half
                q1, q2 = q_rot[..., :rotary_dim], q_rot[..., rotary_dim:rotary_full]
                k1, k2 = k_rot[..., :rotary_dim], k_rot[..., rotary_dim:rotary_full]

                # cos/sin are already in the right shape [1, 1, seq_len, rotary_dim]
                q_rot_new = torch.cat(
                    [q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1
                )
                k_rot_new = torch.cat(
                    [k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1
                )

                q = torch.cat([q_rot_new, q_pass], dim=-1)
                k = torch.cat([k_rot_new, k_pass], dim=-1)

        # Handle past key-value cache
        if past_k is not None and past_k.numel() > 0:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Present key-value for caching
        present_k = k
        present_v = v

        # Expand K, V for GQA (repeat for each head group)
        if n_kv_heads < n_heads:
            n_rep = n_heads // n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Compute attention scale
        if attn_scale is None:
            attn_scale = 1.0 / (head_size**0.5)

        # Use scaled_dot_product_attention
        # For autoregressive with past cache, don't use causal mask for new tokens
        # since past_k/v already handled the causality
        is_causal = seq_len > 1 and past_seq_len == 0
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=attn_scale, is_causal=is_causal
        )

        # Reshape output: [batch, num_heads, seq_len, head_size] -> [batch, seq_len, hidden]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output, present_k, present_v

    # Build the call
    result = builder.call_function(
        _group_query_attention,
        args=(
            query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k,
            total_seq_len,
            cos_cache,
            sin_cache,
            num_heads,
            kv_num_heads,
            scale,
            local_window_size,
            do_rotary,
            rotary_interleaved,
        ),
    )

    # Return tuple output
    return result


# =============================================================================
# Rotary Embedding (com.microsoft domain)
# =============================================================================


@register("RotaryEmbedding", domain="com.microsoft")
def rotary_embedding(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """Rotary Position Embedding (RoPE) operator.

    Applies rotary position embeddings to the input tensor. The positions are
    represented as rotation matrices that are multiplied to query and key
    before the inner product of query and key is taken.

    Inputs:
        - input: 3D tensor with shape (batch_size, sequence_length, hidden_size)
                 or 4D with shape (batch_size, num_heads, sequence_length, head_size)
        - position_ids: 1D tensor with shape (1) or 2D tensor with shape
                        (batch_size, sequence_length)
        - cos_cache: 2D tensor with shape (max_sequence_length, head_size / 2)
                     or (max_sequence_length, rotary_embedding_dim / 2)
        - sin_cache: 2D tensor with shape (max_sequence_length, head_size / 2)
                     or (max_sequence_length, rotary_embedding_dim / 2)

    Attributes:
        - interleaved: Indicates whether the input has real and imaginary parts
                       interleaved. Default is 0 (False).
        - num_heads: Number of attention heads. Default is 0.
        - rotary_embedding_dim: Rotary embedding dimension. Default is 0.
        - scale: Custom scale. Default is 1.0.

    Outputs:
        - output: tensor with same shape as input.
    """
    # Get inputs
    input_tensor = builder.get_value(node.input[0])
    position_ids = builder.get_value(node.input[1])
    cos_cache = builder.get_value(node.input[2])
    sin_cache = builder.get_value(node.input[3])

    # Get attributes
    interleaved = get_attribute(node, "interleaved", 0)
    num_heads = get_attribute(node, "num_heads", 0)
    rotary_embedding_dim = get_attribute(node, "rotary_embedding_dim", 0)
    scale = get_attribute(node, "scale", 1.0)

    def _rotary_embedding(
        x: torch.Tensor,
        pos_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        interleaved: int,
        num_heads: int,
        rotary_dim: int,
        scale: float,
    ) -> torch.Tensor:
        """Apply rotary position embeddings."""
        original_shape = x.shape
        is_3d = x.dim() == 3

        if is_3d:
            # Input is (batch_size, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = x.shape

            # Determine head_size and num_heads
            if num_heads > 0:
                head_size = hidden_size // num_heads
                actual_num_heads = num_heads
            else:
                # Infer head_size from cos_cache dimension
                # cos_cache has shape (max_seq, rotary_dim/2)
                rotary_half_dim = cos_cache.shape[-1]
                head_size = rotary_half_dim * 2  # rotary_dim == head_size typically
                actual_num_heads = hidden_size // head_size

            # Reshape to (batch, num_heads, seq, head_size)
            x = x.view(batch_size, seq_len, actual_num_heads, head_size).transpose(1, 2)
        else:
            # Input is (batch_size, num_heads, seq_len, head_size)
            batch_size, actual_num_heads, seq_len, head_size = x.shape

        # Get cos/sin values for positions
        # position_ids can be (1,) scalar or (batch, seq) or (seq,)
        if pos_ids.dim() == 1:
            if pos_ids.numel() == 1:
                # Single position offset - generate sequence
                start_pos = pos_ids.item()
                positions = torch.arange(
                    start_pos, start_pos + seq_len, device=x.device, dtype=torch.long
                )
            else:
                positions = pos_ids
        else:
            # (batch, seq) - use first batch for now (they should be the same)
            positions = pos_ids[0] if pos_ids.shape[0] > 1 else pos_ids.squeeze(0)

        # Gather cos/sin from cache based on positions
        cos = cos_cache[positions]  # (seq_len, rotary_dim/2)
        sin = sin_cache[positions]  # (seq_len, rotary_dim/2)

        # Determine rotary dimension
        if rotary_dim > 0:
            rot_dim = rotary_dim
        else:
            rot_dim = cos.shape[-1] * 2  # cos/sin cache is half the rotary dim

        # Expand cos/sin for batch and heads: (1, 1, seq_len, rotary_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply scale if specified
        if scale != 1.0:
            x = x * scale

        # Split into rotary and pass-through parts
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:] if rot_dim < x.shape[-1] else None

        if interleaved:
            # Interleaved format: [x0, y0, x1, y1, ...] pairs
            # Rotate pairs: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
            x1 = x_rot[..., ::2]  # Even indices
            x2 = x_rot[..., 1::2]  # Odd indices

            # Make sure cos/sin match the half dimension
            cos_half = cos[..., : x1.shape[-1]]
            sin_half = sin[..., : x1.shape[-1]]

            # Apply rotation
            x_rot_new = torch.stack(
                [x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half], dim=-1
            ).flatten(-2)
        else:
            # Non-interleaved format: first half real, second half imaginary
            # x = [x1, x2] where x1 and x2 are halves
            half_dim = rot_dim // 2
            x1 = x_rot[..., :half_dim]
            x2 = x_rot[..., half_dim:rot_dim]

            # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
            x_rot_new = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # Concatenate with pass-through part
        if x_pass is not None:
            x_out = torch.cat([x_rot_new, x_pass], dim=-1)
        else:
            x_out = x_rot_new

        # Reshape back to original shape
        if is_3d:
            # Always reshape back from (batch, num_heads, seq, head_size) to (batch, seq, hidden)
            x_out = x_out.transpose(1, 2).contiguous().view(original_shape)

        return x_out

    return builder.call_function(
        _rotary_embedding,
        args=(
            input_tensor,
            position_ids,
            cos_cache,
            sin_cache,
            interleaved,
            num_heads,
            rotary_embedding_dim,
            scale,
        ),
    )
