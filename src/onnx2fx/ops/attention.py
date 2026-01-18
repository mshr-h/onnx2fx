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
# Attention operator (ONNX standard domain, since opset 24)
# =============================================================================


@register("Attention")
def attention(builder: "GraphBuilder", node: onnx.NodeProto) -> torch.fx.Node:
    """ONNX Attention operator (standard domain, since opset 24).

    Inputs:
        Q: Query tensor
           - 4D: (batch_size, q_num_heads, q_sequence_length, head_size)
           - 3D: (batch_size, q_sequence_length, q_num_heads * head_size)
        K: Key tensor
           - 4D: (batch_size, kv_num_heads, kv_sequence_length, head_size)
           - 3D: (batch_size, kv_sequence_length, kv_num_heads * head_size)
        V: Value tensor (same format as K)
        attn_mask (optional): Attention mask, broadcastable to
            (batch_size, q_num_heads, q_sequence_length, total_sequence_length)
        past_key (optional): Past key cache
        past_value (optional): Past value cache
        nonpad_kv_seqlen (optional): Non-padding KV sequence lengths

    Attributes:
        is_causal: If 1, use causal (lower triangular) mask
        scale: Scaling factor for Q*K^T (default: 1/sqrt(head_size))
        softcap: Softcap value for attention weights
        q_num_heads: Number of query heads (required for 3D inputs)
        kv_num_heads: Number of key/value heads (required for 3D inputs)
        qk_matmul_output_mode: Output mode for QK matmul (0, 1, or 2)

    Outputs:
        Y: Output tensor (same format as Q)
        present_key (optional): Updated key cache
        present_value (optional): Updated value cache
        qk_matmul_output (optional): QK matmul output
    """
    # Get inputs
    query = builder.get_value(node.input[0])
    key = builder.get_value(node.input[1])
    value = builder.get_value(node.input[2])

    attn_mask = (
        builder.get_value(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else None
    )
    past_key = (
        builder.get_value(node.input[4])
        if len(node.input) > 4 and node.input[4]
        else None
    )
    past_value = (
        builder.get_value(node.input[5])
        if len(node.input) > 5 and node.input[5]
        else None
    )
    nonpad_kv_seqlen = (
        builder.get_value(node.input[6])
        if len(node.input) > 6 and node.input[6]
        else None
    )

    # Get attributes
    is_causal = get_attribute(node, "is_causal", 0)
    scale = get_attribute(node, "scale", None)
    softcap = get_attribute(node, "softcap", 0.0)
    q_num_heads = get_attribute(node, "q_num_heads", None)
    kv_num_heads = get_attribute(node, "kv_num_heads", None)
    qk_matmul_output_mode = get_attribute(node, "qk_matmul_output_mode", 0)

    # Determine which outputs are needed
    # Output positions: 0=Y, 1=present_key, 2=present_value, 3=qk_matmul_output
    num_outputs = len(node.output)
    has_present_key = num_outputs > 1 and node.output[1]
    has_present_value = num_outputs > 2 and node.output[2]
    has_qk_matmul_output = num_outputs > 3 and node.output[3]

    # Check if we need multiple outputs (even if some are empty)
    _needs_multiple_outputs = num_outputs > 1

    # Use manual attention computation when:
    # 1. We need qk_matmul_output
    # 2. We have softcap
    # 3. We have both is_causal and attn_mask
    needs_manual_attention = (
        has_qk_matmul_output or softcap != 0.0 or (is_causal and attn_mask is not None)
    )

    if needs_manual_attention:
        # Manual attention implementation for advanced features

        def _attention_manual(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None,
            past_k: torch.Tensor | None,
            past_v: torch.Tensor | None,
            is_causal: int,
            scale: float | None,
            softcap: float,
            q_num_heads: int | None,
            kv_num_heads: int | None,
            num_outputs: int,
            qk_matmul_output_mode: int,
        ) -> (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            is_3d = q.dim() == 3

            if is_3d:
                batch_size = q.shape[0]
                q_seq_len = q.shape[1]
                kv_seq_len = k.shape[1]
                n_q_heads = q_num_heads if q_num_heads is not None else 1
                n_kv_heads = kv_num_heads if kv_num_heads is not None else n_q_heads
                q_head_size = q.shape[2] // n_q_heads
                kv_head_size = k.shape[2] // n_kv_heads
                v_head_size = v.shape[2] // n_kv_heads
                q_4d = q.view(batch_size, q_seq_len, n_q_heads, q_head_size).transpose(
                    1, 2
                )
                k_4d = k.view(
                    batch_size, kv_seq_len, n_kv_heads, kv_head_size
                ).transpose(1, 2)
                v_4d = v.view(
                    batch_size, kv_seq_len, n_kv_heads, v_head_size
                ).transpose(1, 2)
            else:
                q_4d = q
                k_4d = k
                v_4d = v
                batch_size = q.shape[0]
                n_q_heads = q.shape[1]
                n_kv_heads = k.shape[1]
                q_seq_len = q.shape[2]
                q_head_size = q.shape[3]

            # Handle past key/value (KV cache)
            past_seq_len = 0
            if past_k is not None and past_v is not None:
                past_seq_len = past_k.shape[2]
                k_4d = torch.cat([past_k, k_4d], dim=2)
                v_4d = torch.cat([past_v, v_4d], dim=2)

            # Save present_key and present_value before GQA expansion (if needed for output)
            present_k = k_4d if num_outputs > 1 else None
            present_v = v_4d if num_outputs > 2 else None

            # Handle GQA: expand KV heads to match Q heads
            if n_kv_heads != n_q_heads:
                n_rep = n_q_heads // n_kv_heads
                k_4d = k_4d.repeat_interleave(n_rep, dim=1)
                v_4d = v_4d.repeat_interleave(n_rep, dim=1)

            # Compute attention scale
            if scale is None:
                scale_val = 1.0 / (q_head_size**0.5)
            else:
                scale_val = scale

            # Q @ K^T with scaling
            # (batch, heads, q_seq, head) @ (batch, heads, head, kv_seq)
            # -> (batch, heads, q_seq, kv_seq)
            qk = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale_val

            # Save QK matmul output before applying mask/softmax (if needed for output)
            # Mode 0: raw QK matmul output
            qk_output = None
            if num_outputs > 3 and qk_matmul_output_mode == 0:
                qk_output = qk.clone()

            # Build combined attention mask (applied BEFORE softcap per ONNX spec)
            # The ONNX approach: create causal mask first, add attn_mask to it,
            # then add combined mask to QK scores
            kv_seq = k_4d.shape[2]
            combined_mask = None

            # Create causal mask if is_causal=1
            # ONNX uses: Less(q_pos + past_len, k_pos) to determine masked positions
            # This creates a strictly lower triangular mask where q_pos + past_len >= k_pos is allowed
            if is_causal:
                # Create causal mask: (q_pos + past_seq_len) < k_pos means masked
                row = (
                    torch.arange(q_seq_len, device=q.device).view(-1, 1) + past_seq_len
                )
                col = torch.arange(kv_seq, device=q.device).view(1, -1)
                causal_bool = row < col  # True where masked
                causal_mask = (
                    torch.where(causal_bool, float("-inf"), 0.0)
                    .to(q.dtype)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                combined_mask = causal_mask

            # Add attention mask to causal mask (or use just attn_mask if no causal)
            if mask is not None:
                if combined_mask is not None:
                    combined_mask = mask + combined_mask  # attn_mask + causal_mask
                else:
                    combined_mask = mask

            # Add combined mask to QK scores
            if combined_mask is not None:
                qk = qk + combined_mask

            # Mode 1: after attention mask addition (including causal)
            if num_outputs > 3 and qk_matmul_output_mode == 1:
                qk_output = qk.clone()

            # Apply softcap if specified (after mask, before softmax per ONNX spec)
            if softcap != 0.0:
                qk = softcap * torch.tanh(qk / softcap)

            # Mode 2: after softcap
            if num_outputs > 3 and qk_matmul_output_mode == 2:
                qk_output = qk.clone()

            # Softmax
            attn_weights = torch.nn.functional.softmax(qk, dim=-1)

            # Mode 3: after softmax
            if num_outputs > 3 and qk_matmul_output_mode == 3:
                qk_output = attn_weights.clone()

            # Attention @ V
            output = torch.matmul(attn_weights, v_4d)

            if is_3d:
                output = (
                    output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)
                )

            # Return based on num_outputs (must match exactly)
            # Output positions: 0=Y, 1=present_key, 2=present_value, 3=qk_matmul_output
            if num_outputs == 1:
                return output
            elif num_outputs == 2:
                return (output, present_k)
            elif num_outputs == 3:
                return (output, present_k, present_v)
            else:  # num_outputs == 4
                return (output, present_k, present_v, qk_output)

        return builder.call_function(
            _attention_manual,
            args=(
                query,
                key,
                value,
                attn_mask,
                past_key,
                past_value,
                is_causal,
                scale,
                softcap,
                q_num_heads,
                kv_num_heads,
                num_outputs,
                qk_matmul_output_mode,
            ),
        )
    elif has_present_key or has_present_value:
        # Use SDPA but also return present_key/present_value

        def _attention_with_cache(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None,
            past_k: torch.Tensor | None,
            past_v: torch.Tensor | None,
            is_causal: int,
            scale: float | None,
            q_num_heads: int | None,
            kv_num_heads: int | None,
            has_present_key: bool,
            has_present_value: bool,
        ) -> (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            is_3d = q.dim() == 3

            if is_3d:
                batch_size = q.shape[0]
                q_seq_len = q.shape[1]
                kv_seq_len = k.shape[1]
                n_q_heads = q_num_heads if q_num_heads is not None else 1
                n_kv_heads = kv_num_heads if kv_num_heads is not None else n_q_heads
                q_head_size = q.shape[2] // n_q_heads
                kv_head_size = k.shape[2] // n_kv_heads
                v_head_size = v.shape[2] // n_kv_heads
                q_4d = q.view(batch_size, q_seq_len, n_q_heads, q_head_size).transpose(
                    1, 2
                )
                k_4d = k.view(
                    batch_size, kv_seq_len, n_kv_heads, kv_head_size
                ).transpose(1, 2)
                v_4d = v.view(
                    batch_size, kv_seq_len, n_kv_heads, v_head_size
                ).transpose(1, 2)
            else:
                q_4d = q
                k_4d = k
                v_4d = v
                batch_size = q.shape[0]
                n_q_heads = q.shape[1]
                n_kv_heads = k.shape[1]
                q_seq_len = q.shape[2]

            # Handle past key/value (KV cache)
            if past_k is not None and past_v is not None:
                k_4d = torch.cat([past_k, k_4d], dim=2)
                v_4d = torch.cat([past_v, v_4d], dim=2)

            # Save present_key and present_value before GQA expansion
            present_k = k_4d if has_present_key else None
            present_v = v_4d if has_present_value else None

            # Handle GQA: expand KV heads to match Q heads
            if n_kv_heads != n_q_heads:
                n_rep = n_q_heads // n_kv_heads
                k_4d = k_4d.repeat_interleave(n_rep, dim=1)
                v_4d = v_4d.repeat_interleave(n_rep, dim=1)

            # Call SDPA
            if mask is not None:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, attn_mask=mask, is_causal=False, scale=scale
                )
            elif is_causal:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, is_causal=True, scale=scale
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, scale=scale
                )

            if is_3d:
                output = (
                    output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)
                )

            # Build return tuple
            results = [output]
            if has_present_key:
                results.append(present_k)
            if has_present_value:
                results.append(present_v)

            if len(results) == 1:
                return results[0]
            return tuple(results)

        return builder.call_function(
            _attention_with_cache,
            args=(
                query,
                key,
                value,
                attn_mask,
                past_key,
                past_value,
                is_causal,
                scale,
                q_num_heads,
                kv_num_heads,
                has_present_key,
                has_present_value,
            ),
        )
    else:
        # Simple case: just use SDPA

        def _attention_standard(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None,
            past_k: torch.Tensor | None,
            past_v: torch.Tensor | None,
            is_causal: int,
            scale: float | None,
            q_num_heads: int | None,
            kv_num_heads: int | None,
            nonpad_kv_seqlen: torch.Tensor | None,
        ) -> torch.Tensor:
            is_3d = q.dim() == 3

            if is_3d:
                batch_size = q.shape[0]
                q_seq_len = q.shape[1]
                kv_seq_len = k.shape[1]
                n_q_heads = q_num_heads if q_num_heads is not None else 1
                n_kv_heads = kv_num_heads if kv_num_heads is not None else n_q_heads
                q_head_size = q.shape[2] // n_q_heads
                kv_head_size = k.shape[2] // n_kv_heads
                v_head_size = v.shape[2] // n_kv_heads
                q_4d = q.view(batch_size, q_seq_len, n_q_heads, q_head_size).transpose(
                    1, 2
                )
                k_4d = k.view(
                    batch_size, kv_seq_len, n_kv_heads, kv_head_size
                ).transpose(1, 2)
                v_4d = v.view(
                    batch_size, kv_seq_len, n_kv_heads, v_head_size
                ).transpose(1, 2)
            else:
                q_4d = q
                k_4d = k
                v_4d = v
                batch_size = q.shape[0]
                n_q_heads = q.shape[1]
                n_kv_heads = k.shape[1]
                q_seq_len = q.shape[2]

            # Handle past key/value (KV cache)
            if past_k is not None and past_v is not None:
                k_4d = torch.cat([past_k, k_4d], dim=2)
                v_4d = torch.cat([past_v, v_4d], dim=2)

            # Handle GQA: expand KV heads to match Q heads
            if n_kv_heads != n_q_heads:
                n_rep = n_q_heads // n_kv_heads
                k_4d = k_4d.repeat_interleave(n_rep, dim=1)
                v_4d = v_4d.repeat_interleave(n_rep, dim=1)

            # Handle mask padding if mask is shorter than KV sequence
            # Per ONNX spec: "The last dimension can also be shorter than
            # total_sequence_length and will be padded with negative infinity"
            kv_seq_len_actual = k_4d.shape[2]

            if mask is not None:
                # Pad mask if shorter than KV sequence (BEFORE adding nonpad mask)
                if mask.shape[-1] < kv_seq_len_actual:
                    pad_size = kv_seq_len_actual - mask.shape[-1]
                    # For bool masks, pad with False; for float, pad with -inf
                    if mask.dtype == torch.bool:
                        mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
                    else:
                        mask = torch.nn.functional.pad(
                            mask, (0, pad_size), value=float("-inf")
                        )

                # Expand mask for GQA if needed (mask might have n_kv_heads dimension)
                if (
                    mask.dim() == 4
                    and mask.shape[1] == n_kv_heads
                    and n_kv_heads != n_q_heads
                ):
                    n_rep = n_q_heads // n_kv_heads
                    mask = mask.repeat_interleave(n_rep, dim=1)

            # Handle nonpad_kv_seqlen: create a padding mask for each batch
            # that masks out positions >= nonpad_kv_seqlen[batch]
            if nonpad_kv_seqlen is not None:
                # Create a position index tensor: (1, 1, 1, kv_seq_len)
                positions = torch.arange(kv_seq_len_actual, device=q.device).view(
                    1, 1, 1, -1
                )
                # Create mask: True where position < nonpad_kv_seqlen[batch]
                # nonpad_kv_seqlen: (batch_size,) -> (batch_size, 1, 1, 1)
                valid_mask = positions < nonpad_kv_seqlen.view(-1, 1, 1, 1)
                # Convert to additive mask: 0 for valid, -inf for padding
                pad_mask = torch.where(valid_mask, 0.0, float("-inf")).to(q.dtype)
                # Combine with existing mask
                if mask is not None:
                    mask = mask + pad_mask
                else:
                    mask = pad_mask

            # Call SDPA
            if mask is not None:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, attn_mask=mask, is_causal=False, scale=scale
                )
            elif is_causal:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, is_causal=True, scale=scale
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, scale=scale
                )

            if is_3d:
                output = (
                    output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)
                )

            return output

        return builder.call_function(
            _attention_standard,
            args=(
                query,
                key,
                value,
                attn_mask,
                past_key,
                past_value,
                is_causal,
                scale,
                q_num_heads,
                kv_num_heads,
                nonpad_kv_seqlen,
            ),
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
# Rotary Embedding (ONNX standard domain, since opset 23)
# =============================================================================


@register("RotaryEmbedding", since_version=23)
def rotary_embedding_onnx(
    builder: "GraphBuilder", node: onnx.NodeProto
) -> torch.fx.Node:
    """ONNX RotaryEmbedding operator (standard domain, since opset 23).

    Applies rotary position embeddings (RoPE) to the input tensor based on
    https://arxiv.org/pdf/2104.09864

    Inputs:
        - X: 4D tensor (batch_size, num_heads, sequence_length, head_size) or
             3D tensor (batch_size, sequence_length, hidden_size)
        - cos_cache: 2D tensor (max_position_id+1, head_size/2) when position_ids provided,
                     or 3D tensor (batch_size, sequence_length, head_size/2) otherwise
        - sin_cache: Same shape as cos_cache
        - position_ids (optional): 2D tensor (batch_size, sequence_length)

    Attributes:
        - interleaved: Whether to use interleaved pattern. Default is 0 (False).
        - num_heads: Number of attention heads (required for 3D input).
        - rotary_embedding_dim: Partial rotary dimension. Default is 0 (full rotation).

    Outputs:
        - Y: Tensor with same shape as input.
    """
    # Get inputs
    input_tensor = builder.get_value(node.input[0])
    cos_cache = builder.get_value(node.input[1])
    sin_cache = builder.get_value(node.input[2])
    position_ids = (
        builder.get_value(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else None
    )

    # Get attributes
    interleaved = get_attribute(node, "interleaved", 0)
    num_heads = get_attribute(node, "num_heads", 0)
    rotary_embedding_dim = get_attribute(node, "rotary_embedding_dim", 0)

    def _rotary_embedding_onnx(
        x: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        position_ids: torch.Tensor | None,
        interleaved: int,
        num_heads: int,
        rotary_dim: int,
    ) -> torch.Tensor:
        """Apply ONNX-standard rotary position embeddings."""
        original_shape = x.shape
        is_3d = x.dim() == 3

        # First ensure input has shape [batch_size, seq_len, num_heads, head_size]
        if x.dim() == 4:
            # Input is (batch_size, num_heads, seq_len, head_size)
            # Transpose to (batch_size, seq_len, num_heads, head_size)
            x = x.transpose(1, 2)
            batch_size, seq_len, n_heads, head_size = x.shape
        else:
            # Input is (batch_size, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = x.shape
            assert num_heads != 0, "num_heads must be provided for 3D input"
            head_size = hidden_size // num_heads
            x = x.view(batch_size, seq_len, num_heads, head_size)
            _n_heads = num_heads

        # Determine rotary_embedding_dim
        if rotary_dim == 0:
            rot_dim = head_size
        else:
            rot_dim = rotary_dim

        rotary_dim_half = rot_dim // 2

        # Split into rotary and pass-through parts
        x_rotate = x[..., :rot_dim]
        x_not_rotate = x[..., rot_dim:] if rot_dim < head_size else None

        # Retrieve sin and cos caches using position ids
        if position_ids is not None:
            # cos_cache/sin_cache shape: (max_pos+1, rotary_dim/2)
            # position_ids shape: (batch_size, seq_len)
            # Result shape: (batch_size, seq_len, rotary_dim/2)
            cos = cos_cache[position_ids]
            sin = sin_cache[position_ids]
        else:
            # cos_cache/sin_cache already have shape (batch_size, seq_len, rotary_dim/2)
            cos = cos_cache
            sin = sin_cache

        # Validate cache dimensions
        if cos.shape[-1] != rotary_dim_half:
            raise ValueError(
                f"Last dimension of cos cache ({cos.shape[-1]}) does not match "
                f"rotary_embedding_dim/2 ({rotary_dim_half})."
            )

        # Add head dimension: (batch_size, seq_len, 1, rotary_dim/2)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        # Apply rotation based on interleaved pattern
        if interleaved:
            # Interleaved: x_rotate[..., 0::2] and x_rotate[..., 1::2]
            x1 = x_rotate[..., 0::2]
            x2 = x_rotate[..., 1::2]

            # Calculate real and imaginary values
            real = (cos * x1) - (sin * x2)
            imag = (sin * x1) + (cos * x2)

            # Interleave back
            real = real.unsqueeze(-1)
            imag = imag.unsqueeze(-1)
            x_rotate_result = torch.cat((real, imag), dim=-1).flatten(-2)
        else:
            # Non-interleaved: split in halves
            x1 = x_rotate[..., :rotary_dim_half]
            x2 = x_rotate[..., rotary_dim_half:rot_dim]

            # Calculate real and imaginary values
            real = (cos * x1) - (sin * x2)
            imag = (sin * x1) + (cos * x2)

            x_rotate_result = torch.cat((real, imag), dim=-1)

        # Concatenate with non-rotated part
        if x_not_rotate is not None:
            output = torch.cat((x_rotate_result, x_not_rotate), dim=-1)
        else:
            output = x_rotate_result

        # Reshape back to original shape
        if is_3d:
            output = output.view(original_shape)
        else:
            # Transpose back to (batch_size, num_heads, seq_len, head_size)
            output = output.transpose(1, 2)

        return output

    return builder.call_function(
        _rotary_embedding_onnx,
        args=(
            input_tensor,
            cos_cache,
            sin_cache,
            position_ids,
            interleaved,
            num_heads,
            rotary_embedding_dim,
        ),
    )


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
