# SPDX-License-Identifier: Apache-2.0
"""Tests for LFM2-350M-ENJP-MT ONNX model.

This test downloads the LFM2-350M-ENJP-MT-ONNX model from HuggingFace
and verifies that onnx2fx can convert and run it correctly.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2fx import convert

# Check if huggingface_hub is available
try:
    from huggingface_hub import hf_hub_download

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# Check if transformers is available for tokenizer
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


MODEL_ID = "onnx-community/LFM2-350M-ENJP-MT-ONNX"


def download_model(model_variant: str = "model_fp16") -> str:
    """Download ONNX model from HuggingFace Hub.

    Args:
        model_variant: Model variant to download. Options:
            - "model_fp16": FP16 model (725MB)
            - "model_q4f16": Q4F16 quantized model (312MB)
            - "model_quantized": INT8 quantized model (387MB)

    Returns:
        Path to downloaded ONNX model file.
    """
    # Download the model file
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=f"onnx/{model_variant}.onnx",
    )
    # Also download the data file (weights are stored separately)
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=f"onnx/{model_variant}.onnx_data",
    )
    return model_path


@pytest.fixture(scope="module")
def lfm2_model_path():
    """Download and cache the LFM2 model."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub not available")
    return download_model("model_fp16")


@pytest.fixture(scope="module")
def lfm2_onnx_model(lfm2_model_path):
    """Load the ONNX model."""
    return onnx.load(lfm2_model_path)


def create_lfm2_inputs(
    text: str,
    past_seq_len: int = 0,
):
    """Create inputs for LFM2 model.

    Args:
        text: Text to tokenize using the model's tokenizer.
        past_seq_len: Past sequence length for KV cache

    Returns:
        Dict of numpy inputs and list of torch inputs
    """
    hidden_size = 1024
    num_kv_heads = 8
    head_dim = 64
    conv_kernel = 3

    # Generate input_ids from text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask_base = encoded["attention_mask"]
    batch_size = input_ids.shape[0]

    # Extend attention_mask to include past sequence length
    if past_seq_len > 0:
        past_mask = np.ones((batch_size, past_seq_len), dtype=np.int64)
        attention_mask = np.concatenate([past_mask, attention_mask_base], axis=1)
    else:
        attention_mask = attention_mask_base

    np_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Add past_conv inputs (for causal conv layers)
    for i in [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]:
        np_inputs[f"past_conv.{i}"] = np.zeros(
            (batch_size, hidden_size, conv_kernel), dtype=np.float16
        )

    # Add past_key_values (for attention layers)
    for i in [2, 5, 8, 10, 12, 14]:
        np_inputs[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, past_seq_len, head_dim), dtype=np.float16
        )
        np_inputs[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, past_seq_len, head_dim), dtype=np.float16
        )

    # Create torch inputs in the correct order
    input_order = [
        "input_ids",
        "attention_mask",
        "past_conv.0",
        "past_conv.1",
        "past_key_values.2.key",
        "past_key_values.2.value",
        "past_conv.3",
        "past_conv.4",
        "past_key_values.5.key",
        "past_key_values.5.value",
        "past_conv.6",
        "past_conv.7",
        "past_key_values.8.key",
        "past_key_values.8.value",
        "past_conv.9",
        "past_key_values.10.key",
        "past_key_values.10.value",
        "past_conv.11",
        "past_key_values.12.key",
        "past_key_values.12.value",
        "past_conv.13",
        "past_key_values.14.key",
        "past_key_values.14.value",
        "past_conv.15",
    ]

    torch_inputs = [torch.from_numpy(np_inputs[name]) for name in input_order]

    return np_inputs, torch_inputs


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
class TestLFM2Model:
    """Test LFM2-350M-ENJP-MT ONNX model conversion."""

    @pytest.mark.slow
    def test_model_conversion(self, lfm2_onnx_model):
        """Test that the model can be converted to FX."""
        fx_module = convert(lfm2_onnx_model)
        assert fx_module is not None
        assert hasattr(fx_module, "forward")

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
    def test_model_forward_pass(self, lfm2_model_path, lfm2_onnx_model):
        """Test forward pass of converted model matches ONNX Runtime."""
        fx_module = convert(lfm2_onnx_model)

        # Create inputs from actual text
        test_prompt = "Hello, how are you today?"
        np_inputs, torch_inputs = create_lfm2_inputs(text=test_prompt, past_seq_len=0)

        # Run ONNX Runtime
        ort_session = ort.InferenceSession(
            lfm2_model_path,
            providers=["CPUExecutionProvider"],
        )
        ort_outputs = ort_session.run(None, np_inputs)

        # Run FX module
        with torch.inference_mode():
            fx_outputs = fx_module(*torch_inputs)

        # Compare first output (logits)
        fx_logits = fx_outputs[0] if isinstance(fx_outputs, tuple) else fx_outputs

        # Convert to numpy for comparison
        fx_numpy = fx_logits.float().numpy()
        ort_logits = ort_outputs[0].astype(np.float32)

        # Use relaxed tolerance for FP16 models
        # FP16 has limited precision, and slight implementation differences
        # in attention, RoPE, and LayerNorm can accumulate through 16 layers
        np.testing.assert_allclose(
            fx_numpy,
            ort_logits,
            rtol=0.05,
            atol=0.1,
            err_msg="Logits mismatch between FX and ONNX Runtime",
        )
