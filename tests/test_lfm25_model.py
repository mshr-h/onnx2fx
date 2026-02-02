# SPDX-License-Identifier: Apache-2.0
"""Tests for LFM2.5-1.2B-Instruct ONNX model.

This test downloads the LFM2.5-1.2B-Instruct-ONNX model from HuggingFace
and verifies that onnx2fx can convert and run it correctly.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from conftest import run_onnx_test, convert_onnx_model

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


MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"


def download_model(model_variant: str = "model_fp16") -> str:
    """Download ONNX model from HuggingFace Hub.

    Args:
        model_variant: Model variant to download. Options:
            - "model": FP32 model
            - "model_fp16": FP16 model (~2.35GB)
            - "model_q4": Q4 quantized model (~1.22GB)
            - "model_q8": Q8 quantized model (~1.77GB)

    Returns:
        Path to downloaded ONNX model file.
    """
    # Download the model file
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=f"onnx/{model_variant}.onnx",
    )
    # Also download the data file(s) (weights are stored separately)
    # FP16 model has multiple data files
    try:
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=f"onnx/{model_variant}.onnx_data",
        )
    except Exception:
        pass  # Some variants may not have _data file

    # Try to download additional data files for larger models
    for i in range(1, 5):
        try:
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=f"onnx/{model_variant}.onnx_data_{i}",
            )
        except Exception:
            break

    return model_path


@pytest.fixture(scope="module")
def lfm25_model_path():
    """Download and cache the LFM2.5 model."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub not available")
    return download_model("model_fp16")


@pytest.fixture(scope="module")
def lfm25_onnx_model(lfm25_model_path):
    """Load the ONNX model."""
    return onnx.load(lfm25_model_path)


def create_lfm25_inputs(
    text: str,
    ort_session: ort.InferenceSession,
    past_seq_len: int = 0,
):
    """Create inputs for LFM2.5 model.

    LFM2.5-1.2B model architecture:
    - 16 layers with layer_types: [conv, conv, full_attention, conv, conv,
      full_attention, conv, conv, full_attention, conv, full_attention,
      conv, full_attention, conv, full_attention, conv]
    - hidden_size: 2048
    - num_kv_heads: 8
    - head_dim: 64 (hidden_size / num_attention_heads = 2048 / 32)
    - conv_kernel (conv_L_cache): 3

    Args:
        text: Text to tokenize using the model's tokenizer.
        ort_session: ONNX Runtime session to get input specs from.
        past_seq_len: Past sequence length for KV cache

    Returns:
        Dict of numpy inputs and list of torch inputs
    """
    # Map ONNX types to numpy dtypes
    ONNX_DTYPE = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
    }

    # Generate input_ids from text using chat template
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = np.array(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
    )
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Create attention_mask
    attention_mask = np.ones((batch_size, seq_len + past_seq_len), dtype=np.int64)

    # Create position_ids
    position_ids = np.arange(past_seq_len, past_seq_len + seq_len, dtype=np.int64)
    position_ids = position_ids.reshape(1, -1)

    np_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Initialize cache inputs from ONNX model specs
    for inp in ort_session.get_inputs():
        if inp.name in {"input_ids", "attention_mask", "position_ids"}:
            continue
        # Determine shape - replace dynamic dims with appropriate values
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, int):
                shape.append(d)
            elif isinstance(d, str) and "sequence" in d.lower():
                shape.append(past_seq_len)
            else:
                shape.append(batch_size)
        dtype = ONNX_DTYPE.get(inp.type, np.float32)
        np_inputs[inp.name] = np.zeros(shape, dtype=dtype)

    # Create torch inputs in the correct order (matching model input order)
    input_order = [inp.name for inp in ort_session.get_inputs()]
    torch_inputs = [torch.from_numpy(np_inputs[name]) for name in input_order]

    return np_inputs, torch_inputs


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
class TestLFM25Model:
    """Test LFM2.5-1.2B-Instruct ONNX model conversion."""

    def test_model_conversion(self, lfm25_onnx_model):
        """Test that the model can be converted to FX."""
        fx_module = convert_onnx_model(lfm25_onnx_model)
        assert fx_module is not None
        assert hasattr(fx_module, "forward")

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
    def test_model_forward_pass(self, lfm25_model_path, lfm25_onnx_model):
        """Test forward pass of converted model matches ONNX Runtime."""
        # Create ONNX Runtime session first to get input specs
        ort_session = ort.InferenceSession(
            lfm25_model_path,
            providers=["CPUExecutionProvider"],
        )

        # Create inputs from actual text
        test_prompt = "What is the capital of France?"
        np_inputs, torch_inputs = create_lfm25_inputs(
            text=test_prompt, ort_session=ort_session, past_seq_len=0
        )

        # Run ONNX Runtime
        ort_outputs = ort_session.run(None, np_inputs)

        ort_logits = torch.from_numpy(ort_outputs[0].astype(np.float32))
        run_onnx_test(
            lfm25_onnx_model,
            tuple(torch_inputs),
            ort_logits,
            rtol=0.05,
            atol=0.1,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )
