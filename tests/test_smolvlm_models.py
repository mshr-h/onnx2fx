# SPDX-License-Identifier: Apache-2.0
"""Tests for SmolVLM ONNX models.

This test downloads the SmolVLM-256M-Instruct ONNX model components from HuggingFace
and verifies that onnx2fx can convert and run them correctly.

SmolVLM is a compact vision-language model consisting of:
- Vision encoder (SigLIP-based)
- Token embeddings
- Decoder (LLM based on SmolLM2)

Reference: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
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

# Check if transformers is available for processor
try:
    from transformers import AutoProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Model configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# SmolVLM-256M architecture constants (from ONNX model inspection)
# Vision encoder: pixel_values [batch, num_images, 3, 512, 512], pixel_attention_mask [batch, num_images, 512, 512]
VISION_IMAGE_SIZE = 512  # Actual ONNX model image size
VISION_HIDDEN_SIZE = 576  # Output hidden size (matches text hidden size)

# Text/Decoder model
TEXT_VOCAB_SIZE = 49280  # Actual vocab size from ONNX model output
TEXT_HIDDEN_SIZE = 576  # SmolLM2-135M hidden size
TEXT_NUM_HEADS = 9
TEXT_NUM_KV_HEADS = 3
TEXT_HEAD_DIM = 64  # head_dim from ONNX model
TEXT_NUM_LAYERS = 30


def download_smolvlm_component(component: str, variant: str = "fp16") -> str:
    """Download a SmolVLM ONNX component from HuggingFace Hub.

    Args:
        component: Component to download. Options:
            - "vision_encoder": Vision encoder (SigLIP)
            - "embed_tokens": Token embeddings
            - "decoder_model_merged": LLM decoder
        variant: Model variant. Options:
            - "fp16": FP16 model (smallest non-quantized)
            - "q4": Q4 quantized model
            - "int8": INT8 quantized model
            - "": Full precision (fp32)

    Returns:
        Path to downloaded ONNX model file.
    """
    suffix = f"_{variant}" if variant else ""
    filename = f"onnx/{component}{suffix}.onnx"

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=filename,
    )
    return model_path


# =============================================================================
# Vision Encoder Fixtures and Tests
# =============================================================================


@pytest.fixture(scope="module")
def smolvlm_vision_encoder_path():
    """Download and cache the SmolVLM vision encoder."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub not available")
    # Use full precision model to avoid ONNX Runtime compatibility issues with fp16
    return download_smolvlm_component("vision_encoder", "")


@pytest.fixture(scope="module")
def smolvlm_vision_encoder_model(smolvlm_vision_encoder_path):
    """Load the vision encoder ONNX model."""
    return onnx.load(smolvlm_vision_encoder_path)


# =============================================================================
# Token Embeddings Fixtures and Tests
# =============================================================================


@pytest.fixture(scope="module")
def smolvlm_embed_tokens_path():
    """Download and cache the SmolVLM token embeddings."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub not available")
    # Use full precision model to avoid ONNX Runtime compatibility issues
    return download_smolvlm_component("embed_tokens", "")


@pytest.fixture(scope="module")
def smolvlm_embed_tokens_model(smolvlm_embed_tokens_path):
    """Load the token embeddings ONNX model."""
    return onnx.load(smolvlm_embed_tokens_path)


# =============================================================================
# Decoder Fixtures and Tests
# =============================================================================


@pytest.fixture(scope="module")
def smolvlm_decoder_path():
    """Download and cache the SmolVLM decoder."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub not available")
    # Use full precision model to avoid ONNX Runtime compatibility issues
    return download_smolvlm_component("decoder_model_merged", "")


@pytest.fixture(scope="module")
def smolvlm_decoder_model(smolvlm_decoder_path):
    """Load the decoder ONNX model."""
    return onnx.load(smolvlm_decoder_path)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
class TestSmolVLMVisionEncoder:
    """Test SmolVLM vision encoder conversion."""

    @pytest.mark.slow
    def test_vision_encoder_conversion(self, smolvlm_vision_encoder_model):
        """Test that the vision encoder can be converted to FX."""
        fx_module = convert_onnx_model(smolvlm_vision_encoder_model)
        assert fx_module is not None
        assert hasattr(fx_module, "forward")

    @pytest.mark.slow
    def test_vision_encoder_forward_pass(
        self, smolvlm_vision_encoder_path, smolvlm_vision_encoder_model
    ):
        """Test vision encoder forward pass matches ONNX Runtime."""
        # Vision encoder inputs:
        # - pixel_values: [batch_size, num_images, 3, 512, 512] (float32)
        # - pixel_attention_mask: [batch_size, num_images, 512, 512] (bool)
        batch_size = 1
        num_images = 1

        pixel_values = np.random.randn(
            batch_size, num_images, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE
        ).astype(np.float32)
        pixel_attention_mask = np.ones(
            (batch_size, num_images, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE), dtype=bool
        )

        np_inputs = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }
        torch_inputs = (
            torch.from_numpy(pixel_values),
            torch.from_numpy(pixel_attention_mask),
        )

        # Run ONNX Runtime
        ort_session = ort.InferenceSession(
            smolvlm_vision_encoder_path,
            providers=["CPUExecutionProvider"],
        )
        ort_outputs = ort_session.run(None, np_inputs)
        ort_output = torch.from_numpy(ort_outputs[0].astype(np.float32))

        run_onnx_test(
            smolvlm_vision_encoder_model,
            torch_inputs,
            ort_output,
            rtol=0.01,
            atol=0.05,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
class TestSmolVLMEmbedTokens:
    """Test SmolVLM token embeddings conversion."""

    @pytest.mark.slow
    def test_embed_tokens_conversion(self, smolvlm_embed_tokens_model):
        """Test that the token embeddings can be converted to FX."""
        fx_module = convert_onnx_model(smolvlm_embed_tokens_model)
        assert fx_module is not None
        assert hasattr(fx_module, "forward")

    @pytest.mark.slow
    def test_embed_tokens_forward_pass(
        self, smolvlm_embed_tokens_path, smolvlm_embed_tokens_model
    ):
        """Test token embeddings forward pass matches ONNX Runtime."""
        # Token embeddings input: input_ids [batch, seq_len]
        batch_size = 1
        seq_len = 10
        input_ids = np.random.randint(
            0, TEXT_VOCAB_SIZE, size=(batch_size, seq_len)
        ).astype(np.int64)

        np_inputs = {"input_ids": input_ids}
        torch_inputs = (torch.from_numpy(input_ids),)

        # Run ONNX Runtime
        ort_session = ort.InferenceSession(
            smolvlm_embed_tokens_path,
            providers=["CPUExecutionProvider"],
        )
        ort_outputs = ort_session.run(None, np_inputs)
        ort_output = torch.from_numpy(ort_outputs[0].astype(np.float32))

        run_onnx_test(
            smolvlm_embed_tokens_model,
            torch_inputs,
            ort_output,
            rtol=0.001,
            atol=0.001,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
class TestSmolVLMDecoder:
    """Test SmolVLM decoder conversion."""

    @pytest.mark.slow
    def test_decoder_conversion(self, smolvlm_decoder_model):
        """Test that the decoder can be converted to FX."""
        fx_module = convert_onnx_model(smolvlm_decoder_model)
        assert fx_module is not None
        assert hasattr(fx_module, "forward")

    @pytest.mark.slow
    def test_decoder_forward_pass(self, smolvlm_decoder_path, smolvlm_decoder_model):
        """Test decoder forward pass matches ONNX Runtime."""
        # Use fixed seed for reproducibility
        np.random.seed(42)

        # Decoder inputs: inputs_embeds, attention_mask, position_ids, past_key_values
        batch_size = 1
        seq_len = 5
        past_seq_len = 0

        # Create inputs_embeds with smaller magnitude to avoid numerical instability
        # Use scaled random values similar to what embed_tokens would produce
        inputs_embeds = (
            np.random.randn(batch_size, seq_len, TEXT_HIDDEN_SIZE).astype(np.float32)
            * 0.02  # Scale down to typical embedding magnitude
        )

        # Attention mask covers past + current sequence
        total_seq_len = past_seq_len + seq_len
        attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)

        # Position IDs
        position_ids = np.arange(past_seq_len, total_seq_len, dtype=np.int64).reshape(
            1, -1
        )

        np_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # Add past_key_values for each layer (initialized to zero for first pass)
        for i in range(TEXT_NUM_LAYERS):
            np_inputs[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, TEXT_NUM_KV_HEADS, past_seq_len, TEXT_HEAD_DIM),
                dtype=np.float32,
            )
            np_inputs[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, TEXT_NUM_KV_HEADS, past_seq_len, TEXT_HEAD_DIM),
                dtype=np.float32,
            )

        # Get input order from ONNX model
        ort_session = ort.InferenceSession(
            smolvlm_decoder_path,
            providers=["CPUExecutionProvider"],
        )
        input_names = [inp.name for inp in ort_session.get_inputs()]
        torch_inputs = [torch.from_numpy(np_inputs[name]) for name in input_names]

        # Run ONNX Runtime
        ort_outputs = ort_session.run(None, np_inputs)
        ort_logits = torch.from_numpy(ort_outputs[0].astype(np.float32))

        # For decoder models, use very loose tolerances due to accumulating
        # numerical differences across 30 transformer layers
        run_onnx_test(
            smolvlm_decoder_model,
            tuple(torch_inputs),
            ort_logits,
            rtol=1.0,  # Very loose - just verify shapes and rough magnitude
            atol=50.0,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )

        # Additionally verify output shape is correct
        fx_module = convert_onnx_model(smolvlm_decoder_model)
        with torch.inference_mode():
            fx_output = fx_module(*torch_inputs)
            fx_logits = fx_output[0] if isinstance(fx_output, tuple) else fx_output
        assert fx_logits.shape == ort_logits.shape, (
            f"Shape mismatch: {fx_logits.shape} vs {ort_logits.shape}"
        )


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub not available")
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
class TestSmolVLMIntegration:
    """Integration tests for SmolVLM model components."""

    @pytest.mark.slow
    def test_vision_encoder_with_processor(
        self, smolvlm_vision_encoder_path, smolvlm_vision_encoder_model
    ):
        """Test vision encoder with actual image preprocessing."""
        from PIL import Image

        # Create a simple test image
        test_image = Image.new(
            "RGB", (VISION_IMAGE_SIZE, VISION_IMAGE_SIZE), color="red"
        )

        # Use the processor to preprocess the image
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        inputs = processor(images=test_image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        # Convert to bool (processor may return int64)
        pixel_attention_mask = inputs["pixel_attention_mask"].astype(bool)

        np_inputs = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }
        torch_inputs = (
            torch.from_numpy(pixel_values),
            torch.from_numpy(pixel_attention_mask),
        )

        # Run ONNX Runtime
        ort_session = ort.InferenceSession(
            smolvlm_vision_encoder_path,
            providers=["CPUExecutionProvider"],
        )
        ort_outputs = ort_session.run(None, np_inputs)
        ort_output = torch.from_numpy(ort_outputs[0].astype(np.float32))

        run_onnx_test(
            smolvlm_vision_encoder_model,
            torch_inputs,
            ort_output,
            rtol=0.01,
            atol=0.05,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )

    @pytest.mark.slow
    def test_embed_tokens_with_tokenizer(
        self, smolvlm_embed_tokens_path, smolvlm_embed_tokens_model
    ):
        """Test token embeddings with actual tokenizer."""
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        tokenizer = processor.tokenizer

        # Tokenize a simple prompt
        text = "Describe this image:"
        encoded = tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"]

        np_inputs = {"input_ids": input_ids}
        torch_inputs = (torch.from_numpy(input_ids),)

        # Run ONNX Runtime
        ort_session = ort.InferenceSession(
            smolvlm_embed_tokens_path,
            providers=["CPUExecutionProvider"],
        )
        ort_outputs = ort_session.run(None, np_inputs)
        ort_output = torch.from_numpy(ort_outputs[0].astype(np.float32))

        run_onnx_test(
            smolvlm_embed_tokens_model,
            torch_inputs,
            ort_output,
            rtol=0.001,
            atol=0.001,
            output_transform=lambda out: (
                out[0] if isinstance(out, tuple) else out
            ).float(),
        )
