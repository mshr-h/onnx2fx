# AGENTS.md

## Project Overview

**onnx2fx** is a Python library that converts ONNX models into PyTorch FX `GraphModule`s. This enables seamless integration with PyTorch's ecosystem for optimization, analysis, and deployment.

## Repository Structure

```
onnx2fx/
├── src/onnx2fx/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── converter.py       # Entry point (convert function)
│   ├── graph_builder.py   # FX graph construction logic
│   ├── op_registry.py     # Operator registration system
│   ├── exceptions.py      # Custom exception classes
│   ├── ops/               # ONNX operator implementations
│   │   ├── activation.py  # Activation functions (Relu, Sigmoid, etc.)
│   │   ├── arithmetic.py  # Arithmetic ops (Add, Mul, Sin, Cos, etc.)
│   │   ├── attention.py   # Attention ops (MultiHeadAttention, etc.)
│   │   ├── attention_msft.py # Microsoft domain attention ops
│   │   ├── control_flow.py # Control flow ops (Loop, If, Scan)
│   │   ├── convolution.py # Convolution ops (Conv, ConvTranspose, DeformConv)
│   │   ├── image.py       # Image ops (Resize, DepthToSpace, etc.)
│   │   ├── linalg.py      # Linear algebra ops (Einsum, Det)
│   │   ├── loss.py        # Loss functions (SoftmaxCrossEntropyLoss, etc.)
│   │   ├── nn.py          # Core neural network ops (MatMul, Gemm, Dropout)
│   │   ├── normalization.py # Normalization ops (BatchNorm, LayerNorm, etc.)
│   │   ├── pooling.py     # Pooling ops (MaxPool, AveragePool, etc.)
│   │   ├── quantization.py # Quantization ops (QLinear*, etc.)
│   │   ├── random.py      # Random ops (RandomNormal, Bernoulli, etc.)
│   │   ├── recurrent.py   # Recurrent neural networks (LSTM, GRU, RNN)
│   │   ├── reduction.py   # Reduction ops (Sum, Mean, etc.)
│   │   ├── sequence.py    # Sequence ops (SequenceConstruct, etc.)
│   │   ├── signal.py      # Signal processing (STFT, MelWeightMatrix, window functions, NMS)
│   │   ├── string.py      # String ops (StringNormalizer)
│   │   ├── tensor.py      # Tensor ops (Reshape, Transpose, etc.)
│   │   └── training.py    # Training ops (Gradient, Momentum, Adagrad)
│   └── utils/             # Utility modules
│       ├── analyze.py     # Model analysis utilities
│       ├── attributes.py  # ONNX attribute parsing
│       ├── dtype.py       # ONNX to PyTorch dtype mapping
│       ├── names.py       # Name sanitization utilities
│       ├── op_helpers.py  # Op helper utilities
│       └── training.py    # Training utilities (make_trainable)
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

## Development Setup

```bash
# Clone and install in development mode (using uv for faster dependency resolution)
git clone https://github.com/mshr-h/onnx2fx.git
cd onnx2fx
uv sync --dev

# Run Python scripts using uv
uv run python your_script.py
```

Core Requirements:
- Python >= 3.11
- PyTorch >= 2.9.0
- ONNX >= 1.19.1
- onnxscript >= 0.3.0

Development Tools:
- uv (recommended for development)
- See `pyproject.toml` for full list of development dependencies

## Running Tests

```bash
# Run all tests
uv run pytest

# Run all tests in parallel for faster execution
uv run pytest -n auto

# Run tests excluding slow tests
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/test_activation.py

# Run with verbose output
uv run pytest -v
```

## Code Style

This project uses `ruff` for linting and formatting. Run before committing:

```bash
uv run ruff check .
uv run ruff format .
```

## Adding New ONNX Operators

1. **Choose the appropriate file** in `src/onnx2fx/ops/` based on operator category.

2. **Register the operator** using the `@register` decorator:

```python
import torch
from ..op_registry import register
from ..utils.attributes import get_attribute

@register("NewOp")  # Standard ONNX domain
def new_op(builder, node):
    x = builder.get_value(node.input[0])
    alpha = get_attribute(node, "alpha", default=1.0)  # Get ONNX attribute
    return builder.call_function(torch.some_func, args=(x, alpha))
```

3. **For version-specific behavior**, use `since_version`:

```python
@register("SomeOp", since_version=1)
def some_op_v1(builder, node):
    # Behavior for opset 1-12
    ...

@register("SomeOp", since_version=13)
def some_op_v13(builder, node):
    # Behavior for opset 13+
    ...
```

4. **For custom domains** (e.g., Microsoft operators):

```python
@register("BiasGelu", domain="com.microsoft")
def bias_gelu(builder, node):
    ...
```

5. **Add tests** in the appropriate test file under `tests/`.

## Key APIs

### GraphBuilder Methods

- `builder.get_value(name)` - Get FX node by ONNX tensor name
- `builder.has_value(name)` - Check if value exists in environment
- `builder.call_function(func, args, kwargs)` - Create function call node
- `builder.call_module(module_name, args, kwargs)` - Create module call node
- `builder.add_submodule(name, module)` - Register a submodule (returns safe name)
- `builder.opset_version` - Get current opset version for default domain
- `builder.get_opset_version(domain)` - Get opset version for specific domain

### Attribute Utilities

For parsing ONNX node attributes, use functions from `onnx2fx.utils.attributes`:

- `get_attribute(node, name, default)` - Get a single attribute from an ONNX node
- `get_attributes(node)` - Get all attributes as a dictionary

### Public API

#### Core Functions
- `convert(model)` - Convert ONNX model to FX GraphModule
- `make_trainable(module)` - Convert buffers to trainable parameters for training

#### Model Analysis
- `analyze_model(model)` - Analyze ONNX model for operator support
- `AnalysisResult` - Dataclass with analysis results (supported_ops, unsupported_ops, etc.)

#### Operator Registration
- `register_op(op_type, handler=None, domain="", since_version=1)` - Register custom operator
- `unregister_op(op_type, domain="", since_version=None)` - Unregister an operator handler
- `is_supported(op_type, domain)` - Check if operator is supported
- `get_supported_ops(domain)` - List supported operators for a domain
- `get_all_supported_ops()` - Get all supported operators across all domains
- `get_registered_domains()` - Get list of registered domains

#### Exceptions
- `Onnx2FxError` - Base exception for all onnx2fx errors
- `UnsupportedOpError` - Raised when an operator is not supported
- `ConversionError` - Raised when conversion fails
- `ValueNotFoundError` - Raised when a value is not found in environment

## Testing Guidelines

- Use `onnxscript` to create test ONNX models programmatically
- Test across multiple opset versions using `conftest.py` fixtures
- Compare outputs with ONNX Runtime for numerical correctness
- Mark slow tests (e.g., large models) with `@pytest.mark.slow`

### Test Fixtures (from `conftest.py`)

- `OPSET_MODULES` - List of opset modules (opset 11-23) for parametrized tests
- `EINSUM_OPSET_MODULES` - Opset modules supporting Einsum (opset 12+)
- `DEFAULT_OPSET` - Default opset module (opset23)
- `opset_id(opset)` - Helper function for parametrize ids

Example parametrized test:
```python
import pytest
from conftest import OPSET_MODULES, opset_id

@pytest.mark.parametrize("opset", OPSET_MODULES, ids=opset_id)
def test_my_op(opset):
    # Test with each opset version
    ...
```

## License

Apache-2.0
