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
│   ├── ops/               # ONNX operator implementations
│   │   ├── activation.py  # Activation functions (Relu, Sigmoid, etc.)
│   │   ├── arithmetic.py  # Arithmetic ops (Add, Mul, etc.)
│   │   ├── tensor.py      # Tensor ops (Reshape, Transpose, etc.)
│   │   ├── nn.py          # Neural network ops (Conv, BatchNorm, etc.)
│   │   ├── reduction.py   # Reduction ops (Sum, Mean, etc.)
│   │   ├── attention.py   # Attention ops (MultiHeadAttention, etc.)
│   │   ├── control.py     # Control flow ops (If, Loop, etc.)
│   │   └── advanced.py    # Advanced ops (Einsum, etc.)
│   └── utils/             # Utility modules
│       ├── attributes.py  # ONNX attribute parsing
│       └── dtype.py       # ONNX to PyTorch dtype mapping
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/mshr-h/onnx2fx.git
cd onnx2fx
uv sync --dev
```

Requirements:
- Python >= 3.11
- PyTorch >= 2.9.0
- ONNX >= 1.19.1
- uv

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
from ..op_registry import register

@register("NewOp")  # Standard ONNX domain
def new_op(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.some_func, args=(x,))
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
- `builder.call_function(func, args, kwargs)` - Create function call node
- `builder.get_attribute(node, name, default)` - Get ONNX node attribute
- `builder.opset_version` - Get current opset version for default domain
- `builder.get_opset_version(domain)` - Get opset version for specific domain

### Public API

- `convert(model)` - Convert ONNX model to FX GraphModule
- `register_custom_op(op_type, handler, domain)` - Register custom operator
- `is_supported(op_type, domain)` - Check if operator is supported
- `get_supported_ops(domain)` - List supported operators for a domain

## Testing Guidelines

- Use `onnxscript` to create test ONNX models programmatically
- Test across multiple opset versions using `conftest.py` fixtures
- Compare outputs with ONNX Runtime for numerical correctness
- Mark slow tests (e.g., large models) with `@pytest.mark.slow`

## License

Apache-2.0
