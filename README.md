# onnx2fx

Yet another ONNX to PyTorch FX converter.

> **⚠️ Note:** This project is under active development. The public API may change at any time.

`onnx2fx` converts ONNX models into PyTorch FX `GraphModule`s, enabling seamless integration with PyTorch's ecosystem for optimization, analysis, and deployment.

## Features

- **Simple API**: Convert ONNX models with a single function call
- **Extensive Operator Support**: 170+ ONNX operators including standard and Microsoft domain operators
- **Multi-Opset Version Support**: Automatic selection of version-specific operator handlers based on model opset
- **Custom Operator Registration**: Easily extend support for unsupported or custom ONNX operators
- **PyTorch FX Output**: Get a `torch.fx.GraphModule` for easy inspection, optimization, and compilation
- **Dynamic Shape Support**: Handle models with dynamic input dimensions
- **Quantization Support**: Support for quantized operators (QLinear*, DequantizeLinear, etc.)
- **Training Support**: Convert models to trainable modules with `make_trainable()` utility

## Tested Models

The following models have been tested and verified to work with onnx2fx:

- **PaddleOCRv5**: Text detection and recognition models (mobile and server variants)
  - PP-OCRv5_mobile_det, PP-OCRv5_mobile_rec
  - PP-OCRv5_server_det, PP-OCRv5_server_rec
- **TorchVision Models**: ResNet, VGG, MobileNet, etc. (via ONNX export)
- **LFM2**: Liquid Foundation Model (LFM2-350M-ENJP-MT)
- **LFM2.5**: Liquid Foundation Model 2.5
- **TinyLlama**: TinyLlama-1.1B-Chat

## Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.9.0
- ONNX >= 1.19.1
- onnxscript >= 0.3.0

### From Source

```bash
git clone https://github.com/mshr-h/onnx2fx.git
cd onnx2fx
pip install .
```

### Development Installation

```bash
git clone https://github.com/mshr-h/onnx2fx.git
cd onnx2fx
pip install -e ".[dev]"
```

## Quick Start

### Basic Conversion

```python
import torch
import onnx
from onnx2fx import convert

# Load from file path
fx_module = convert("model.onnx")

# Or from onnx.ModelProto
onnx_model = onnx.load("model.onnx")
fx_module = convert(onnx_model)

# Run inference
input_tensor = torch.randn(1, 3, 224, 224)
output = fx_module(input_tensor)
```

### Inspecting the Converted Graph

```python
from onnx2fx import convert

fx_module = convert("model.onnx")

# Print the FX graph
print(fx_module.graph)

# Get the graph code
print(fx_module.code)
```

### Registering Custom Operators

For unsupported or custom ONNX operators, you can register your own handlers:

```python
import torch
from onnx2fx import convert, register_op

# Using decorator
@register_op("MyCustomOp")
def my_custom_op(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sigmoid, args=(x,))

# Or register directly
def my_handler(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tanh, args=(x,))

register_op("TanhCustom", my_handler)

# For custom domains (e.g., Microsoft operators)
@register_op("BiasGelu", domain="com.microsoft")
def bias_gelu(builder, node):
    x = builder.get_value(node.input[0])
    bias = builder.get_value(node.input[1])
    return builder.call_function(
        lambda t, b: torch.nn.functional.gelu(t + b),
        args=(x, bias)
    )
```


> Note: `ai.onnx.ml` is treated as a distinct domain. If you register or query
> operators in that domain, pass `domain="ai.onnx.ml"` explicitly.

### Multi-Opset Version Support

The library automatically selects the appropriate operator handler based on the model's opset version. For operators with version-specific behavior (e.g., `Softmax` changed default axis in opset 13), the correct implementation is used automatically:

```python
from onnx2fx import convert

# Models with different opset versions are handled automatically
fx_module_v11 = convert("model_opset11.onnx")  # Uses opset 11 semantics
fx_module_v17 = convert("model_opset17.onnx")  # Uses opset 17 semantics
```

### Training Converted Models

By default, ONNX weights are loaded as non-trainable buffers. Use `make_trainable()` to enable training:

```python
import torch
from onnx2fx import convert, make_trainable

# Convert and make trainable
fx_module = convert("model.onnx")
fx_module = make_trainable(fx_module)  # Convert buffers to trainable parameters

# Now you can train the model
optimizer = torch.optim.Adam(fx_module.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = fx_module(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Querying Supported Operators

```python
from onnx2fx import (
    get_supported_ops,
    get_all_supported_ops,
    get_registered_domains,
    is_supported,
)

# Check if an operator is supported
print(is_supported("Conv"))  # True
print(is_supported("BiasGelu", domain="com.microsoft"))  # True

# Get all operators for a domain
standard_ops = get_supported_ops()  # Default ONNX domain
microsoft_ops = get_supported_ops("com.microsoft")

# Get all operators across all domains
all_ops = get_all_supported_ops()

# Get registered domains
domains = get_registered_domains()  # ['', 'com.microsoft']
```

### Analyzing Model Compatibility

Before converting, you can analyze a model to check operator support:

```python
from onnx2fx import analyze_model

# Analyze an ONNX model
result = analyze_model("model.onnx")

# Check results
print(f"Supported operators: {result.supported_ops}")
print(f"Unsupported operators: {result.unsupported_ops}")
print(f"Is fully supported: {result.is_fully_supported()}")

# Get detailed summary
print(result.summary())
```

### Exception Handling

Handle conversion errors gracefully:

```python
from onnx2fx import (
    convert,
    Onnx2FxError,
    UnsupportedOpError,
    ConversionError,
)

try:
    fx_module = convert("model.onnx")
except UnsupportedOpError as e:
    print(f"Unsupported operator: {e}")
except ConversionError as e:
    print(f"Conversion failed: {e}")
except Onnx2FxError as e:
    print(f"onnx2fx error: {e}")
```

## Supported Operators

### Standard ONNX Domain

This is a short list of representative operators. For the full list, call
`get_supported_ops()` or `get_all_supported_ops()`.

- **Core tensor & shape**: Reshape, Transpose, Concat, Split, Slice, Gather, Pad, Resize, Shape, Cast
- **Math & activations**: Add, Mul, MatMul, Gemm, Relu, Gelu, SiLU, Softmax, LogSoftmax
- **Normalization & pooling**: BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization, MaxPool, AveragePool, GlobalAveragePool
- **Reductions & indexing**: ReduceSum, ReduceMean, ArgMax, ArgMin, TopK
- **Control flow & sequence**: If, Loop, SequenceConstruct, SplitToSequence, ConcatFromSequence
- **Quantization**: QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul
- **Other**: Einsum, NonMaxSuppression, StringNormalizer

#### Attention & Normalization Extensions
- Attention (opset 24+)
- RotaryEmbedding (opset 23+)
- GroupQueryAttention
- EmbedLayerNormalization
- SkipLayerNormalization
- SimplifiedLayerNormalization
- SkipSimplifiedLayerNormalization

### Microsoft Domain (`com.microsoft`)

> Note: Some operators are available in both the standard and Microsoft domains (e.g., Attention, RotaryEmbedding, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, GroupQueryAttention, SkipLayerNormalization, EmbedLayerNormalization).

- Attention
- RotaryEmbedding
- SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization
- SkipLayerNormalization, EmbedLayerNormalization
- GroupQueryAttention

## API Reference

### `convert(model)`

Converts an ONNX model to a PyTorch FX `GraphModule`.

**Parameters:**
- `model` (`Union[onnx.ModelProto, str]`): Either an in-memory `onnx.ModelProto` or a file path to an ONNX model.

**Returns:**
- `torch.fx.GraphModule`: A PyTorch FX Graph module.

### `register_op(op_type, handler=None, domain="", since_version=1)`

Register a custom ONNX operator handler.

**Parameters:**
- `op_type` (`str`): The ONNX operator type name.
- `handler` (`OpHandler`, optional): The handler function. If not provided, returns a decorator.
- `domain` (`str`, optional): The ONNX domain. Default is "" (standard ONNX domain).
- `since_version` (`int`, optional): The minimum opset version for this handler. Default is 1.

### `unregister_op(op_type, domain="", since_version=None)`

Unregister an operator handler.

**Parameters:**
- `op_type` (`str`): The ONNX operator type name.
- `domain` (`str`, optional): The ONNX domain.
- `since_version` (`int`, optional): The specific opset handler to remove. If None, removes all versions.

**Returns:**
- `bool`: True if the operator was unregistered.

### `is_supported(op_type, domain="")`

Check if an operator is supported.

### `get_supported_ops(domain="")`

Get list of supported ONNX operators for a domain.

### `get_all_supported_ops()`

Get all supported operators across all domains.

### `get_registered_domains()`

Get list of registered domains.

### `analyze_model(model)`

Analyze an ONNX model for operator support.

**Parameters:**
- `model` (`Union[onnx.ModelProto, str]`): Either an in-memory `onnx.ModelProto` or a file path.

**Returns:**
- `AnalysisResult`: Analysis results with supported/unsupported operators.

### `AnalysisResult`

Dataclass containing model analysis results.

**Attributes:**
- `total_nodes` (`int`): Total number of nodes in the model graph.
- `unique_ops` (`Set[Tuple[str, str]]`): Set of unique (op_type, domain) tuples.
- `supported_ops` (`List[Tuple[str, str]]`): List of supported (op_type, domain) tuples.
- `unsupported_ops` (`List[Tuple[str, str, int]]`): List of unsupported (op_type, domain, opset_version) tuples.
- `opset_versions` (`Dict[str, int]`): Mapping of domain to opset version.
- `op_counts` (`Dict[Tuple[str, str], int]`): Count of each (op_type, domain) in the model.

**Methods:**
- `is_fully_supported()`: Returns `True` if all operators are supported.
- `summary()`: Returns a human-readable summary string.

### Exceptions

- `Onnx2FxError`: Base exception for all onnx2fx errors.
- `UnsupportedOpError`: Raised when an operator is not supported.
- `ConversionError`: Raised when conversion fails.
- `ValueNotFoundError`: Raised when a value is not found in the environment.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run all tests in parallel for faster execution
pytest -n auto

# Run specific test file
pytest tests/test_activation.py

# Skip slow tests
pytest -m "not slow"
```

### Code Formatting

```bash
# Format code with ruff
ruff format .

# Check linting
ruff check .
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

Masahiro Hiramori (contact@mshr-h.com)
