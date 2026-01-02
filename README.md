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

## Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.9.0
- ONNX >= 1.19.1

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
from onnx2fx import convert, register_custom_op

# Using decorator
@register_custom_op("MyCustomOp")
def my_custom_op(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.sigmoid, args=(x,))

# Or register directly
def my_handler(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.tanh, args=(x,))

register_custom_op("TanhCustom", my_handler)

# For custom domains (e.g., Microsoft operators)
@register_custom_op("BiasGelu", domain="com.microsoft")
def bias_gelu(builder, node):
    x = builder.get_value(node.input[0])
    bias = builder.get_value(node.input[1])
    return builder.call_function(
        lambda t, b: torch.nn.functional.gelu(t + b),
        args=(x, bias)
    )
```

### Multi-Opset Version Support

The library automatically selects the appropriate operator handler based on the model's opset version. For operators with version-specific behavior (e.g., `Softmax` changed default axis in opset 13), the correct implementation is used automatically:

```python
from onnx2fx import convert

# Models with different opset versions are handled automatically
fx_module_v11 = convert("model_opset11.onnx")  # Uses opset 11 semantics
fx_module_v17 = convert("model_opset17.onnx")  # Uses opset 17 semantics
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

## Supported Operators

### Standard ONNX Domain

#### Activation Functions
- Relu, LeakyRelu, PRelu, Elu, Selu, Celu
- Sigmoid, HardSigmoid, HardSwish, Tanh
- Softmax, LogSoftmax, Hardmax
- Softplus, Softsign
- Gelu, Silu, Mish
- ThresholdedRelu

#### Arithmetic & Element-wise
- Add, Sub, Mul, Div, Pow, Mod
- Neg, Abs, Sign, Ceil, Floor, Round
- Sqrt, Exp, Log, Reciprocal
- Min, Max, Mean, Sum
- Clip, Erf

#### Comparison & Logical
- Equal, Greater, Less, GreaterOrEqual, LessOrEqual
- And, Or, Not, Xor
- Where, IsNaN, IsInf
- BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift

#### Trigonometric
- Sin, Cos, Tan
- Sinh, Cosh, Tanh
- Asin, Acos, Atan
- Asinh, Acosh, Atanh

#### Reduction
- ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
- ReduceL1, ReduceL2
- ReduceLogSum, ReduceLogSumExp, ReduceSumSquare
- ArgMax, ArgMin

#### Tensor Manipulation
- Reshape, Transpose, Squeeze, Unsqueeze
- Concat, Split, Slice, Gather, GatherElements, GatherND
- ScatterElements, ScatterND
- Expand, Tile, Flatten
- Pad, Resize
- Shape, Size
- Cast, CastLike, Identity
- Constant, ConstantOfShape

#### Neural Network Layers
- Conv, ConvTranspose, ConvInteger
- MatMul, Gemm, MatMulInteger
- MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool
- BatchNormalization, InstanceNormalization, LayerNormalization, GroupNormalization
- Dropout, LRN

#### Quantization
- QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear
- QLinearConv, QLinearMatMul, QLinearAdd, QLinearMul
- QLinearSigmoid, QLinearLeakyRelu, QLinearGlobalAveragePool

#### Control Flow
- If, Loop, Scan

#### Sequence Operations
- SequenceConstruct, SequenceAt, SequenceEmpty
- SequenceInsert, SequenceErase, SequenceLength
- ConcatFromSequence, SplitToSequence

#### Other
- Einsum, TopK, NonZero, NonMaxSuppression
- OneHot, Range, EyeLike, Det
- Unique, Compress, Trilu
- DepthToSpace, SpaceToDepth
- ReverseSequence, CumSum
- Resize, StringNormalizer

#### Random & Sampling
- RandomNormal, RandomNormalLike
- RandomUniform, RandomUniformLike
- Multinomial, Bernoulli

#### Optional & Type Operations
- Optional, OptionalGetElement, OptionalHasElement
- Select

#### Loss Functions
- NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss

### Microsoft Domain (`com.microsoft`)

- Attention, GroupQueryAttention
- EmbedLayerNormalization
- SkipLayerNormalization, SkipSimplifiedLayerNormalization
- SimplifiedLayerNormalization

## API Reference

### `convert(model)`

Converts an ONNX model to a PyTorch FX `GraphModule`.

**Parameters:**
- `model` (`Union[onnx.ModelProto, str]`): Either an in-memory `onnx.ModelProto` or a file path to an ONNX model.

**Returns:**
- `torch.fx.GraphModule`: A PyTorch FX Graph module.

### `register_custom_op(op_type, handler=None, domain="")`

Register a custom ONNX operator handler.

**Parameters:**
- `op_type` (`str`): The ONNX operator type name.
- `handler` (`OpHandler`, optional): The handler function. If not provided, returns a decorator.
- `domain` (`str`, optional): The ONNX domain. Default is "" (standard ONNX domain).

### `unregister_op(op_type, domain="")`

Unregister an operator handler.

**Parameters:**
- `op_type` (`str`): The ONNX operator type name.
- `domain` (`str`, optional): The ONNX domain.

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
