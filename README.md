# onnx2fx

_onnx2fx_ converts ONNX graphs into `torch.fx.GraphModule` objects, enabling PyTorch-native optimization, analysis, and execution of ONNX models.

## Features

- **140+ ONNX operators** supported across arithmetic, activation, tensor, reduction, neural network, and control flow categories
- **Dynamic shape support** for batch size, sequence length, and image dimensions
- **Custom operator registration** for domain-specific ops (e.g., `com.microsoft`)
- **ONNX metadata preservation** in `node.meta["onnx_*"]` for downstream passes
- **Initializers as buffers** for proper parameter management

## Installation

```bash
git clone https://github.com/mshr-h/onnx2fx
cd onnx2fx
uv pip install -e .              # or: pip install -e .
```

Requirements: Python 3.11+, PyTorch 2.9+, ONNX 1.19+, ONNXScript 0.5.6+ (see `pyproject.toml`).

## Quick Start

```python
import torch
import onnx
import onnx2fx

# Load an ONNX model (file path or in-memory ModelProto)
model_proto = onnx.load("model.onnx")

# Convert to FX
fx_module = onnx2fx.convert(model_proto)
fx_module.graph.print_tabular()

# Run inference just like any torch.nn.Module
x = torch.randn(1, 3, 224, 224)
with torch.inference_mode():
    y = fx_module(x)
```

## Custom Operators

Register custom operator handlers for unsupported or domain-specific operators:

```python
from onnx2fx import convert, register_custom_op, unregister_op

# Using decorator
@register_custom_op("MyCustomOp")
def my_custom_op(builder, node):
    x = builder.get_value(node.input[0])
    return builder.call_function(torch.relu, args=(x,))

# For custom domains (e.g., ONNX Runtime extensions)
@register_custom_op("BiasGelu", domain="com.microsoft")
def bias_gelu(builder, node):
    x = builder.get_value(node.input[0])
    bias = builder.get_value(node.input[1])
    return builder.call_function(
        lambda t, b: torch.nn.functional.gelu(t + b),
        args=(x, bias)
    )

# Query supported operators
from onnx2fx import get_supported_ops, is_supported
print(get_supported_ops())  # List all supported ops
print(is_supported("Conv"))  # Check specific op
```

## Supported Operators

<details>
<summary>Click to expand full operator list (140+ operators)</summary>

### Arithmetic
Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs, Neg, Sign, Floor, Ceil, Round, Mod, Reciprocal, Min, Max, Sum, Mean, MatMul, MatMulInteger, Gemm, Einsum

### Activation
Relu, LeakyRelu, PRelu, Elu, Selu, Celu, Gelu, Sigmoid, Tanh, Softmax, LogSoftmax, Softplus, Softsign, HardSigmoid, HardSwish, Mish, ThresholdedRelu, Shrink

### Tensor
Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Concat, Split, Slice, Gather, GatherElements, GatherND, Scatter, ScatterElements, ScatterND, Tile, Expand, Pad, Shape, Size, ConstantOfShape, Identity, Cast, CastLike, Constant, Where, NonZero, Compress, ReverseSequence, Trilu, Unique

### Reduction
ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare, ArgMax, ArgMin

### Neural Network
Conv, ConvTranspose, MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool, BatchNormalization, InstanceNormalization, LayerNormalization, GroupNormalization, LRN, Dropout, Upsample, Resize, MaxRoiPool

### Advanced
Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Erf, Range, CumSum, TopK, Det, Hardmax, NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss

### Comparison & Logic
Equal, Less, LessOrEqual, Greater, GreaterOrEqual, Not, And, Or, Xor, IsNaN, IsInf, BitShift

### Control Flow
If, Loop, Scan, Optional, OptionalHasElement, OptionalGetElement, SequenceConstruct, SequenceAt, SequenceLength, SequenceEmpty, SequenceInsert, SequenceErase, ConcatFromSequence, SplitToSequence

### Random
RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Bernoulli, Multinomial

### Quantization
QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul

</details>

## Dynamic Shapes

Models with dynamic dimensions (batch size, sequence length, image size) are fully supported:

```python
# Export PyTorch model with dynamic axes
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={"input": {0: "batch", 1: "seq_len"}}
)

# Convert and run with varying input sizes
fx_module = onnx2fx.convert(onnx.load("model.onnx"))
fx_module(torch.randn(1, 10, 64))   # batch=1, seq=10
fx_module(torch.randn(8, 100, 64))  # batch=8, seq=100
```

## Testing

```bash
uv run pytest -v
```

## Development

Install pre-commit hooks to automatically format code with ruff on commit:

```bash
uv run pre-commit install
```

## Architecture

```
src/onnx2fx/
├── __init__.py          # Public API
├── converter.py         # Main convert() function
├── graph_builder.py     # FX graph construction
├── op_registry.py       # Operator registration system
├── ops/                 # Operator implementations
│   ├── arithmetic.py
│   ├── activation.py
│   ├── tensor.py
│   ├── reduction.py
│   ├── nn.py
│   ├── advanced.py
│   ├── attention.py
│   └── control.py
└── utils/
    ├── dtype.py         # ONNX-PyTorch dtype mapping
    └── attributes.py    # ONNX attribute parsing
```

## License

Apache-2.0
