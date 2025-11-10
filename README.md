# onnx2fx

_onnx2fx_ is an experiment that turns ONNX graphs into `torch.fx.GraphModule` objects.

## Installation

```bash
git clone https://github.com/mshr-h/onnx2fx
cd onnx2fx
uv pip install -e .              # or: pip install -e .
```

Requirements: Python 3.11+, PyTorch 2.9+, ONNX 1.19+, ONNXScript 0.5.6+ (see `pyproject.toml`).

## Quick start

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
x = torch.randn(1, 4)
with torch.no_grad():
    y = fx_module(x)
```

The converter stores ONNX metadata under `node.meta["onnx_*"]`, which can be consumed by passes that need original shapes, dtypes, or node names.

## Testing

```bash
uv run pytest -v
```

## Roadmap / ideas

- Implement core arithmetic, activation, and linear-algebra ops.
- Handle multiple-output nodes (Sequences, Dicts, tuples).
- Preserve initializers/parameters as registered buffers instead of implicit constants.
- Provide higher-level passes (e.g., graph clean-up, module partitioning) after conversion.
