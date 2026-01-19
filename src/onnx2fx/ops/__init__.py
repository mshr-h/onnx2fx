# SPDX-License-Identifier: Apache-2.0
"""ONNX operator implementations.

This package contains all ONNX operator implementations organized by category:

- activation.py: Activation functions (Relu, Sigmoid, Softmax, etc.)
- arithmetic.py: Arithmetic and math ops (Add, Mul, Sin, Cos, etc.)
- attention.py: Attention mechanisms (standard ONNX domain)
- attention_msft.py: Attention mechanisms (com.microsoft domain)
- control_flow.py: Control flow ops (Loop, If, Scan)
- convolution.py: Convolution ops (Conv, ConvTranspose, DeformConv)
- image.py: Image processing ops (Resize, DepthToSpace, etc.)
- linalg.py: Linear algebra ops (Einsum, Det)
- loss.py: Loss functions (SoftmaxCrossEntropyLoss, etc.)
- nn.py: Core neural network ops (MatMul, Gemm, Dropout)
- normalization.py: Normalization ops (BatchNorm, LayerNorm, etc.)
- pooling.py: Pooling ops (MaxPool, AveragePool, etc.)
- quantization.py: Quantization ops (QLinearConv, etc.)
- random.py: Random number generation (RandomNormal, etc.)
- recurrent.py: Recurrent neural networks (LSTM, GRU, RNN)
- reduction.py: Reduction ops (ReduceSum, ReduceMean, etc.)
- sequence.py: Sequence ops (SequenceConstruct, etc.)
- signal.py: Signal processing (STFT, MelWeightMatrix, window functions, NMS)
- string.py: String ops (StringNormalizer)
- tensor.py: Tensor manipulation ops (Reshape, Transpose, etc.)
- training.py: Training ops (Gradient, Momentum, Adagrad)
"""

# Import all operator modules to register handlers
from . import activation
from . import arithmetic
from . import attention
from . import attention_msft
from . import control_flow
from . import convolution
from . import image
from . import linalg
from . import loss
from . import nn
from . import normalization
from . import pooling
from . import quantization
from . import random
from . import recurrent
from . import reduction
from . import sequence
from . import signal
from . import string
from . import tensor
from . import training

__all__ = [
    "activation",
    "arithmetic",
    "attention",
    "attention_msft",
    "control_flow",
    "convolution",
    "image",
    "linalg",
    "loss",
    "nn",
    "normalization",
    "pooling",
    "quantization",
    "random",
    "recurrent",
    "reduction",
    "sequence",
    "signal",
    "string",
    "tensor",
    "training",
]
