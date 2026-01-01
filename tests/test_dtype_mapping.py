# SPDX-License-Identifier: Apache-2.0

import torch
import onnx

from onnx2fx.utils.dtype import DTYPE_MAP, onnx_dtype_to_torch, torch_dtype_to_onnx


class TestDtypeMapping:
    """Test data type mapping utilities."""

    def test_dtype_map_has_common_types(self):
        """DTYPE_MAP should contain common ONNX types."""
        assert onnx.TensorProto.FLOAT in DTYPE_MAP
        assert onnx.TensorProto.FLOAT16 in DTYPE_MAP
        assert onnx.TensorProto.DOUBLE in DTYPE_MAP
        assert onnx.TensorProto.INT32 in DTYPE_MAP
        assert onnx.TensorProto.INT64 in DTYPE_MAP
        assert onnx.TensorProto.BOOL in DTYPE_MAP

    def test_onnx_dtype_to_torch(self):
        """onnx_dtype_to_torch should return correct PyTorch types."""
        assert onnx_dtype_to_torch(onnx.TensorProto.FLOAT) == torch.float32
        assert onnx_dtype_to_torch(onnx.TensorProto.FLOAT16) == torch.float16
        assert onnx_dtype_to_torch(onnx.TensorProto.INT64) == torch.int64
        assert onnx_dtype_to_torch(onnx.TensorProto.BOOL) == torch.bool

    def test_onnx_dtype_to_torch_unknown(self):
        """onnx_dtype_to_torch should return None for unknown types."""
        assert onnx_dtype_to_torch(9999) is None

    def test_torch_dtype_to_onnx(self):
        """torch_dtype_to_onnx should return correct ONNX types."""
        assert torch_dtype_to_onnx(torch.float32) == onnx.TensorProto.FLOAT
        assert torch_dtype_to_onnx(torch.int64) == onnx.TensorProto.INT64
