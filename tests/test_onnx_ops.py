from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import torch

from onnx2fx.converter import convert


class TestAddOps:
    """Test models with different data types."""

    @script()
    def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Add(x, y)

    def test_float32_model(self):
        """Test float32 model conversion."""
        x = torch.randn(2, 4, dtype=torch.float32)
        y = torch.randn(2, 4, dtype=torch.float32)
        onnx_model = self.add_script.to_model_proto()

        fx_model = convert(onnx_model)

        with torch.no_grad():
            fx_output = fx_model(x, y)

        expected = x + y
        assert torch.allclose(fx_output, expected), "Float32 output mismatch!"
