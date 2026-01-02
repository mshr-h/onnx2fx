# SPDX-License-Identifier: Apache-2.0
"""Tests for trigonometric operators."""

import torch
from onnxscript import FLOAT, script
from onnxscript import opset23 as op

from onnx2fx import convert


class TestTrigOps:
    """Test trigonometric operators."""

    @script()
    def sin_script(x: FLOAT) -> FLOAT:
        return op.Sin(x)

    @script()
    def cos_script(x: FLOAT) -> FLOAT:
        return op.Cos(x)

    @script()
    def tan_script(x: FLOAT) -> FLOAT:
        return op.Tan(x)

    def test_sin(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.sin_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.sin(x))

    def test_cos(self):
        x = torch.randn(2, 4)
        fx_model = convert(self.cos_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.cos(x))

    def test_tan(self):
        x = torch.randn(2, 4) * 0.5  # Avoid large values
        fx_model = convert(self.tan_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(x)
        torch.testing.assert_close(result, torch.tan(x), atol=1e-5, rtol=1e-5)
