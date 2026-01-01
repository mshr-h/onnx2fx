from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import torch

from onnx2fx.converter import convert


@script()
def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
    add = op.Add(x, y)
    return add


def test_add_operation():
    example_input1 = torch.randn(2, 4)
    example_input2 = torch.randn(2, 4)
    onnx_model = add_script.to_model_proto()

    # eager mode evaluation
    eager_output = add_script(example_input1.numpy(), example_input2.numpy())

    fx_model = convert(onnx_model)

    with torch.no_grad():
        fx_output = fx_model(example_input1, example_input2)

    assert torch.allclose(torch.from_numpy(eager_output), fx_output), "❌Outputs do not match!"
