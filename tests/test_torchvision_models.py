# SPDX-License-Identifier: Apache-2.0
"""Back-to-back tests for torchvision models.

These tests export torchvision models to ONNX, convert to FX,
and compare outputs with the original PyTorch model.
"""

import torch
import torch.nn as nn
import onnx
import pytest

from onnx2fx import convert

# Check if torchvision is available
try:
    import torchvision.models as models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def get_device():
    return torch.device("cpu")


def export_to_onnx(
    model: nn.Module, input_shape: tuple, opset_version: int = 23
) -> onnx.ModelProto:
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = (torch.randn(*input_shape),)
    return torch.onnx.export(
        model,
        dummy_input,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
    ).model_proto  # ty:ignore[possibly-missing-attribute]


def validate_model_output(model, input_shape=(1, 3, 224, 224), rtol=1e-3, atol=1e-4):
    """Helper to test a model."""
    device = get_device()

    onnx_model = export_to_onnx(model, input_shape)
    model = model.to(device)
    fx_module = convert(onnx_model).to(device)

    test_input = torch.randn(*input_shape, device=device)
    with torch.inference_mode():
        expected = model(test_input)
        result = fx_module(test_input)

    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
@pytest.mark.parametrize(
    "model_fn,weights",
    [
        (models.alexnet, None),
        pytest.param(
            models.convnext_tiny,
            None,
            marks=pytest.mark.skip(reason="Requires SequenceEmpty operator"),
        ),
        (models.densenet121, None),
        (models.efficientnet_b0, None),
        (models.efficientnet_v2_s, None),
        (models.googlenet, None),
        (models.inception_v3, (models.Inception_V3_Weights.DEFAULT)),
        (models.mnasnet0_5, None),
        (models.mobilenet_v2, None),
        (models.mobilenet_v3_small, None),
        (models.regnet_y_400mf, None),
        (models.resnet18, None),
        (models.resnext50_32x4d, None),
        (models.shufflenet_v2_x0_5, None),
        (models.shufflenet_v2_x1_0, None),
        (models.squeezenet1_0, None),
        (models.squeezenet1_1, None),
        (models.vgg11, None),
        (models.wide_resnet50_2, None),
        (models.maxvit_t, None),
        (models.swin_t, None),
        (models.swin_v2_b, None),
        (models.vit_b_16, None),
    ],
)
def test_torchvision_classifications(model_fn, weights):
    validate_model_output(model_fn(weights=weights).eval())


class SegmentationWrapper(torch.nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


class DetectionBackboneWrapper(torch.nn.Module):
    """Wrapper for detection models to test only the backbone.

    Detection models have data-dependent postprocessing (NMS) that cannot
    be exported with torch.export, so we test only the backbone.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, dict):
            return list(features.values())[-1]
        return features


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
@pytest.mark.parametrize(
    "model_fn",
    [
        models.segmentation.lraspp_mobilenet_v3_large,
        models.segmentation.fcn_resnet50,
    ],
)
def test_torchvision_segmentations(model_fn):
    wrapped_model = SegmentationWrapper(model_fn(weights=None).eval())
    validate_model_output(wrapped_model)


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
@pytest.mark.parametrize(
    "model_fn,input_shape",
    [
        (models.detection.fasterrcnn_resnet50_fpn, (1, 3, 224, 224)),
        (models.detection.ssd300_vgg16, (1, 3, 300, 300)),
        (models.detection.retinanet_resnet50_fpn, (1, 3, 224, 224)),
    ],
)
def test_torchvision_detection_backbones(model_fn, input_shape):
    wrapped_model = DetectionBackboneWrapper(model_fn(weights=None).eval())
    validate_model_output(wrapped_model, input_shape=input_shape)
