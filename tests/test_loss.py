# SPDX-License-Identifier: Apache-2.0
"""Tests for loss function operators."""

import torch
from onnxscript import FLOAT, INT64, script
from onnxscript import opset23 as op

from onnx2fx import convert


class TestLossOps:
    """Test loss function operators."""

    @script()
    def softmax_cross_entropy_loss_script(scores: FLOAT, labels: INT64) -> FLOAT:
        loss, _ = op.SoftmaxCrossEntropyLoss(scores, labels, reduction="mean")
        return loss

    @script()
    def softmax_cross_entropy_loss_none_script(scores: FLOAT, labels: INT64) -> FLOAT:
        loss, _ = op.SoftmaxCrossEntropyLoss(scores, labels, reduction="none")
        return loss

    @script()
    def softmax_cross_entropy_loss_sum_script(scores: FLOAT, labels: INT64) -> FLOAT:
        loss, _ = op.SoftmaxCrossEntropyLoss(scores, labels, reduction="sum")
        return loss

    @script()
    def nll_loss_script(input: FLOAT, target: INT64) -> FLOAT:
        (loss,) = op.NegativeLogLikelihoodLoss(input, target, reduction="mean")
        return loss

    @script()
    def nll_loss_none_script(input: FLOAT, target: INT64) -> FLOAT:
        (loss,) = op.NegativeLogLikelihoodLoss(input, target, reduction="none")
        return loss

    def test_softmax_cross_entropy_loss_mean(self):
        # Scores: (batch, num_classes), Labels: (batch,)
        scores = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,), dtype=torch.int64)

        fx_model = convert(self.softmax_cross_entropy_loss_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(scores, labels)

        expected = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        torch.testing.assert_close(result, expected)

    def test_softmax_cross_entropy_loss_none(self):
        scores = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,), dtype=torch.int64)

        fx_model = convert(self.softmax_cross_entropy_loss_none_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(scores, labels)

        expected = torch.nn.functional.cross_entropy(scores, labels, reduction="none")
        torch.testing.assert_close(result, expected)

    def test_softmax_cross_entropy_loss_sum(self):
        scores = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,), dtype=torch.int64)

        fx_model = convert(self.softmax_cross_entropy_loss_sum_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(scores, labels)

        expected = torch.nn.functional.cross_entropy(scores, labels, reduction="sum")
        torch.testing.assert_close(result, expected)

    def test_nll_loss_mean(self):
        # NLL expects log-probabilities as input
        input_tensor = torch.randn(4, 5).log_softmax(dim=1)
        target = torch.randint(0, 5, (4,), dtype=torch.int64)

        fx_model = convert(self.nll_loss_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(input_tensor, target)

        expected = torch.nn.functional.nll_loss(input_tensor, target, reduction="mean")
        torch.testing.assert_close(result, expected)

    def test_nll_loss_none(self):
        input_tensor = torch.randn(4, 5).log_softmax(dim=1)
        target = torch.randint(0, 5, (4,), dtype=torch.int64)

        fx_model = convert(self.nll_loss_none_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(input_tensor, target)

        expected = torch.nn.functional.nll_loss(input_tensor, target, reduction="none")
        torch.testing.assert_close(result, expected)


class TestLossOpsWithIgnoreIndex:
    """Test loss operators with ignore_index."""

    @script()
    def cross_entropy_ignore_script(scores: FLOAT, labels: INT64) -> FLOAT:
        loss, _ = op.SoftmaxCrossEntropyLoss(
            scores, labels, reduction="mean", ignore_index=-100
        )
        return loss

    def test_cross_entropy_with_ignore_index(self):
        scores = torch.randn(4, 5)
        labels = torch.tensor([0, 1, -100, 3], dtype=torch.int64)  # -100 is ignored

        fx_model = convert(self.cross_entropy_ignore_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(scores, labels)

        expected = torch.nn.functional.cross_entropy(
            scores, labels, reduction="mean", ignore_index=-100
        )
        torch.testing.assert_close(result, expected)
