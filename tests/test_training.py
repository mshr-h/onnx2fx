# SPDX-License-Identifier: Apache-2.0
"""Tests for training converted ONNX models.

This module tests that converted FX modules can:
1. Compute gradients for inputs
2. Compute gradients for parameters (after make_trainable)
3. Update parameters with optimizers
4. Handle train/eval modes correctly
"""

import io

import onnx
import pytest
import torch
import torch.nn as nn

from onnx2fx import convert, make_trainable


def export_to_onnx(
    model: nn.Module, input_shape: tuple, opset_version: int = 23
) -> onnx.ModelProto:
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    buffer.seek(0)
    return onnx.load(buffer)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLPWithBatchNorm(nn.Module):
    """MLP with BatchNorm for train/eval mode testing."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TestInputGradients:
    """Test gradient flow through inputs."""

    def test_input_gradient_simple_mlp(self):
        """Test input gradients match between original and converted model."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)
        fx_module.eval()

        # Use same input for both models
        torch.manual_seed(42)
        test_input_orig = torch.randn(1, 32, requires_grad=True)
        test_input_fx = test_input_orig.clone().detach().requires_grad_(True)

        # Forward and backward on original model
        output_orig = model(test_input_orig)
        loss_orig = output_orig.sum()
        loss_orig.backward()

        # Forward and backward on converted model
        output_fx = fx_module(test_input_fx)
        loss_fx = output_fx.sum()
        loss_fx.backward()

        # Compare gradients
        assert test_input_orig.grad is not None
        assert test_input_fx.grad is not None
        torch.testing.assert_close(
            test_input_fx.grad, test_input_orig.grad, rtol=1e-3, atol=1e-4
        )

    def test_input_gradient_batch(self):
        """Test input gradients with batch size > 1."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (4, 32))
        fx_module = convert(onnx_model)
        fx_module.eval()

        torch.manual_seed(42)
        test_input_orig = torch.randn(4, 32, requires_grad=True)
        test_input_fx = test_input_orig.clone().detach().requires_grad_(True)

        # Forward and backward on original model
        output_orig = model(test_input_orig)
        loss_orig = output_orig.mean()
        loss_orig.backward()

        # Forward and backward on converted model
        output_fx = fx_module(test_input_fx)
        loss_fx = output_fx.mean()
        loss_fx.backward()

        # Compare gradients
        torch.testing.assert_close(
            test_input_fx.grad, test_input_orig.grad, rtol=1e-3, atol=1e-4
        )


class TestMakeTrainable:
    """Test make_trainable utility."""

    def test_make_trainable_converts_buffers(self):
        """Test that make_trainable converts buffers to parameters."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)

        # Before: should have buffers, no parameters
        num_buffers_before = len(list(fx_module.buffers()))
        num_params_before = len(list(fx_module.parameters()))
        assert num_buffers_before > 0
        assert num_params_before == 0

        # Apply make_trainable
        fx_module = make_trainable(fx_module)

        # After: should have parameters, no buffers
        num_buffers_after = len(list(fx_module.buffers()))
        num_params_after = len(list(fx_module.parameters()))
        assert num_buffers_after == 0
        assert num_params_after == num_buffers_before

    def test_make_trainable_preserves_values(self):
        """Test that make_trainable preserves tensor values."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)

        # Store buffer values before conversion
        buffer_values = {name: buf.clone() for name, buf in fx_module.named_buffers()}

        fx_module = make_trainable(fx_module)

        # Check parameter values match original buffer values
        for name, param in fx_module.named_parameters():
            assert name in buffer_values
            torch.testing.assert_close(param.data, buffer_values[name])


class TestParameterGradients:
    """Test gradient computation for parameters."""

    def test_parameter_gradients_match(self):
        """Test parameter gradients match between original and converted model."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)
        fx_module = make_trainable(fx_module)
        fx_module.eval()

        # Use same input
        torch.manual_seed(42)
        test_input = torch.randn(1, 32)

        # Forward and backward on original model
        model.zero_grad()
        output_orig = model(test_input)
        loss_orig = output_orig.sum()
        loss_orig.backward()

        # Forward and backward on converted model
        fx_module.zero_grad()
        output_fx = fx_module(test_input)
        loss_fx = output_fx.sum()
        loss_fx.backward()

        # Build mapping from original param names to FX param names
        # ONNX export uses names like "fc1.weight", FX converts "." to "_"
        orig_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Convert name: fc1.weight -> fc1_weight
                fx_name = name.replace(".", "_")
                orig_grads[fx_name] = param.grad.clone()

        # Compare gradients
        for name, param in fx_module.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            if name in orig_grads:
                torch.testing.assert_close(
                    param.grad, orig_grads[name], rtol=1e-3, atol=1e-4
                )


class TestOptimizerUpdate:
    """Test optimizer updates."""

    def test_optimizer_step_changes_parameters(self):
        """Test that optimizer.step() actually updates parameters."""
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)
        fx_module = make_trainable(fx_module)

        # Store initial parameter values
        initial_params = {
            name: param.clone() for name, param in fx_module.named_parameters()
        }

        # Training step
        optimizer = torch.optim.SGD(fx_module.parameters(), lr=0.1)
        optimizer.zero_grad()

        test_input = torch.randn(1, 32)
        output = fx_module(test_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        for name, param in fx_module.named_parameters():
            assert not torch.allclose(param, initial_params[name]), (
                f"Parameter {name} did not change"
            )

    def test_optimizer_update_matches_original(self):
        """Test that optimizer updates produce same results as original model."""
        # Create two identical models with same initialization
        torch.manual_seed(42)
        model = SimpleMLP()
        model.eval()

        onnx_model = export_to_onnx(model, (1, 32))
        fx_module = convert(onnx_model)
        fx_module = make_trainable(fx_module)

        # Same learning rate and optimizer settings
        lr = 0.01
        optimizer_orig = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer_fx = torch.optim.SGD(fx_module.parameters(), lr=lr)

        # Same input
        torch.manual_seed(123)
        test_input = torch.randn(1, 32)

        # Training step on original
        optimizer_orig.zero_grad()
        output_orig = model(test_input)
        loss_orig = output_orig.sum()
        loss_orig.backward()
        optimizer_orig.step()

        # Training step on converted
        optimizer_fx.zero_grad()
        output_fx = fx_module(test_input)
        loss_fx = output_fx.sum()
        loss_fx.backward()
        optimizer_fx.step()

        # Compare updated parameters
        orig_params = dict(model.named_parameters())
        for name, param in fx_module.named_parameters():
            orig_name = name.replace("_", ".")
            if orig_name in orig_params:
                torch.testing.assert_close(
                    param, orig_params[orig_name], rtol=1e-3, atol=1e-4
                )


class TestTrainEvalModes:
    """Test train/eval mode behavior."""

    def test_batchnorm_eval_mode(self):
        """Test BatchNorm in eval mode produces same output."""
        model = MLPWithBatchNorm()
        model.eval()

        onnx_model = export_to_onnx(model, (4, 32))
        fx_module = convert(onnx_model)
        fx_module.eval()

        torch.manual_seed(42)
        test_input = torch.randn(4, 32)

        with torch.inference_mode():
            output_orig = model(test_input)
            output_fx = fx_module(test_input)

        torch.testing.assert_close(output_fx, output_orig, rtol=1e-4, atol=1e-5)

    def test_gradient_flow_with_batchnorm(self):
        """Test gradient flow through BatchNorm layer."""
        model = MLPWithBatchNorm()
        model.eval()

        onnx_model = export_to_onnx(model, (4, 32))
        fx_module = convert(onnx_model)
        fx_module = make_trainable(fx_module)
        fx_module.eval()

        torch.manual_seed(42)
        test_input_orig = torch.randn(4, 32, requires_grad=True)
        test_input_fx = test_input_orig.clone().detach().requires_grad_(True)

        # Forward and backward on original
        output_orig = model(test_input_orig)
        loss_orig = output_orig.sum()
        loss_orig.backward()

        # Forward and backward on converted
        output_fx = fx_module(test_input_fx)
        loss_fx = output_fx.sum()
        loss_fx.backward()

        # Compare input gradients
        torch.testing.assert_close(
            test_input_fx.grad, test_input_orig.grad, rtol=1e-3, atol=1e-4
        )


@pytest.mark.slow
class TestSlowTrainingModels:
    """Slow tests with larger models."""

    def test_resnet18_input_gradients(self):
        """Test input gradients on ResNet18."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not installed")

        model = resnet18(weights=None)
        model.eval()

        input_shape = (1, 3, 224, 224)
        onnx_model = export_to_onnx(model, input_shape)
        fx_module = convert(onnx_model)
        fx_module.eval()

        torch.manual_seed(42)
        test_input_orig = torch.randn(*input_shape, requires_grad=True)
        test_input_fx = test_input_orig.clone().detach().requires_grad_(True)

        # Forward and backward on original
        output_orig = model(test_input_orig)
        loss_orig = output_orig.sum()
        loss_orig.backward()

        # Forward and backward on converted
        output_fx = fx_module(test_input_fx)
        loss_fx = output_fx.sum()
        loss_fx.backward()

        # Compare gradients (allow more tolerance for deep model)
        torch.testing.assert_close(
            test_input_fx.grad, test_input_orig.grad, rtol=1e-2, atol=1e-3
        )

    def test_resnet18_training_step(self):
        """Test full training step on ResNet18."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not installed")

        torch.manual_seed(42)
        model = resnet18(weights=None)
        model.eval()

        input_shape = (2, 3, 224, 224)
        onnx_model = export_to_onnx(model, input_shape)
        fx_module = convert(onnx_model)
        fx_module = make_trainable(fx_module)

        # Verify we can do a training step
        optimizer = torch.optim.SGD(fx_module.parameters(), lr=0.01)

        initial_params = {
            name: param.clone() for name, param in fx_module.named_parameters()
        }

        optimizer.zero_grad()
        test_input = torch.randn(*input_shape)
        output = fx_module(test_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Verify at least some parameters changed
        changed_count = 0
        for name, param in fx_module.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed_count += 1

        assert changed_count > 0, "No parameters were updated"
