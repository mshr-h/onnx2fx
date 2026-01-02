# SPDX-License-Identifier: Apache-2.0
"""Tests for multiple ONNX opset version support."""

import onnx
import pytest
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper
from onnxscript import FLOAT, INT64, script
from onnxscript import opset11, opset13, opset14, opset15
from onnxscript import opset16, opset17, opset18, opset19, opset20
from onnxscript import opset21, opset22, opset23

from onnx2fx import convert
from onnx2fx.op_registry import get_handler, get_handler_versions

from conftest import OPSET_MODULES, EINSUM_OPSET_MODULES


class TestOpsetVersionRegistry:
    """Test opset version-aware registry functionality."""

    def test_get_handler_versions(self):
        """Test that version-specific handlers are registered correctly."""
        # Softmax should have handlers for opset 1 and 13
        versions = get_handler_versions("Softmax")
        assert 1 in versions
        assert 13 in versions

    def test_get_handler_selects_correct_version(self):
        """Test that get_handler returns the correct version-specific handler."""
        # For opset 11, should get the opset 1 handler (since 1 <= 11 < 13)
        handler_v11 = get_handler("Softmax", "", 11)
        handler_v13 = get_handler("Softmax", "", 13)
        handler_v23 = get_handler("Softmax", "", 23)

        # Handlers for opset 11 and 12 should be the same (opset 1 handler)
        assert handler_v11 is get_handler("Softmax", "", 12)
        # Handlers for opset 13+ should be the opset 13 handler
        assert handler_v13 is handler_v23

    def test_squeeze_handler_versions(self):
        """Test Squeeze handler versions."""
        versions = get_handler_versions("Squeeze")
        assert 1 in versions
        assert 13 in versions


class TestSoftmaxOpsets:
    """Tests for Softmax opset version differences.

    Opset < 13: default axis=1, coerced 2D softmax
    Opset 13+: default axis=-1, direct softmax
    """

    @script(default_opset=opset11)
    def softmax_v11_default(x: FLOAT) -> FLOAT:
        return opset11.Softmax(x)

    @script(default_opset=opset11)
    def softmax_v11_axis2(x: FLOAT) -> FLOAT:
        return opset11.Softmax(x, axis=2)

    @script(default_opset=opset13)
    def softmax_v13_default(x: FLOAT) -> FLOAT:
        return opset13.Softmax(x)

    @script(default_opset=opset13)
    def softmax_v13_axis1(x: FLOAT) -> FLOAT:
        return opset13.Softmax(x, axis=1)

    @script(default_opset=opset23)
    def softmax_v23_default(x: FLOAT) -> FLOAT:
        return opset23.Softmax(x)

    def test_softmax_opset11_default_axis(self):
        """Opset 11: default axis=1, coerced 2D softmax."""
        model = self.softmax_v11_default.to_model_proto()
        assert model.opset_import[0].version == 11

        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)

        # Opset < 13: coerce to 2D at axis=1, apply softmax, reshape back
        x_2d = x.reshape(2, -1)  # [2, 12]
        expected = F.softmax(x_2d, dim=1).reshape(2, 3, 4)
        torch.testing.assert_close(result, expected)

    def test_softmax_opset11_explicit_axis(self):
        """Opset 11: explicit axis=2."""
        model = self.softmax_v11_axis2.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)

        # Coerce to 2D at axis=2
        x_2d = x.reshape(6, 4)  # [2*3, 4]
        expected = F.softmax(x_2d, dim=1).reshape(2, 3, 4)
        torch.testing.assert_close(result, expected)

    def test_softmax_opset13_default_axis(self):
        """Opset 13+: default axis=-1, direct softmax."""
        model = self.softmax_v13_default.to_model_proto()
        assert model.opset_import[0].version == 13

        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)

        # Opset 13+: direct softmax on last dimension
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected)

    def test_softmax_opset13_explicit_axis(self):
        """Opset 13+: explicit axis=1."""
        model = self.softmax_v13_axis1.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)

        expected = F.softmax(x, dim=1)
        torch.testing.assert_close(result, expected)

    def test_softmax_opset23(self):
        """Opset 23: same behavior as opset 13."""
        model = self.softmax_v23_default.to_model_proto()
        assert model.opset_import[0].version == 23

        fx_model = convert(model)
        x = torch.randn(2, 3, 4)
        result = fx_model(x)

        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected)


class TestSqueezeOpsets:
    """Tests for Squeeze opset version differences.

    Opset < 13: axes is an attribute
    Opset 13+: axes is an optional input
    """

    @script(default_opset=opset11)
    def squeeze_v11_attr(x: FLOAT) -> FLOAT:
        return opset11.Squeeze(x, axes=[1])

    @script(default_opset=opset13)
    def squeeze_v13_input(x: FLOAT, axes: INT64) -> FLOAT:
        return opset13.Squeeze(x, axes)

    @script(default_opset=opset13)
    def squeeze_v13_no_axes(x: FLOAT) -> FLOAT:
        return opset13.Squeeze(x)

    @script(default_opset=opset23)
    def squeeze_v23_input(x: FLOAT, axes: INT64) -> FLOAT:
        return opset23.Squeeze(x, axes)

    def test_squeeze_opset11_attribute(self):
        """Opset 11: axes as attribute."""
        model = self.squeeze_v11_attr.to_model_proto()
        assert model.opset_import[0].version == 11

        fx_model = convert(model)
        x = torch.randn(2, 1, 4)
        result = fx_model(x)

        assert result.shape == (2, 4)
        torch.testing.assert_close(result, x.squeeze(1))

    def test_squeeze_opset13_input(self):
        """Opset 13+: axes as input."""
        model = self.squeeze_v13_input.to_model_proto()
        assert model.opset_import[0].version == 13

        fx_model = convert(model)
        x = torch.randn(2, 1, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = fx_model(x, axes)

        assert result.shape == (2, 4)
        torch.testing.assert_close(result, x.squeeze(1))

    def test_squeeze_opset13_no_axes(self):
        """Opset 13+: no axes input - squeeze all."""
        model = self.squeeze_v13_no_axes.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 1, 1, 4)
        result = fx_model(x)

        assert result.shape == (2, 4)
        torch.testing.assert_close(result, x.squeeze())

    def test_squeeze_opset23(self):
        """Opset 23: same behavior as opset 13."""
        model = self.squeeze_v23_input.to_model_proto()
        assert model.opset_import[0].version == 23

        fx_model = convert(model)
        x = torch.randn(2, 1, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = fx_model(x, axes)

        assert result.shape == (2, 4)


class TestUnsqueezeOpsets:
    """Tests for Unsqueeze opset version differences.

    Opset < 13: axes is an attribute
    Opset 13+: axes is a required input
    """

    @script(default_opset=opset11)
    def unsqueeze_v11_attr(x: FLOAT) -> FLOAT:
        return opset11.Unsqueeze(x, axes=[1])

    @script(default_opset=opset13)
    def unsqueeze_v13_input(x: FLOAT, axes: INT64) -> FLOAT:
        return opset13.Unsqueeze(x, axes)

    @script(default_opset=opset23)
    def unsqueeze_v23_input(x: FLOAT, axes: INT64) -> FLOAT:
        return opset23.Unsqueeze(x, axes)

    def test_unsqueeze_opset11_attribute(self):
        """Opset 11: axes as attribute."""
        model = self.unsqueeze_v11_attr.to_model_proto()
        assert model.opset_import[0].version == 11

        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)

        assert result.shape == (2, 1, 4)
        torch.testing.assert_close(result, x.unsqueeze(1))

    def test_unsqueeze_opset13_input(self):
        """Opset 13+: axes as input."""
        model = self.unsqueeze_v13_input.to_model_proto()
        assert model.opset_import[0].version == 13

        fx_model = convert(model)
        x = torch.randn(2, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = fx_model(x, axes)

        assert result.shape == (2, 1, 4)
        torch.testing.assert_close(result, x.unsqueeze(1))

    def test_unsqueeze_opset23(self):
        """Opset 23: same behavior as opset 13."""
        model = self.unsqueeze_v23_input.to_model_proto()
        assert model.opset_import[0].version == 23

        fx_model = convert(model)
        x = torch.randn(2, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = fx_model(x, axes)

        assert result.shape == (2, 1, 4)


class TestSplitOpsets:
    """Tests for Split opset version differences.

    Opset < 13: split sizes is an optional attribute
    Opset 13+: split sizes is an optional input
    Opset 18+: added num_outputs attribute

    Note: Split with multiple outputs is complex to test with onnxscript,
    so we use onnx.helper for these tests.
    """

    def test_split_opset11_attribute(self):
        """Opset 11: split as attribute."""
        from onnx import helper, TensorProto

        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        y1_info = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [2, 2])
        y2_info = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 2])

        split_node = helper.make_node(
            "Split", ["X"], ["Y1", "Y2"], axis=1, split=[2, 2]
        )
        graph = helper.make_graph([split_node], "test", [x_info], [y1_info, y2_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        fx_model = convert(model)
        x = torch.randn(2, 4)
        y1, y2 = fx_model(x)

        assert y1.shape == (2, 2)
        assert y2.shape == (2, 2)
        torch.testing.assert_close(torch.cat([y1, y2], dim=1), x)

    def test_split_opset13_input(self):
        """Opset 13+: split as input."""
        from onnx import helper, TensorProto

        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        split_info = helper.make_tensor_value_info("split", TensorProto.INT64, [2])
        y1_info = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [2, 2])
        y2_info = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 2])

        split_node = helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=1)
        graph = helper.make_graph(
            [split_node], "test", [x_info, split_info], [y1_info, y2_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        fx_model = convert(model)
        x = torch.randn(2, 4)
        split = torch.tensor([2, 2], dtype=torch.int64)
        y1, y2 = fx_model(x, split)

        assert y1.shape == (2, 2)
        assert y2.shape == (2, 2)
        torch.testing.assert_close(torch.cat([y1, y2], dim=1), x)

    def test_split_opset23(self):
        """Opset 23: same behavior as opset 13."""
        from onnx import helper, TensorProto

        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        split_info = helper.make_tensor_value_info("split", TensorProto.INT64, [2])
        y1_info = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [2, 2])
        y2_info = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 2])

        split_node = helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=1)
        graph = helper.make_graph(
            [split_node], "test", [x_info, split_info], [y1_info, y2_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])

        fx_model = convert(model)
        x = torch.randn(2, 4)
        split = torch.tensor([2, 2], dtype=torch.int64)
        y1, y2 = fx_model(x, split)

        assert y1.shape == (2, 2)
        assert y2.shape == (2, 2)


class TestReluAllOpsets:
    """Test Relu works identically across all opsets."""

    @pytest.mark.parametrize("op", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_relu_all_opsets(self, op):
        """Relu should work identically across all opsets."""

        @script(default_opset=op)
        def relu_script(x: FLOAT) -> FLOAT:
            return op.Relu(x)

        model = relu_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        result = fx_model(x)
        expected = torch.relu(x)
        torch.testing.assert_close(result, expected)


class TestAddAllOpsets:
    """Test Add works identically across all opsets."""

    @pytest.mark.parametrize("op", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_add_all_opsets(self, op):
        """Add should work identically across all opsets."""

        @script(default_opset=op)
        def add_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return op.Add(x, y)

        model = add_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        result = fx_model(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)


class TestMatMulAllOpsets:
    """Test MatMul works identically across all opsets."""

    @pytest.mark.parametrize("op", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_matmul_all_opsets(self, op):
        """MatMul should work identically across all opsets."""

        @script(default_opset=op)
        def matmul_script(x: FLOAT, y: FLOAT) -> FLOAT:
            return op.MatMul(x, y)

        model = matmul_script.to_model_proto()
        fx_model = convert(model)
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        result = fx_model(x, y)
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected)


class TestEinsumAllOpsets:
    """Test Einsum works identically across all supporting opsets (12+)."""

    @pytest.mark.parametrize(
        "opset", EINSUM_OPSET_MODULES, ids=lambda x: f"opset{x.version}"
    )
    def test_einsum_matmul_all_opsets(self, opset):
        """Einsum for matrix multiplication across opsets."""
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 4])
        z_info = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [2, 4])

        einsum_node = onnx.helper.make_node(
            "Einsum", ["X", "Y"], ["Z"], equation="ij,jk->ik"
        )

        graph = onnx.helper.make_graph(
            [einsum_node], "test", [x_info, y_info], [z_info]
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", opset.version)]
        )

        fx_model = convert(model)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        with torch.inference_mode():
            result = fx_model(x, y)

        expected = torch.einsum("ij,jk->ik", x, y)
        torch.testing.assert_close(result, expected)


class TestGatherAllOpsets:
    """Test Gather works identically across all opsets."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_gather_all_opsets(self, opset):
        """Gather should work across all opsets."""

        @script(default_opset=opset)
        def gather_script(data: FLOAT, indices: INT64) -> FLOAT:
            return opset.Gather(data, indices, axis=0)

        model = gather_script.to_model_proto()
        fx_model = convert(model)

        data = torch.randn(5, 4)
        indices = torch.tensor([0, 2, 4], dtype=torch.int64)
        result = fx_model(data, indices)
        expected = data[indices]
        torch.testing.assert_close(result, expected)


class TestWhereAllOpsets:
    """Test Where works identically across all opsets."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_where_all_opsets(self, opset):
        """Where should work across all opsets."""
        cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [2, 2])
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
        z_info = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 2])

        where_node = helper.make_node("Where", ["cond", "X", "Y"], ["Z"])

        graph = helper.make_graph(
            [where_node], "test", [cond_info, x_info, y_info], [z_info]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_model = convert(model)

        cond = torch.tensor([[True, False], [False, True]])
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        result = fx_model(cond, x, y)
        expected = torch.where(cond, x, y)
        torch.testing.assert_close(result, expected)


class TestHardmaxAllOpsets:
    """Test Hardmax works identically across all opsets."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_hardmax_all_opsets(self, opset):
        """Hardmax should work across all opsets (11+)."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])

        hardmax_node = helper.make_node("Hardmax", ["X"], ["Y"], axis=-1)

        graph = helper.make_graph([hardmax_node], "test", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_module = convert(model)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

        result = fx_module(x)
        torch.testing.assert_close(result, expected)


class TestLogSoftmaxAllOpsets:
    """Test LogSoftmax across supporting opsets."""

    @pytest.mark.parametrize(
        "opset",
        [
            opset13,
            opset14,
            opset15,
            opset16,
            opset17,
            opset18,
            opset19,
            opset20,
            opset21,
            opset22,
            opset23,
        ],
        ids=lambda x: f"opset{x.version}",
    )
    def test_log_softmax_all_opsets(self, opset):
        """LogSoftmax should work across opsets 13+ (axis semantics changed)."""
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])

        node = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=-1)

        graph = helper.make_graph([node], "test", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_module = convert(model)

        x = torch.randn(2, 3, 4)
        expected = F.log_softmax(x, dim=-1)

        result = fx_module(x)
        torch.testing.assert_close(result, expected)


class TestGatherNDAllOpsets:
    """Test GatherND works identically across all opsets."""

    @pytest.mark.parametrize("opset", OPSET_MODULES, ids=lambda x: f"opset{x.version}")
    def test_gather_nd_all_opsets(self, opset):
        """GatherND should work across all opsets (11+)."""
        data_info = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2])
        indices_info = helper.make_tensor_value_info(
            "indices", TensorProto.INT64, [2, 2]
        )
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        gather_node = helper.make_node("GatherND", ["data", "indices"], ["output"])

        graph = helper.make_graph(
            [gather_node], "test", [data_info, indices_info], [output_info]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset.version)]
        )

        fx_module = convert(model)

        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)

        result = fx_module(data, indices)
        expected = torch.tensor([0.0, 3.0])
        torch.testing.assert_close(result, expected)
