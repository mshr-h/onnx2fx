# SPDX-License-Identifier: Apache-2.0
"""Tests for custom operator support."""

from onnx import helper, TensorProto
import torch

from onnx2fx import (
    convert,
    register_custom_op,
    unregister_op,
    get_supported_ops,
    get_all_supported_ops,
    get_registered_domains,
    is_supported,
)


def make_simple_onnx_model(
    op_type: str,
    domain: str = "",
    input_shapes: list = [[2, 3]],
    output_shape: list = [2, 3],
    attributes: dict = {},
):
    """Create a simple ONNX model with one operator."""
    inputs = []
    input_names = []
    for i, shape in enumerate(input_shapes):
        name = f"input_{i}"
        inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        input_names.append(name)

    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    node = helper.make_node(
        op_type,
        inputs=input_names,
        outputs=["output"],
        domain=domain,
        **(attributes),
    )

    graph = helper.make_graph([node], "test_graph", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    if domain:
        model.opset_import.append(helper.make_opsetid(domain, 1))

    return model


class TestCustomOpRegistration:
    """Test custom operator registration."""

    def test_register_custom_op_decorator(self):
        """Test registering custom op using decorator."""

        @register_custom_op("TestCustomRelu")
        def test_custom_relu(builder, node):
            x = builder.get_value(node.input[0])
            return builder.call_function(torch.relu, args=(x,))

        try:
            assert is_supported("TestCustomRelu")

            # Create and convert model
            model = make_simple_onnx_model("TestCustomRelu")
            fx_module = convert(model)

            # Test
            test_input = torch.randn(2, 3)
            result = fx_module(test_input)
            expected = torch.relu(test_input)
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("TestCustomRelu")

    def test_register_custom_op_function(self):
        """Test registering custom op using function call."""

        def custom_sigmoid(builder, node):
            x = builder.get_value(node.input[0])
            return builder.call_function(torch.sigmoid, args=(x,))

        register_custom_op("TestCustomSigmoid", custom_sigmoid)

        try:
            assert is_supported("TestCustomSigmoid")

            model = make_simple_onnx_model("TestCustomSigmoid")
            fx_module = convert(model)

            test_input = torch.randn(2, 3)
            result = fx_module(test_input)
            expected = torch.sigmoid(test_input)
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("TestCustomSigmoid")

    def test_register_custom_domain_op(self):
        """Test registering custom op with custom domain."""

        @register_custom_op("BiasAdd", domain="com.test")
        def bias_add(builder, node):
            x = builder.get_value(node.input[0])
            bias = builder.get_value(node.input[1])
            return builder.call_function(torch.add, args=(x, bias))

        try:
            assert is_supported("BiasAdd", domain="com.test")
            assert not is_supported("BiasAdd")  # Not in default domain

            model = make_simple_onnx_model(
                "BiasAdd",
                domain="com.test",
                input_shapes=[[2, 3], [3]],
                output_shape=[2, 3],
            )
            fx_module = convert(model)

            x = torch.randn(2, 3)
            bias = torch.randn(3)
            result = fx_module(x, bias)
            expected = x + bias
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("BiasAdd", domain="com.test")

    def test_unregister_op(self):
        """Test unregistering an operator."""

        @register_custom_op("TestTempOp")
        def temp_op(builder, node):
            x = builder.get_value(node.input[0])
            return builder.call_function(torch.neg, args=(x,))

        assert is_supported("TestTempOp")
        assert unregister_op("TestTempOp")
        assert not is_supported("TestTempOp")

        # Unregistering non-existent op returns False
        assert not unregister_op("NonExistentOp")

    def test_override_builtin_op(self):
        """Test overriding a built-in operator."""
        original_ops = get_supported_ops()
        assert "Relu" in original_ops

        # Override Relu with a custom implementation
        @register_custom_op("Relu")
        def custom_relu(builder, node):
            x = builder.get_value(node.input[0])

            # Custom: apply relu then scale by 2
            def scaled_relu(t):
                return torch.relu(t) * 2

            return builder.call_function(scaled_relu, args=(x,))

        try:
            model = make_simple_onnx_model("Relu")
            fx_module = convert(model)

            test_input = torch.randn(2, 3)
            result = fx_module(test_input)
            expected = torch.relu(test_input) * 2
            torch.testing.assert_close(result, expected)
        finally:
            # Restore original Relu (re-import ops to reset)

            # Force re-registration by calling the decorator again
            from onnx2fx.op_registry import register

            @register("Relu")
            def relu(builder, node):
                x = builder.get_value(node.input[0])
                return builder.call_function(torch.relu, args=(x,))


class TestMultiInputCustomOp:
    """Test custom operators with multiple inputs."""

    def test_custom_binary_op(self):
        """Test custom op with two inputs."""

        @register_custom_op("WeightedAdd")
        def weighted_add(builder, node):
            x = builder.get_value(node.input[0])
            y = builder.get_value(node.input[1])

            def _weighted_add(a, b):
                return 0.7 * a + 0.3 * b

            return builder.call_function(_weighted_add, args=(x, y))

        try:
            model = make_simple_onnx_model(
                "WeightedAdd",
                input_shapes=[[2, 3], [2, 3]],
                output_shape=[2, 3],
            )
            fx_module = convert(model)

            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            result = fx_module(x, y)
            expected = 0.7 * x + 0.3 * y
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("WeightedAdd")

    def test_custom_op_with_attributes(self):
        """Test custom op that reads attributes."""
        from onnx2fx.utils.attributes import get_attribute

        @register_custom_op("ScaleOp")
        def scale_op(builder, node):
            x = builder.get_value(node.input[0])
            scale = get_attribute(node, "scale", 1.0)

            return builder.call_function(lambda t, s: t * s, args=(x, scale))

        try:
            # Create model with attribute
            inputs = [
                helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [2, 3])
            ]
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

            node = helper.make_node(
                "ScaleOp",
                inputs=["input_0"],
                outputs=["output"],
                scale=2.5,
            )

            graph = helper.make_graph([node], "test_graph", inputs, [output])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)]
            )

            fx_module = convert(model)

            test_input = torch.randn(2, 3)
            result = fx_module(test_input)
            expected = test_input * 2.5
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("ScaleOp")


class TestRegistryQueries:
    """Test registry query functions."""

    def test_get_supported_ops(self):
        """Test getting list of supported ops."""
        ops = get_supported_ops()
        assert isinstance(ops, list)
        assert "Add" in ops
        assert "Relu" in ops
        assert "Conv" in ops

    def test_get_all_supported_ops(self):
        """Test getting all ops across domains."""
        all_ops = get_all_supported_ops()
        assert isinstance(all_ops, dict)
        assert "" in all_ops  # Default domain
        assert "Add" in all_ops[""]

    def test_get_registered_domains(self):
        """Test getting registered domains."""
        domains = get_registered_domains()
        assert isinstance(domains, list)
        assert "" in domains  # Default domain always exists

    def test_is_supported(self):
        """Test checking if op is supported."""
        assert is_supported("Add")
        assert is_supported("Relu")
        assert not is_supported("NonExistentOp")
        assert not is_supported("Add", domain="non.existent.domain")


class TestMicrosoftDomainOps:
    """Test Microsoft ONNX Runtime custom ops."""

    def test_bias_gelu(self):
        """Test BiasGelu from com.microsoft domain."""

        @register_custom_op("BiasGelu", domain="com.microsoft")
        def bias_gelu(builder, node):
            x = builder.get_value(node.input[0])
            bias = builder.get_value(node.input[1])

            def _bias_gelu(a, b):
                return torch.nn.functional.gelu(a + b)

            return builder.call_function(_bias_gelu, args=(x, bias))

        try:
            model = make_simple_onnx_model(
                "BiasGelu",
                domain="com.microsoft",
                input_shapes=[[2, 64], [64]],
                output_shape=[2, 64],
            )
            fx_module = convert(model)

            x = torch.randn(2, 64)
            bias = torch.randn(64)
            result = fx_module(x, bias)
            expected = torch.nn.functional.gelu(x + bias)
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("BiasGelu", domain="com.microsoft")

    def test_fused_matmul(self):
        """Test FusedMatMul from com.microsoft domain."""

        @register_custom_op("FusedMatMul", domain="com.microsoft")
        def fused_matmul(builder, node):
            from onnx2fx.utils.attributes import get_attribute

            a = builder.get_value(node.input[0])
            b = builder.get_value(node.input[1])
            alpha = get_attribute(node, "alpha", 1.0)
            trans_a = get_attribute(node, "transA", 0)
            trans_b = get_attribute(node, "transB", 0)

            def _fused_matmul(x, y, alpha, trans_a, trans_b):
                if trans_a:
                    x = x.transpose(-2, -1)
                if trans_b:
                    y = y.transpose(-2, -1)
                return alpha * torch.matmul(x, y)

            return builder.call_function(
                _fused_matmul, args=(a, b, alpha, trans_a, trans_b)
            )

        try:
            # Create model with attributes
            inputs = [
                helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [3, 4]),
            ]
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])

            node = helper.make_node(
                "FusedMatMul",
                inputs=["input_0", "input_1"],
                outputs=["output"],
                domain="com.microsoft",
                alpha=2.0,
                transA=0,
                transB=0,
            )

            graph = helper.make_graph([node], "test_graph", inputs, [output])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)]
            )
            model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

            fx_module = convert(model)

            a = torch.randn(2, 3)
            b = torch.randn(3, 4)
            result = fx_module(a, b)
            expected = 2.0 * torch.matmul(a, b)
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("FusedMatMul", domain="com.microsoft")


class TestCustomOpIntegration:
    """Integration tests for custom ops in complex models."""

    def test_custom_op_in_sequence(self):
        """Test custom op used in a sequence with built-in ops."""

        @register_custom_op("DoubleScale")
        def double_scale(builder, node):
            x = builder.get_value(node.input[0])
            return builder.call_function(lambda t: t * 2, args=(x,))

        try:
            # Create model: Relu -> DoubleScale -> Sigmoid
            inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
            output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])

            nodes = [
                helper.make_node("Relu", ["x"], ["relu_out"]),
                helper.make_node("DoubleScale", ["relu_out"], ["scale_out"]),
                helper.make_node("Sigmoid", ["scale_out"], ["out"]),
            ]

            graph = helper.make_graph(nodes, "test_graph", inputs, [output])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)]
            )

            fx_module = convert(model)

            test_input = torch.randn(2, 3)
            result = fx_module(test_input)
            expected = torch.sigmoid(torch.relu(test_input) * 2)
            torch.testing.assert_close(result, expected)
        finally:
            unregister_op("DoubleScale")
