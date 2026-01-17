# SPDX-License-Identifier: Apache-2.0
"""Tests for control flow operators (Loop, If)."""

import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from onnx2fx import convert


class TestLoopOp:
    """Test Loop operator."""

    def test_loop_simple_sum(self):
        """Test simple loop that sums values from 0 to n-1."""
        # Create the loop body graph
        # Body inputs: iteration_num (i), condition_in, sum_in
        # Body outputs: condition_out, sum_out
        iter_num = helper.make_tensor_value_info("i", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [])

        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [])

        # sum_out = sum_in + cast(i, float)
        cast_node = helper.make_node("Cast", ["i"], ["i_float"], to=TensorProto.FLOAT)
        add_node = helper.make_node("Add", ["sum_in", "i_float"], ["sum_out"])
        # cond_out = cond_in (always true, loop terminates by trip count)
        identity_node = helper.make_node("Identity", ["cond_in"], ["cond_out"])

        body_graph = helper.make_graph(
            [cast_node, add_node, identity_node],
            "loop_body",
            [iter_num, cond_in, sum_in],
            [cond_out, sum_out],
        )

        # Create the main graph with Loop node
        max_trip_count = helper.make_tensor_value_info(
            "max_trip_count", TensorProto.INT64, []
        )
        initial_cond = helper.make_tensor_value_info(
            "initial_cond", TensorProto.BOOL, []
        )
        initial_sum = helper.make_tensor_value_info("initial_sum", TensorProto.FLOAT, [])
        final_sum = helper.make_tensor_value_info("final_sum", TensorProto.FLOAT, [])

        loop_node = helper.make_node(
            "Loop",
            ["max_trip_count", "initial_cond", "initial_sum"],
            ["final_sum"],
            body=body_graph,
        )

        graph = helper.make_graph(
            [loop_node],
            "main_graph",
            [max_trip_count, initial_cond, initial_sum],
            [final_sum],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model = onnx.shape_inference.infer_shapes(model)

        # Convert and test
        fx_module = convert(model)

        # Loop 5 times: sum = 0 + 1 + 2 + 3 + 4 = 10
        max_count = torch.tensor(5, dtype=torch.int64)
        cond = torch.tensor(True, dtype=torch.bool)
        init_sum = torch.tensor(0.0, dtype=torch.float32)

        with torch.inference_mode():
            result = fx_module(max_count, cond, init_sum)

        # Loop returns a tuple, first element is the final loop-carried value
        if isinstance(result, tuple):
            result = result[0]

        expected = torch.tensor(10.0, dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_loop_with_condition(self):
        """Test loop that terminates based on condition."""
        # Loop body that doubles value until it exceeds threshold
        iter_num = helper.make_tensor_value_info("i", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        val_in = helper.make_tensor_value_info("val_in", TensorProto.FLOAT, [])

        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        val_out = helper.make_tensor_value_info("val_out", TensorProto.FLOAT, [])

        # val_out = val_in * 2
        two_tensor = numpy_helper.from_array(
            np.array(2.0, dtype=np.float32), name="two"
        )
        mul_node = helper.make_node("Mul", ["val_in", "two"], ["val_out"])

        # cond_out = val_out < 100 (continue while less than 100)
        threshold_tensor = numpy_helper.from_array(
            np.array(100.0, dtype=np.float32), name="threshold"
        )
        less_node = helper.make_node("Less", ["val_out", "threshold"], ["cond_out"])

        body_graph = helper.make_graph(
            [mul_node, less_node],
            "loop_body",
            [iter_num, cond_in, val_in],
            [cond_out, val_out],
            [two_tensor, threshold_tensor],
        )

        # Main graph
        max_trip_count = helper.make_tensor_value_info(
            "max_trip_count", TensorProto.INT64, []
        )
        initial_cond = helper.make_tensor_value_info(
            "initial_cond", TensorProto.BOOL, []
        )
        initial_val = helper.make_tensor_value_info("initial_val", TensorProto.FLOAT, [])
        final_val = helper.make_tensor_value_info("final_val", TensorProto.FLOAT, [])

        loop_node = helper.make_node(
            "Loop",
            ["max_trip_count", "initial_cond", "initial_val"],
            ["final_val"],
            body=body_graph,
        )

        graph = helper.make_graph(
            [loop_node],
            "main_graph",
            [max_trip_count, initial_cond, initial_val],
            [final_val],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model = onnx.shape_inference.infer_shapes(model)

        fx_module = convert(model)

        # Start with 1.0, double each iteration until >= 100
        # 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 (stop, 128 >= 100)
        max_count = torch.tensor(100, dtype=torch.int64)  # High limit
        cond = torch.tensor(True, dtype=torch.bool)
        init_val = torch.tensor(1.0, dtype=torch.float32)

        with torch.inference_mode():
            result = fx_module(max_count, cond, init_val)

        # Loop returns a tuple, first element is the final loop-carried value
        if isinstance(result, tuple):
            result = result[0]

        expected = torch.tensor(128.0, dtype=torch.float32)
        torch.testing.assert_close(result, expected)


class TestIfOp:
    """Test If operator."""

    def test_if_simple(self):
        """Test simple if-else that returns different values based on condition."""
        # Then branch: return x * 2 (x is from outer scope)
        then_output = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2, 3])
        two_tensor = numpy_helper.from_array(
            np.array(2.0, dtype=np.float32), name="two"
        )
        then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
        then_graph = helper.make_graph(
            [then_mul],
            "then_branch",
            [],  # No inputs - uses outer scope
            [then_output],
            [two_tensor],
        )

        # Else branch: return x * 3
        else_output = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2, 3])
        three_tensor = numpy_helper.from_array(
            np.array(3.0, dtype=np.float32), name="three"
        )
        else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
        else_graph = helper.make_graph(
            [else_mul],
            "else_branch",
            [],  # No inputs - uses outer scope
            [else_output],
            [three_tensor],
        )

        # Main graph
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        if_node = helper.make_node(
            "If",
            ["cond"],
            ["output"],
            then_branch=then_graph,
            else_branch=else_graph,
        )

        graph = helper.make_graph(
            [if_node],
            "main_graph",
            [cond, x],
            [output],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        fx_module = convert(model)

        x_val = torch.ones(2, 3, dtype=torch.float32)

        # Test with condition = True (then branch: * 2)
        with torch.inference_mode():
            result_true = fx_module(torch.tensor(True), x_val)
        torch.testing.assert_close(result_true, x_val * 2)

        # Test with condition = False (else branch: * 3)
        with torch.inference_mode():
            result_false = fx_module(torch.tensor(False), x_val)
        torch.testing.assert_close(result_false, x_val * 3)

    def test_if_with_outer_scope(self):
        """Test if with references to outer scope values."""
        # Then branch: return outer_val + 10
        then_output = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [])
        ten_tensor = numpy_helper.from_array(
            np.array(10.0, dtype=np.float32), name="ten"
        )
        # Reference outer_val from main graph
        then_add = helper.make_node("Add", ["outer_val", "ten"], ["then_out"])
        then_graph = helper.make_graph(
            [then_add],
            "then_branch",
            [],  # No direct inputs, uses outer scope
            [then_output],
            [ten_tensor],
        )

        # Else branch: return outer_val - 10
        else_output = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [])
        else_sub = helper.make_node("Sub", ["outer_val", "ten"], ["else_out"])
        else_graph = helper.make_graph(
            [else_sub],
            "else_branch",
            [],
            [else_output],
            [ten_tensor],
        )

        # Main graph
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        outer_val = helper.make_tensor_value_info("outer_val", TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])

        if_node = helper.make_node(
            "If",
            ["cond"],
            ["output"],
            then_branch=then_graph,
            else_branch=else_graph,
        )

        graph = helper.make_graph(
            [if_node],
            "main_graph",
            [cond, outer_val],
            [output],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        fx_module = convert(model)

        val = torch.tensor(50.0, dtype=torch.float32)

        # Test with condition = True (then branch: + 10)
        with torch.inference_mode():
            result_true = fx_module(torch.tensor(True), val)
        torch.testing.assert_close(result_true, torch.tensor(60.0))

        # Test with condition = False (else branch: - 10)
        with torch.inference_mode():
            result_false = fx_module(torch.tensor(False), val)
        torch.testing.assert_close(result_false, torch.tensor(40.0))


class TestNestedControlFlow:
    """Test nested control flow operators."""

    def test_loop_with_if(self):
        """Test loop containing an if statement."""
        # Loop body that adds 1 if i is even, subtracts 1 if i is odd
        iter_num = helper.make_tensor_value_info("i", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        val_in = helper.make_tensor_value_info("val_in", TensorProto.FLOAT, [])

        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        val_out = helper.make_tensor_value_info("val_out", TensorProto.FLOAT, [])

        # Check if i is even: i % 2 == 0
        two_i64 = numpy_helper.from_array(np.array(2, dtype=np.int64), name="two_i64")
        zero_i64 = numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero_i64")
        mod_node = helper.make_node("Mod", ["i", "two_i64"], ["i_mod_2"])
        equal_node = helper.make_node("Equal", ["i_mod_2", "zero_i64"], ["is_even"])

        # Then branch (even): val + 1
        then_output = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [])
        one_tensor = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one")
        then_add = helper.make_node("Add", ["val_in", "one"], ["then_out"])
        then_graph = helper.make_graph(
            [then_add],
            "then_branch",
            [],
            [then_output],
            [one_tensor],
        )

        # Else branch (odd): val - 1
        else_output = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [])
        else_sub = helper.make_node("Sub", ["val_in", "one"], ["else_out"])
        else_graph = helper.make_graph(
            [else_sub],
            "else_branch",
            [],
            [else_output],
            [one_tensor],
        )

        # If node in loop body
        if_node = helper.make_node(
            "If",
            ["is_even"],
            ["val_out"],
            then_branch=then_graph,
            else_branch=else_graph,
        )

        # cond_out = cond_in
        identity_node = helper.make_node("Identity", ["cond_in"], ["cond_out"])

        body_graph = helper.make_graph(
            [mod_node, equal_node, if_node, identity_node],
            "loop_body",
            [iter_num, cond_in, val_in],
            [cond_out, val_out],
            [two_i64, zero_i64],
        )

        # Main graph
        max_trip_count = helper.make_tensor_value_info(
            "max_trip_count", TensorProto.INT64, []
        )
        initial_cond = helper.make_tensor_value_info(
            "initial_cond", TensorProto.BOOL, []
        )
        initial_val = helper.make_tensor_value_info("initial_val", TensorProto.FLOAT, [])
        final_val = helper.make_tensor_value_info("final_val", TensorProto.FLOAT, [])

        loop_node = helper.make_node(
            "Loop",
            ["max_trip_count", "initial_cond", "initial_val"],
            ["final_val"],
            body=body_graph,
        )

        graph = helper.make_graph(
            [loop_node],
            "main_graph",
            [max_trip_count, initial_cond, initial_val],
            [final_val],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        fx_module = convert(model)

        # Loop 6 times starting from 0:
        # i=0 (even): 0 + 1 = 1
        # i=1 (odd): 1 - 1 = 0
        # i=2 (even): 0 + 1 = 1
        # i=3 (odd): 1 - 1 = 0
        # i=4 (even): 0 + 1 = 1
        # i=5 (odd): 1 - 1 = 0
        max_count = torch.tensor(6, dtype=torch.int64)
        cond = torch.tensor(True, dtype=torch.bool)
        init_val = torch.tensor(0.0, dtype=torch.float32)

        with torch.inference_mode():
            result = fx_module(max_count, cond, init_val)

        # Loop returns a tuple, first element is the final loop-carried value
        if isinstance(result, tuple):
            result = result[0]

        expected = torch.tensor(0.0, dtype=torch.float32)
        torch.testing.assert_close(result, expected)
