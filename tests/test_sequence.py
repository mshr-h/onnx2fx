# SPDX-License-Identifier: Apache-2.0
"""Tests for sequence operators."""

import torch
from onnxscript import FLOAT, INT64, script
from onnxscript import opset23 as op

from conftest import run_onnx_test
from onnx2fx import convert


class TestSequenceOps:
    """Test sequence operators."""

    @script()
    def sequence_construct_script(a: FLOAT, b: FLOAT, c: FLOAT) -> FLOAT:
        seq = op.SequenceConstruct(a, b, c)
        return op.ConcatFromSequence(seq, axis=0)

    @script()
    def sequence_at_script(a: FLOAT, b: FLOAT, c: FLOAT, idx: INT64) -> FLOAT:
        seq = op.SequenceConstruct(a, b, c)
        return op.SequenceAt(seq, idx)

    @script()
    def sequence_length_script(a: FLOAT, b: FLOAT, c: FLOAT) -> INT64:
        seq = op.SequenceConstruct(a, b, c)
        return op.SequenceLength(seq)

    @script()
    def sequence_empty_script() -> INT64:
        seq = op.SequenceEmpty()
        return op.SequenceLength(seq)

    @script()
    def sequence_insert_script(a: FLOAT, b: FLOAT, new_elem: FLOAT) -> FLOAT:
        seq = op.SequenceConstruct(a, b)
        # Insert at end (no position specified - append)
        seq2 = op.SequenceInsert(seq, new_elem)
        return op.ConcatFromSequence(seq2, axis=0)

    @script()
    def sequence_insert_at_pos_script(
        a: FLOAT, b: FLOAT, new_elem: FLOAT, pos: INT64
    ) -> FLOAT:
        seq = op.SequenceConstruct(a, b)
        seq2 = op.SequenceInsert(seq, new_elem, pos)
        return op.ConcatFromSequence(seq2, axis=0)

    @script()
    def sequence_erase_script(a: FLOAT, b: FLOAT, c: FLOAT) -> FLOAT:
        seq = op.SequenceConstruct(a, b, c)
        # Erase last element (no position specified)
        seq2 = op.SequenceErase(seq)
        return op.ConcatFromSequence(seq2, axis=0)

    @script()
    def sequence_erase_at_pos_script(a: FLOAT, b: FLOAT, c: FLOAT, pos: INT64) -> FLOAT:
        seq = op.SequenceConstruct(a, b, c)
        seq2 = op.SequenceErase(seq, pos)
        return op.ConcatFromSequence(seq2, axis=0)

    @script()
    def concat_from_sequence_stack_script(a: FLOAT, b: FLOAT, c: FLOAT) -> FLOAT:
        seq = op.SequenceConstruct(a, b, c)
        return op.ConcatFromSequence(seq, axis=0, new_axis=1)

    @script()
    def split_to_sequence_script(x: FLOAT) -> FLOAT:
        seq = op.SplitToSequence(x, axis=0)
        return op.ConcatFromSequence(seq, axis=0)

    def test_sequence_construct(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        expected = torch.cat([a, b, c], dim=0)
        run_onnx_test(
            self.sequence_construct_script.to_model_proto, (a, b, c), expected
        )

    def test_sequence_at(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        idx = torch.tensor(1, dtype=torch.int64)
        run_onnx_test(self.sequence_at_script.to_model_proto, (a, b, c, idx), b)

    def test_sequence_length(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        fx_model = convert(self.sequence_length_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model(a, b, c)
        assert result.item() == 3

    def test_sequence_empty(self):
        fx_model = convert(self.sequence_empty_script.to_model_proto())
        with torch.inference_mode():
            result = fx_model()
        assert result.item() == 0

    def test_sequence_insert(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        new_elem = torch.randn(2, 3)
        expected = torch.cat([a, b, new_elem], dim=0)
        run_onnx_test(
            self.sequence_insert_script.to_model_proto, (a, b, new_elem), expected
        )

    def test_sequence_insert_at_position(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        new_elem = torch.randn(2, 3)
        pos = torch.tensor(1, dtype=torch.int64)
        # Insert at position 1: [a, new_elem, b]
        expected = torch.cat([a, new_elem, b], dim=0)
        run_onnx_test(
            self.sequence_insert_at_pos_script.to_model_proto,
            (a, b, new_elem, pos),
            expected,
        )

    def test_sequence_erase(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        # Erase last element: [a, b]
        expected = torch.cat([a, b], dim=0)
        run_onnx_test(self.sequence_erase_script.to_model_proto, (a, b, c), expected)

    def test_sequence_erase_at_position(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        pos = torch.tensor(1, dtype=torch.int64)
        # Erase at position 1: [a, c]
        expected = torch.cat([a, c], dim=0)
        run_onnx_test(
            self.sequence_erase_at_pos_script.to_model_proto, (a, b, c, pos), expected
        )

    def test_concat_from_sequence_with_stack(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        expected = torch.stack([a, b, c], dim=0)
        run_onnx_test(
            self.concat_from_sequence_stack_script.to_model_proto, (a, b, c), expected
        )

    def test_split_to_sequence(self):
        x = torch.randn(6, 3)
        run_onnx_test(self.split_to_sequence_script.to_model_proto, x, x)
