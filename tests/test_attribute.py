# SPDX-License-Identifier: Apache-2.0

import onnx
import pytest
import torch

from onnx2fx.exceptions import ConversionError
from onnx2fx.utils import get_attribute, get_attributes, get_attribute_or_input


class TestAttributeParser:
    """Test attribute parsing utilities."""

    def test_get_attribute_int(self):
        """get_attribute should parse integer attributes."""
        node = onnx.helper.make_node("Test", [], [], axis=2)
        assert get_attribute(node, "axis") == 2

    def test_get_attribute_float(self):
        """get_attribute should parse float attributes."""
        node = onnx.helper.make_node("Test", [], [], alpha=0.5)
        assert get_attribute(node, "alpha") == 0.5

    def test_get_attribute_ints(self):
        """get_attribute should parse integer list attributes."""
        node = onnx.helper.make_node("Test", [], [], pads=[1, 2, 3, 4])
        assert get_attribute(node, "pads") == [1, 2, 3, 4]

    def test_get_attribute_string(self):
        """get_attribute should parse string attributes."""
        node = onnx.helper.make_node("Test", [], [], mode="linear")
        assert get_attribute(node, "mode") == "linear"

    def test_get_attribute_default(self):
        """get_attribute should return default for missing attributes."""
        node = onnx.helper.make_node("Test", [], [])
        assert get_attribute(node, "missing", default=42) == 42
        assert get_attribute(node, "missing") is None

    def test_get_attributes_all(self):
        """get_attributes should return all attributes as dict."""
        node = onnx.helper.make_node("Test", [], [], axis=1, keepdims=0)
        attrs = get_attributes(node)
        assert attrs["axis"] == 1
        assert attrs["keepdims"] == 0


class DummyBuilder:
    def __init__(self, initializer_map=None):
        self.initializer_map = initializer_map or {}

    def get_value(self, name):
        return name


class TestAttributeOrInput:
    def test_attribute_allowed(self):
        node = onnx.helper.make_node("Test", ["x"], ["y"], axes=[1, 2])
        builder = DummyBuilder()
        result = get_attribute_or_input(
            builder,
            node,
            attr_name="axes",
            input_index=1,
            opset_version=12,
            attr_allowed_until=17,
            input_allowed_since=13,
        )
        assert result == [1, 2]

    def test_attribute_not_allowed(self):
        node = onnx.helper.make_node("Test", ["x"], ["y"], axes=[1])
        builder = DummyBuilder()
        with pytest.raises(ConversionError):
            get_attribute_or_input(
                builder,
                node,
                attr_name="axes",
                input_index=1,
                opset_version=18,
                attr_allowed_until=17,
                input_allowed_since=13,
            )

    def test_input_not_allowed(self):
        node = onnx.helper.make_node("Test", ["x", "axes"], ["y"])
        builder = DummyBuilder({"axes": torch.tensor([1, 2])})
        with pytest.raises(ConversionError):
            get_attribute_or_input(
                builder,
                node,
                attr_name="axes",
                input_index=1,
                opset_version=12,
                attr_allowed_until=17,
                input_allowed_since=13,
            )

    def test_input_allowed(self):
        node = onnx.helper.make_node("Test", ["x", "axes"], ["y"])
        builder = DummyBuilder({"axes": torch.tensor([1, 2])})
        result = get_attribute_or_input(
            builder,
            node,
            attr_name="axes",
            input_index=1,
            opset_version=13,
            attr_allowed_until=17,
            input_allowed_since=13,
        )
        assert result == [1, 2]
