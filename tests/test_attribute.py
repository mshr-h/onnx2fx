# SPDX-License-Identifier: Apache-2.0

import onnx

from onnx2fx.utils import get_attribute, get_attributes


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
