# SPDX-License-Identifier: Apache-2.0
"""Utilities for sanitizing ONNX names to valid Python identifiers."""

import keyword
import re


def sanitize_name(name: str) -> str:
    """Sanitize a name to be a valid Python identifier.

    ONNX tensor names can contain characters that are not valid in Python
    identifiers (e.g., '.', '/', '-') or may start with digits. This function
    converts such names to valid Python identifiers.

    Parameters
    ----------
    name : str
        The ONNX tensor name to sanitize.

    Returns
    -------
    str
        A valid Python identifier.
    """
    # Replace common invalid characters
    safe_name = name.replace(".", "_").replace("/", "_").replace("-", "_")

    # Replace any remaining non-alphanumeric characters (except underscore)
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", safe_name)

    # If name starts with a digit, prefix with underscore
    if safe_name and safe_name[0].isdigit():
        safe_name = "_" + safe_name

    # Handle empty names
    if not safe_name:
        safe_name = "_unnamed"

    # Handle Python keywords
    if keyword.iskeyword(safe_name):
        safe_name = safe_name + "_"

    return safe_name
