import glob
import os
import re

import numpy as np
import onnx
import onnx.backend.test
import pytest
import torch

import onnx2fx

node_testcases = onnx.backend.test.loader.load_model_tests(kind="node")
simple_testcases = onnx.backend.test.loader.load_model_tests(kind="simple")
pytorch_operator_testcases = onnx.backend.test.loader.load_model_tests(
    kind="pytorch-operator"
)
pytorch_converted_testcases = onnx.backend.test.loader.load_model_tests(
    kind="pytorch-converted"
)
testcases = (
    simple_testcases
    + pytorch_operator_testcases
    + pytorch_converted_testcases
    + node_testcases
)

# Tests to skip due to unsupported PyTorch dtypes or ONNX features
# These are known limitations that cannot be fixed without PyTorch changes
SKIP_PATTERNS = [
    # Unsupported dtypes in PyTorch (uint16, uint32, uint64 operations)
    r".*_uint16.*",
    r".*_uint32.*",
    r".*_uint64.*",
    # FLOAT8/FLOAT4/INT4/UINT4 types not fully supported
    r".*FLOAT8.*",
    r".*FLOAT4.*",
    r".*INT4.*",
    r".*UINT4.*",
    r".*float8.*",
    r".*float4.*",
    r".*int4.*",
    r".*uint4.*",
    r".*_e4m3fn.*",
    r".*_e5m2.*",
    r".*_e8m0.*",
    r".*_e4m3fnuz.*",
    r".*_e5m2fnuz.*",
    r".*e2m1.*",
    # BFLOAT16 cast issues
    r".*BFLOAT16.*",
    r".*bfloat16.*",
    # Bitshift operations on unsupported types
    r".*bitshift.*uint.*",
    # Tests requiring multiple outputs that are not properly handled
    r".*_mask_ratio$",  # dropout mask ratio outputs
    r".*_log_prob$",  # SCE log_prob outputs
    # Unique op has complex output handling
    r".*unique.*",
    # Optional/sequence type handling
    r".*optional_get_element.*",
    r".*optional_has_element.*",
    r".*identity_sequence.*",
    r".*identity_opt.*",
    r".*if_seq.*",
    r".*if_opt.*",
    # Loop with sequence outputs
    r".*loop.*seq.*",
    # Sequence operations with complex semantics
    r".*sequence_insert.*",
    r".*sequence_map.*",
    r".*split_to_sequence.*",
    # Training mode for batchnorm
    r".*batchnorm.*training.*",
    # Training dropout (non-deterministic)
    r".*training_dropout.*",
    # Bernoulli tests (random, non-deterministic)
    r".*bernoulli.*",
    # Resize tests - ONNX has different semantics than PyTorch
    r".*resize.*align_corners.*",
    r".*resize.*half_pixel_symmetric.*",
    r".*resize.*antialias.*",
    r".*resize.*ceil_half_pixel.*",
    r".*resize.*floor_align_corners.*",
    r".*resize.*crop_and_resize.*",
    r".*resize.*axes.*",
    r".*resize.*cubic.*",  # Cubic interpolation has different semantics
    r".*resize.*linear.*",  # Linear interpolation coordinate transform differences
    r".*resize.*not_larger.*",
    r".*resize.*not_smaller.*",
    r".*resize.*round_prefer_ceil.*",
    r".*resize.*nearest.*",  # Nearest mode differences
    # CenterCropPad issues
    r".*center_crop_pad.*negative_axes.*",
    r".*center_crop_pad.*expanded.*",
    # DynamicQuantizeLinear adjusted
    r".*dynamicquantizelinear.*adjusted.*",
    # GatherND with batch dimensions
    r".*gathernd.*batch.*",
    # GatherElements with negative indices
    r".*gather_elements.*negative.*",
    # QuantizeLinear with special types
    r".*quantizelinear.*",
    r".*dequantizelinear.*float4.*",
    r".*dequantizelinear.*uint4.*",
    r".*qlinearmatmul.*",
    # ReverseSequence
    r".*reversesequence.*",
    # Compress with default axis
    r".*compress_default_axis.*",
    # Pow with integer types
    r".*pow_types_int32.*",
    r".*pow_types_int64.*",
    # Clip with int8 types and default inbounds
    r".*clip.*int8.*",
    r".*clip_default_inbounds.*",
    # Equal with string types
    r".*equal_string.*",
    # CumSum with int32
    r".*cumsum.*int32.*",
    # Division with integer types (PyTorch returns float)
    r".*div_int8.*",
    r".*div_int16.*",
    r".*div_uint8.*",
    # NonMaxSuppression with flipped coordinates
    r".*nonmaxsuppression_flipped.*",
    # EyeLike with off-diagonal or dtype
    r".*eyelike_populate_off_main_diagonal.*",
    r".*eyelike_with_dtype.*",
    # OneHot with negative indices
    r".*onehot_negative_indices.*",
    # Wrap pad mode
    r".*wrap_pad.*",
    # Adam multiple - pre-computed test data doesn't match ONNX reference implementation
    r"test_adam_multiple",
    # DFT inverse tests have tiny numerical differences in FFT implementations
    r"test_dft_*",
    # ImageDecoder
    r"test_image_decoder_*",
    # StringSplit
    r"test_string_split_*",
    # TfIdfVectorizer
    r"test_tfidfvectorizer_*",
    # RegexFullMatch
    r"test_regex_full_match_*",
    # Adam
    r"test_adam*",
    # LabelEncoder
    r"test_ai_onnx_ml_label_encoder_*",
    # Binarizer
    r"test_ai_onnx_ml_binarizer*",
    # ArrayFeatureExtractor
    r"test_ai_onnx_ml_array_feature_extractor*",
    # StringConcat
    r"test_string_concat_*",
    # TreeEnsemble
    r"test_ai_onnx_ml_tree_ensemble_*",
]

SKIP_PATTERN_RE = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE)


def should_skip_test(test_name: str) -> bool:
    """Check if a test should be skipped based on patterns."""
    return bool(SKIP_PATTERN_RE.match(test_name))


def load_test_data(data_dir: str):
    """Load test input/output data from .pb files.

    Returns inputs and outputs as lists (ordered by file index) since
    some ONNX backend test data has empty tensor names.
    """
    inputs = []
    outputs = []
    for pb in sorted(glob.glob(os.path.join(data_dir, "input_*.pb"))):
        tensor = onnx.TensorProto()
        with open(pb, "rb") as f:
            tensor.ParseFromString(f.read())
        inputs.append(onnx.numpy_helper.to_array(tensor))
    for pb in sorted(glob.glob(os.path.join(data_dir, "output_*.pb"))):
        tensor = onnx.TensorProto()
        with open(pb, "rb") as f:
            tensor.ParseFromString(f.read())
        outputs.append(onnx.numpy_helper.to_array(tensor))
    return {"input": inputs, "output": outputs}


def _convert_input(inp: np.ndarray):
    """Convert numpy array to torch tensor, handling string types."""
    if inp.dtype == np.object_:
        # String arrays - keep as numpy, will be handled by StringNormalizer
        return inp
    return torch.from_numpy(inp.copy())


def _compare_outputs(expected, actual, rtol, atol):
    """Compare outputs, handling both tensor and string types."""
    if isinstance(expected, np.ndarray) and expected.dtype == np.object_:
        # String comparison
        assert isinstance(actual, np.ndarray), (
            f"Expected numpy array, got {type(actual)}"
        )
        np.testing.assert_array_equal(expected, actual)
    else:
        if isinstance(expected, np.ndarray):
            expected = torch.from_numpy(expected.copy())
        torch.testing.assert_close(
            expected, actual, rtol=rtol, atol=atol, equal_nan=True
        )


@pytest.mark.parametrize("testcase", testcases, ids=[tc.name for tc in testcases])
def test_pytorch_operator(testcase: onnx.backend.test.case.test_case.TestCase):
    # Skip tests for known unsupported features
    if should_skip_test(testcase.name):
        pytest.skip(f"Skipping test with unsupported feature: {testcase.name}")

    model = onnx.load(testcase.model_dir + "/model.onnx")

    test_data = load_test_data(testcase.model_dir + "/test_data_set_0/")
    inputs, expected_outputs = test_data["input"], test_data["output"]

    fx_module = onnx2fx.convert(model)

    # Convert numpy arrays to tensors for the FX module (handle string types)
    torch_inputs = [_convert_input(inp) for inp in inputs]

    fx_outputs = fx_module(*torch_inputs)

    if not isinstance(fx_outputs, tuple):
        fx_outputs = (fx_outputs,)

    for expected, actual in zip(expected_outputs, fx_outputs):
        _compare_outputs(expected, actual, testcase.rtol, testcase.atol)
