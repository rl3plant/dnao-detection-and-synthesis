"""Regression smoke test: the detection pipeline must run on the bundled example
scan and find at least one shape of each kind."""
from pathlib import Path

import pytest

from dnao.afm_preprocessing import preprocess_image
from dnao.afm_read_in import read_afm_image
from dnao.detection import detect_and_extract_masks
from dnao.parameters import SHAPE_PARAMETERS

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SCAN = next((REPO_ROOT / "examples").glob("*.spm"))


@pytest.fixture(scope="module")
def preprocessed_example():
    raw_image = read_afm_image(str(EXAMPLE_SCAN))
    assert raw_image is not None
    return preprocess_image(raw_image, image_size=1024).astype("uint8")


@pytest.mark.parametrize("shape", ["rectangles", "triangles", "z_shapes"])
def test_detection_finds_shapes(preprocessed_example, shape):
    params = SHAPE_PARAMETERS[shape]
    shape_contours, mask_contours, mask_image = detect_and_extract_masks(preprocessed_example, params)

    assert len(shape_contours) > 0
    assert len(mask_contours) == len(shape_contours)
    assert (mask_image > 0).any()
