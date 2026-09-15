"""End-to-end DNAO detection pipeline: raw AFM scan -> preprocessed image -> shape
contours -> idealized shape masks.

This is a straight port of the classical-CV pipeline from Noah Ruhmer's thesis
(``code/dnao_processing.py`` in the original repo), split into callable
functions instead of a hardcoded top-level script.
"""
import os

from dnao.afm_preprocessing import preprocess_image
from dnao.afm_read_in import read_afm_image
from dnao.afm_thresholding import origami_thresholding, morph_thresholding
from dnao.contour_detection import detect_origami_contours, partition_contours, approximate_contours
from dnao.mask_construction import extract_shape_mask
from dnao.parameters import OrigamiPipelineParameters, SHAPE_PARAMETERS
from dnao.io_utils import create_folder, list_files, read_image, write_image, write_contours


def detect_contours(image, parameters: OrigamiPipelineParameters):
    threshold = origami_thresholding(image, parameters.blur_kernel_width)
    morphed_threshold = morph_thresholding(threshold, parameters.opening_kernel_width, parameters.closing_kernel_width)
    contours = detect_origami_contours(morphed_threshold, parameters.watershed_min_peak_distance, parameters.shape_type)
    return partition_contours(contours, parameters.shape_type)


def extract_masks(image, shape_contours, parameters: OrigamiPipelineParameters):
    approximated_contours = approximate_contours(shape_contours, parameters.approximated_edge_threshold)
    return extract_shape_mask(image, shape_contours, approximated_contours, parameters.shape_type)


def detect_and_extract_masks(image, parameters: OrigamiPipelineParameters):
    shape_contours = detect_contours(image, parameters)
    mask_contours, mask_image = extract_masks(image, shape_contours, parameters)
    return shape_contours, mask_contours, mask_image


def preprocess_directory(src_dir, target_dir, image_size=1024):
    """Read every raw AFM scan in `src_dir` and write a cleaned/normalized PNG to `target_dir`."""
    create_folder(target_dir)
    for file in list_files(src_dir):
        raw_image = read_afm_image(os.path.join(src_dir, file))
        if raw_image is None:
            print(f"Error in reading AFM file, skipping {os.path.join(src_dir, file)} ...")
            continue
        cleaned_image = preprocess_image(raw_image, image_size)
        write_image(os.path.join(target_dir, os.path.splitext(file)[0] + ".png"), cleaned_image)


def _write_sample(target_dir, image, mask_image, shape_contours, mask_contours, file_name):
    write_image(os.path.join(target_dir, "images", file_name + ".png"), image)
    write_image(os.path.join(target_dir, "masks", file_name + ".png"), mask_image)
    write_contours(os.path.join(target_dir, "shape_contours", file_name + ".txt"), shape_contours)
    write_contours(os.path.join(target_dir, "mask_contours", file_name + ".txt"), mask_contours)


def detect_directory(src_dir, target_dir, shape, parameters=None):
    """Run shape detection + mask extraction over every preprocessed image in `src_dir`
    (all assumed to be of the given `shape`: "rectangles" | "triangles" | "z_shapes"),
    writing images/masks/contours into `target_dir`.
    """
    parameters = parameters or SHAPE_PARAMETERS[shape]
    for output in ["images", "masks", "shape_contours", "mask_contours"]:
        create_folder(os.path.join(target_dir, output))

    for index, file in enumerate(list_files(src_dir), start=1):
        image = read_image(os.path.join(src_dir, file))
        if image is None:
            print(f"Error in reading image file, skipping {os.path.join(src_dir, file)} ...")
            continue
        file_name = f"{parameters.shape_type}_{index}"
        shape_contours, mask_contours, mask_image = detect_and_extract_masks(image, parameters)
        _write_sample(target_dir, image, mask_image, shape_contours, mask_contours, file_name)
