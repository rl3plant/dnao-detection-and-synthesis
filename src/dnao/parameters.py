"""Per-shape pipeline parameters, tuned by the thesis author for rectangles/triangles/z-shapes."""
from dataclasses import dataclass


@dataclass
class OrigamiPipelineParameters:
    name: str
    blur_kernel_width: int
    closing_kernel_width: int
    opening_kernel_width: int
    watershed_min_peak_distance: int
    approximated_edge_threshold: int
    shape_type: str


def z_shape_parameters():
    return OrigamiPipelineParameters(
        name="z-origami", blur_kernel_width=9, closing_kernel_width=1,
        opening_kernel_width=5, watershed_min_peak_distance=0,
        approximated_edge_threshold=12, shape_type="z_shape")


def rectangle_shape_parameters():
    return OrigamiPipelineParameters(
        name="rectangle-origami", blur_kernel_width=9, closing_kernel_width=7,
        opening_kernel_width=9, watershed_min_peak_distance=20,
        approximated_edge_threshold=4, shape_type="rectangle")


def triangle_shape_parameters():
    return OrigamiPipelineParameters(
        name="triangle-origami", blur_kernel_width=7, closing_kernel_width=1,
        opening_kernel_width=3, watershed_min_peak_distance=20,
        approximated_edge_threshold=3, shape_type="triangle")


SHAPE_PARAMETERS = {
    "triangles": triangle_shape_parameters(),
    "rectangles": rectangle_shape_parameters(),
    "z_shapes": z_shape_parameters(),
}
