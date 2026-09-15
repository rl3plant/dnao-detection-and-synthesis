"""Detect DNAO shape contours in a thresholded image, and split merged/oversized blobs."""
import imutils
import numpy as np
import cv2

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

RECTANGLE = "rectangle"
Z_SHAPE = "z_shape"
TRIANGLE = "triangle"


def detect_origami_contours(threshold, watershed_min_peak_distance, shape_type):
    if shape_type == RECTANGLE:
        contours = watershed_contours(threshold, watershed_min_peak_distance)
    elif shape_type == Z_SHAPE:
        current_contours = cv2.findContours(threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(current_contours)
    elif shape_type == TRIANGLE:
        # take care of big holes in triangles
        filled_threshold = ndimage.binary_fill_holes(threshold, structure=np.ones((9, 9)))
        contours = watershed_contours(filled_threshold, watershed_min_peak_distance)
    else:
        raise ValueError(f"Shape type not supported: {shape_type}")
    average_area = np.mean([cv2.contourArea(contour) for contour in contours])
    return clean_contours(contours, average_area / 4)


def watershed_contours(image, watershed_min_distance):
    distance = ndimage.distance_transform_edt(image)
    coords = peak_local_max(distance, min_distance=watershed_min_distance, labels=image.astype(int))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=image)

    contours = []
    for label in np.unique(labels):
        # skip background label
        if label == 0:
            continue

        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255
        # extract largest contour in mask
        current_contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_contours = imutils.grab_contours(current_contours)
        contours.append(max(current_contours, key=cv2.contourArea))
    return contours


def clean_contours(contours, min_area=None, max_area=None):
    cleaned_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area and min_area > area:
            continue
        if max_area and max_area < area:
            continue
        cleaned_contours.append(contour)
    return cleaned_contours


def recursive_contour_approximation(contour, max_edges):
    precision = 0.005
    contour_safe = contour
    while len(contour) > max_edges:
        perimeter = cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, precision * perimeter, True)
        if cv2.contourArea(contour) <= 0:
            break
        contour_safe = contour
        precision *= 1.05
    return contour_safe


def split_contour_by_points(contour, vertex_1, vertex_2):
    contour_1 = contour[vertex_1:vertex_2 + 1]
    contour_2 = np.concatenate([contour[vertex_2:len(contour)], contour[0:vertex_1 + 1]])
    return [contour_1, contour_2]


def get_sorted_cuts(contour):
    n = len(contour)
    min_offset = n // 4

    dists = np.linalg.norm((contour - contour[:, None]), axis=3)[:, :, 0]

    indices_with_offset = np.triu_indices_from(dists, k=min_offset)
    relevant_indices = np.where(np.abs(indices_with_offset[0] - indices_with_offset[1]) < n - min_offset)[0]

    points1 = indices_with_offset[0][relevant_indices]
    points2 = indices_with_offset[1][relevant_indices]
    distances = dists[points1, points2]
    sorting = np.argsort(distances)
    return points1[sorting], points2[sorting], distances[sorting]


def area_ratio_of_partitions(contour, points1, points2):
    area_1, area_2 = np.empty(len(points1)), np.empty(len(points1))
    for i in range(len(points1)):
        contour_1, contour_2 = split_contour_by_points(contour, points1[i], points2[i])
        area_1[i], area_2[i] = cv2.contourArea(contour_1), cv2.contourArea(contour_2)

    low, high = np.minimum(area_1, area_2), np.maximum(area_1, area_2)
    return np.divide(high, low, out=np.full_like(area_1, np.inf), where=low != 0)


def best_partition(contour, shape_type):
    distance_exponent = 2 if shape_type == Z_SHAPE else 3
    points1, points2, distances = get_sorted_cuts(contour)
    area_ratio = area_ratio_of_partitions(contour, points1, points2)

    split_metric = distances ** distance_exponent * area_ratio
    best_split = np.argmin(split_metric)

    return split_contour_by_points(contour, points1[best_split], points2[best_split])


def partition_contours(contours, shape_type):
    if shape_type not in (RECTANGLE, Z_SHAPE, TRIANGLE):
        raise ValueError(f"Shape type not supported: {shape_type}")
    average_area = np.mean([cv2.contourArea(contour) for contour in contours])
    max_area = average_area * 1.35
    partitioned_contours = recursive_partition(contours, max_area, shape_type)
    average_area = np.mean([cv2.contourArea(contour) for contour in partitioned_contours])
    return clean_contours(partitioned_contours, min_area=average_area / 4)


def recursive_partition(contours, max_area, shape_type):
    partitioned_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > max_area and len(contour) >= 4:
            cleaned_contour = recursive_contour_approximation(contour, 50)
            split_contours = best_partition(cleaned_contour, shape_type)
            for split_contour in split_contours:
                if cv2.contourArea(split_contour) > max_area and len(split_contour) >= 4:
                    partitioned_contours.extend(recursive_partition([split_contour], max_area, shape_type))
                else:
                    partitioned_contours.append(split_contour)
        else:
            partitioned_contours.append(contour)
    return partitioned_contours


def approximate_contours(contours, max_edges=4):
    approximated_contours = tuple()
    for contour in contours:
        if max_edges > 0:
            contour = recursive_contour_approximation(contour, max_edges)
        else:
            perimeter = cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        approximated_contours = approximated_contours + (contour,)
    return approximated_contours
