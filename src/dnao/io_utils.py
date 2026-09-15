"""Shared image/contour/file I/O helpers used by both the detection and
data-prep pipelines."""
import os

import cv2
import numpy as np


def write_image(target_path, image, hot_map=False):
    if hot_map:
        im_color = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(target_path, im_color)
    else:
        cv2.imwrite(target_path, image)


def write_images(target_path, images):
    for idx, image in enumerate(images):
        write_image(target_path + "_" + str(idx) + ".png", image)


def write_contours(target_path, contours):
    with open(target_path, 'w') as f:
        for idx, contour in enumerate(contours):
            line = f"{idx}; "
            line += "; ".join(f"{e[0]} {e[1]}" for e in contour[:, 0, :])
            line += "\n"
            f.write(line)


def overlay_images(base, overlay):
    new_image = base.copy()
    new_image[overlay != 0] = overlay[overlay != 0]
    return new_image


def draw_contour_image(image, contours):
    contour_image = np.zeros(image.shape + (3,))
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (255, 0, 0), 2, cv2.LINE_AA)
    return contour_image[:, :, 0]


def draw_contour_over_image(image, contours):
    overlay = draw_contour_image(image, contours)
    return overlay_images(image, overlay)


def read_image(src_path, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imread(src_path, flag)


def create_folder(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def list_files(sample_dir):
    return sorted(file for file in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, file)))
