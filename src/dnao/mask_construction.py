"""Fit idealized shape masks (rectangle / triangle / z-shape) to detected contours."""
import cv2
import numpy as np

from dnao.contour_detection import RECTANGLE, Z_SHAPE, TRIANGLE


def contour_center(contour):
    m = cv2.moments(contour)
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def scale_contour(contour, scale):
    center = contour_center(contour)
    cnt_norm = contour - center
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + center
    return cnt_scaled.astype(np.int32)


def rotate_contour(contour, angle):
    centered_contour = contour - contour_center(contour)

    coordinates = centered_contour[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)
    centered_contour[:, 0, 0], centered_contour[:, 0, 1] = xs, ys
    return (centered_contour + contour_center(contour)).astype(np.int32)


def draw_mask_image(image, contours):
    background = np.zeros(image.shape)
    for contour in contours:
        cv2.drawContours(background, [contour], -1, (255, 255, 255), cv2.FILLED)
    return background


def draw_hollow_triangle_mask(image, contours):
    background = np.zeros(image.shape)
    for contour in contours:
        inner_triangle = scale_contour(contour, 0.5)
        cv2.drawContours(background, [contour, inner_triangle], -1, (255, 255, 255), cv2.FILLED)
    return background


def extract_rectangle_bounding_boxes(approximated_contours, raw_contours):
    bounding_boxes = []
    for i in range(len(approximated_contours)):
        x, y = contour_center(raw_contours[i])
        (_, _), (width, height), rotation = cv2.minAreaRect(approximated_contours[i])
        bounding_boxes.append(((x, y), (width, height), rotation))
    return bounding_boxes


def extract_triangle_bounding_boxes(approximated_contours, raw_contours):
    new_triangles = []
    for i in range(len(approximated_contours)):
        area, triangle = cv2.minEnclosingTriangle(approximated_contours[i])
        # cv2.minEnclosingTriangle's output shape has varied across OpenCV versions
        # ((3, 2) vs (3, 1, 2)); normalize to the (N, 1, 2) contour-point convention
        # used everywhere else so downstream shape math (e.g. triangle_rotation) is stable.
        triangle = triangle.reshape(-1, 1, 2)
        x, y = contour_center(triangle)
        real_x, real_y = contour_center(raw_contours[i])
        new_triangle = triangle - [x, y] + [real_x, real_y]
        new_triangles.append(np.round(new_triangle).astype(int))
    return new_triangles


def normalize_triangle_shape(triangles):
    equilateral_triangles = []
    for triangle in triangles:
        rotation = -triangle_rotation(triangle)
        x, y = np.average(triangle, axis=0)[0]

        tri_side_length = np.sqrt(cv2.contourArea(triangle) * 4 / np.sqrt(3))
        equilateral_triangles.append(np.round(build_equilateral_triangle(x, y, rotation, tri_side_length)).astype(int))
    return equilateral_triangles


def build_equilateral_triangle(x, y, rotation, side_length):
    center_distance = side_length / np.sqrt(3)
    p1 = [0, center_distance]
    p2 = [side_length / 2.0, -center_distance / 2.0]
    p3 = [-side_length / 2.0, -center_distance / 2.0]

    def rotate_around_origin(p, rot):
        px, py = p
        xx = px * np.cos(rot) + py * np.sin(rot)
        yy = -px * np.sin(rot) + py * np.cos(rot)
        return [xx, yy]

    new_p1 = np.array(rotate_around_origin(p1, rotation)) + [x, y]
    new_p2 = np.array(rotate_around_origin(p2, rotation)) + [x, y]
    new_p3 = np.array(rotate_around_origin(p3, rotation)) + [x, y]

    return np.array([new_p1[None], new_p2[None], new_p3[None]])


def triangle_rotation(triangle):
    sides = [triangle[0] - triangle[1],
             triangle[1] - triangle[2],
             triangle[2] - triangle[0]]
    longest_side = sides[np.argmax(np.linalg.norm(sides, axis=2))][0]
    return np.arctan2(longest_side[1], longest_side[0])


def get_contour_rotations(contour, rotations):
    rotated_contours = [rotate_contour(contour, rotation) for rotation in rotations]
    return [np.round(contour - contour_center(contour)).astype(np.int32) for contour in rotated_contours]


def fit_z_shape_contours(contours):
    # z_shape_contour hard coded
    contour_s = np.array([[[0, 0]], [[0, 22]], [[22, 22]], [[22, 36]], [[30, 36]], [[30, 14]], [[8, 14]], [[8, 0]]])
    contour_z = contour_s * [[-1, 1]]

    rotations = np.linspace(0, 175, 36)
    rotated_s = get_contour_rotations(contour_s, rotations)
    rotated_z = get_contour_rotations(contour_z, rotations)

    size = 200
    center_offset = [size // 2, size // 2]
    rotated_s = [contour + center_offset for contour in rotated_s]
    rotated_z = [contour + center_offset for contour in rotated_z]
    images_s = [draw_mask_image(np.zeros((size, size)), [contour]) for contour in rotated_s]
    images_z = [draw_mask_image(np.zeros((size, size)), [contour]) for contour in rotated_z]

    masks = []
    for contour in contours:
        s_overlaps, z_overlaps = [], []
        centered_contour = np.round(contour - contour_center(contour) + center_offset).astype(np.int32)
        image = draw_mask_image(np.zeros((size, size)), [centered_contour])
        for rotation in range(len(rotations)):
            s_overlaps.append(np.count_nonzero(np.logical_and(images_s[rotation], image)))
            z_overlaps.append(np.count_nonzero(np.logical_and(images_z[rotation], image)))
        if max(s_overlaps) > max(z_overlaps):
            best_match_contour = rotated_s[np.argmax(s_overlaps)]
        else:
            best_match_contour = rotated_z[np.argmax(z_overlaps)]
        centered_best_match_contour = best_match_contour - contour_center(best_match_contour)
        norm_mask = np.round(centered_best_match_contour + contour_center(contour)).astype(np.int32)

        scaled_mask = scale_contour(norm_mask, np.sqrt(cv2.contourArea(contour) / cv2.contourArea(norm_mask)))
        masks.append(scaled_mask)
    return masks


def extract_shape_mask(image, contours, approximated_contours, shape_type):
    if shape_type == RECTANGLE:
        rectangles = extract_rectangle_bounding_boxes(contours, approximated_contours)
        mask_contours = [np.round(cv2.boxPoints(rectangle)).astype(int)[:, None, :] for rectangle in rectangles]
        mask_image = draw_mask_image(image, mask_contours)
    elif shape_type == Z_SHAPE:
        mask_contours = fit_z_shape_contours(approximated_contours)
        mask_image = draw_mask_image(image, mask_contours)
    elif shape_type == TRIANGLE:
        mask_contours = extract_triangle_bounding_boxes(approximated_contours, contours)
        mask_contours = normalize_triangle_shape(mask_contours)
        mask_image = draw_hollow_triangle_mask(image, mask_contours)
    else:
        raise ValueError(f"Shape type not supported: {shape_type}")
    return mask_contours, mask_image
