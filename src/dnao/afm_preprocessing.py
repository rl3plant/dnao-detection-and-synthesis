"""Turn a raw AFM height map into a cleaned, normalized 8-bit grayscale image."""
import numpy as np
import cv2
import scipy


# https://github.com/scholi/pySPM/blob/master/pySPM/SPM.py
def correct_lines(image):
    """Subtract the average of each line from the image."""
    new = image.copy()
    new -= np.tile(np.mean(image, axis=1).T, (image.shape[1], 1)).T
    return new


# https://github.com/scholi/pySPM/blob/master/pySPM/SPM.py
def correct_plane(image):
    """Correct the image by subtracting a fitted 2D plane."""
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = image

    A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
    c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)

    new = image.copy()
    new -= c[0] * np.ones(image.shape) + c[1] * X + c[2] * Y
    return new


# https://github.com/scholi/pySPM/blob/master/pySPM/SPM.py
def correct_fit2d(image, dx=2, dy=1, mask=None):
    """Fit the image with a 2D polynomial of order dx x dy and subtract it."""
    x = np.arange(image.shape[1], dtype=float)
    y = np.arange(image.shape[0], dtype=float)
    X0, Y0 = np.meshgrid(x, y)
    if mask is not None:
        X = X0[mask]
        Y = Y0[mask]
        Z = image[mask]
    else:
        X = X0
        Y = Y0
        Z = image
    x2 = X.ravel()
    y2 = Y.ravel()
    A = np.vstack([x2 ** i for i in range(dx + 1)])
    A = np.vstack([A] + [y2 ** i for i in range(1, dy + 1)])
    res = scipy.optimize.lsq_linear(A.T, Z.ravel())
    r = res["x"]
    Z2 = r[0] * np.ones(image.shape)
    for i in range(1, dx + 1):
        Z2 += r[i] * (X0 ** i)
    for i in range(1, dy + 1):
        Z2 += r[dx + i] * (Y0 ** i)

    n = image.copy()
    n -= Z2
    return n


def normalize_image(image, min_val=0, max_val=255):
    new = image.copy()
    return cv2.normalize(new, None, min_val, max_val, cv2.NORM_MINMAX)


def resize_image(raw_image, image_size):
    assert raw_image.shape[0] == raw_image.shape[1]
    return cv2.resize(raw_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)


def preprocess_image(raw_image, image_size):
    clean = correct_lines(raw_image)
    clean = correct_plane(clean)
    clean = correct_fit2d(clean, 2, 2)
    clean = normalize_image(clean)
    return resize_image(clean, image_size)
