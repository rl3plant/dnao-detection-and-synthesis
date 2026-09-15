"""Foreground/background thresholding of preprocessed AFM images (GMM + morphology)."""
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture


def sort_labels_by_brightness(centers, labels, k):
    # sort labels by values (labels sorted ascending by center values)
    sort_idx = np.argsort(centers)
    new_labels = np.zeros(labels.size, dtype=np.int8)
    for i in range(k):
        new_labels[(labels == sort_idx[i])] = i
    return new_labels


def gmm_thresholding(image, k):
    pixels = np.float32(np.reshape(image, (-1, 1)))

    gmm = GaussianMixture(n_components=k, covariance_type="tied", n_init=1)
    gmm = gmm.fit(pixels)
    labels = gmm.predict(pixels)

    sorted_labels = sort_labels_by_brightness(gmm.means_[:, 0], labels, k)
    return sorted_labels.reshape(np.shape(image))


def opening(image, kernel_width):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_width))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel_width):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_width))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def origami_thresholding(image, kernel_width=9):
    smoothed = cv2.GaussianBlur(image, (kernel_width, kernel_width), cv2.BORDER_DEFAULT)
    return gmm_thresholding(smoothed, 2)


def morph_thresholding(image, opening_kernel_width=9, closing_kernel_width=9):
    float_image = np.float64(image)
    opened_image = opening(float_image, opening_kernel_width)
    return closing(opened_image, closing_kernel_width)
