"""Readers for raw AFM scan formats (Bruker .spm, Asylum .ibw, JPK .jpk, MI .mi)."""
import numpy as np
import cv2
import pySPM as spm
import tifffile
import igor2 as igor


def read_afm_image(filename):
    if filename[-3:] == 'spm':
        image = read_Bruker_image(filename)
    elif filename[-3:] == 'ibw':
        image = read_IBW_image(filename)
    elif filename[-3:] == 'jpk':
        image = read_JPK_image(filename)
    elif filename.endswith('.mi'):
        image = read_MI_image(filename)
    elif filename[-3].isdigit():
        # Bruker scans are sometimes saved with a numeric extension (e.g. ".001")
        image = read_Bruker_image(filename)
    else:
        return None
    return image


def read_Bruker_image(file_name):
    afm_file = spm.Bruker(file_name)
    if file_name[-3:-1] == 'sp':
        height = afm_file.get_channel("Height Sensor")
    else:
        height = afm_file.get_channel("Height")

    image = cv2.flip(height.pixels, 90)
    image = cv2.rotate(image, cv2.ROTATE_180)
    return image


def read_JPK_image(file_name):
    tif = tifffile.TiffFile(file_name)
    image = tif.pages[1].asarray().astype(np.float64) * (-1)

    image = cv2.flip(image, 90)
    image = cv2.rotate(image, cv2.ROTATE_180)
    return image


def read_IBW_image(file_name):
    ibw_file = igor.binarywave.load(file_name)
    all_images = ibw_file['wave']['wData']
    image = all_images[:, :, 1]
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def read_MI_image(file_name):
    binary_data = b""
    x_pixels = y_pixels = None
    binary_index = 0

    with open(file_name, 'rb') as f:
        for idx, line in enumerate(f):
            if line.startswith(b'xPixels'):
                x_pixels = int(line.split()[1])
            if line.startswith(b'yPixels'):
                y_pixels = int(line.split()[1])
            if line.strip() == b'data          BINARY_32':
                binary_index = idx
                break

    with open(file_name, 'rb') as f:
        file = list(f)
        binary_file = file[binary_index + 1:]
        binary_data = binary_data.join(binary_file)

    image = np.frombuffer(binary_data, dtype=np.int32)
    image = image / (2 ** 32 / 2.)
    assert x_pixels is not None and y_pixels is not None, f"{file_name} contains no information about width and height"
    assert x_pixels == y_pixels, f"{file_name} is not quadratic"
    image = image[0:x_pixels * y_pixels].reshape(x_pixels, y_pixels)
    image = cv2.flip(image, 90)
    image = cv2.rotate(image, cv2.ROTATE_180)

    return image
