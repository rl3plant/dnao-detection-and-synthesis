"""Turn detected (image, mask) pairs into a patched train/val/test dataset for
DDPM training. Ported from the thesis's ``code/tools/data_preparation.py`` and
``code/data_preparation.py``."""
import os
import shutil

import numpy as np

from dnao.io_utils import create_folder, list_files, read_image, write_images

CLEAN, MASK = "images", "masks"  # matches dnao.detection's output folder names
TEST_DIR, VAL_DIR = "fullsize_test", "fullsize_val"


def split_train_val_test_files(files, val_split, test_split, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    random_files = np.array(sorted(zip(indices, files)))[:, 1]

    val_idx = round(len(files) * (1 - val_split - test_split))
    test_idx = round(len(files) * (1 - test_split))

    return np.split(random_files, [val_idx, test_idx])


def tile_image(image, tile_shape):
    x_size, y_size = tile_shape
    img_shape = image.shape

    tiles = []
    for i in range(img_shape[0] // y_size):
        for j in range(img_shape[1] // x_size):
            tiles.append(image[y_size * i:y_size * i + y_size, x_size * j:x_size * j + y_size])
    return tiles


def extract_tiled_image_mask_patches(image, mask, tile_shape):
    return tile_image(image, tile_shape), tile_image(mask, tile_shape)


def extract_random_image_mask_patches(image, mask, tile_shape, n, seed=42):
    assert image.shape == mask.shape
    rng = np.random.default_rng(seed)
    upper_threshold = np.array(image.shape) - np.array(tile_shape)
    x = rng.integers(0, upper_threshold[0], n, endpoint=True)
    y = rng.integers(0, upper_threshold[1], n, endpoint=True)

    image_patches = [image[x[i]:x[i] + tile_shape[0], y[i]:y[i] + tile_shape[1]] for i in range(n)]
    mask_patches = [mask[x[i]:x[i] + tile_shape[0], y[i]:y[i] + tile_shape[1]] for i in range(n)]
    return image_patches, mask_patches


def augment_data(image_patch, mask_patch, seed=None):
    rng = np.random.default_rng(seed)
    if rng.integers(0, 1, endpoint=True) == 1:
        image_patch = np.fliplr(image_patch)
        mask_patch = np.fliplr(mask_patch)

    for _ in range(rng.integers(0, 3, endpoint=True)):
        image_patch = np.rot90(image_patch)
        mask_patch = np.rot90(mask_patch)
    return image_patch, mask_patch


def _validate_matching_files(src_path):
    cleaned_files = list_files(os.path.join(src_path, CLEAN))
    masked_files = list_files(os.path.join(src_path, MASK))
    assert cleaned_files == masked_files, "cleaned/ and masked/ must contain the same filenames"
    return cleaned_files


def prepare_patch_dataset(src_path, target_path, tile_shape, val_split, test_split, patches_per_image,
                           patch_extraction="random", data_augmentation=False):
    """Split `src_path`/{cleaned,masked} into train/val/test, extracting fixed-size
    patches for training and keeping full-size copies of val/test for evaluation."""
    files = _validate_matching_files(src_path)

    train_files, val_files, test_files = split_train_val_test_files(files, val_split, test_split)
    split_files = {"train": train_files, "val": val_files, "test": test_files}

    image_src_dir, mask_src_dir = os.path.join(src_path, CLEAN), os.path.join(src_path, MASK)
    image_target_dir, mask_target_dir = os.path.join(target_path, CLEAN), os.path.join(target_path, MASK)

    full_test_dir, full_val_dir = os.path.join(target_path, TEST_DIR), os.path.join(target_path, VAL_DIR)
    full_image_test_dir, full_mask_test_dir = os.path.join(full_test_dir, CLEAN), os.path.join(full_test_dir, MASK)
    full_image_val_dir, full_mask_val_dir = os.path.join(full_val_dir, CLEAN), os.path.join(full_val_dir, MASK)

    for d in (full_image_test_dir, full_mask_test_dir, full_image_val_dir, full_mask_val_dir):
        create_folder(d)

    if data_augmentation:
        patches_per_image *= 8

    for split in ("train", "val", "test"):
        image_patch_dir, mask_patch_dir = os.path.join(image_target_dir, split), os.path.join(mask_target_dir, split)
        create_folder(image_patch_dir)
        create_folder(mask_patch_dir)

        for file in split_files[split]:
            image = read_image(os.path.join(image_src_dir, file))
            mask = read_image(os.path.join(mask_src_dir, file))

            if patch_extraction == "random" and split == "train":
                image_patches, mask_patches = extract_random_image_mask_patches(image, mask, tile_shape, patches_per_image)
                if data_augmentation:
                    augmented = [augment_data(ip, mp) for ip, mp in zip(image_patches, mask_patches)]
                    image_patches, mask_patches = zip(*augmented) if augmented else ([], [])
            elif patch_extraction == "tiled" or split in ("val", "test"):
                image_patches, mask_patches = extract_tiled_image_mask_patches(image, mask, tile_shape)
            else:
                raise ValueError("patch_extraction must be 'tiled' or 'random'")

            write_images(os.path.join(image_patch_dir, file[:-4]), image_patches)
            write_images(os.path.join(mask_patch_dir, file[:-4]), mask_patches)

            if split == "test":
                shutil.copy(os.path.join(image_src_dir, file), full_image_test_dir)
                shutil.copy(os.path.join(mask_src_dir, file), full_mask_test_dir)
            if split == "val":
                shutil.copy(os.path.join(image_src_dir, file), full_image_val_dir)
                shutil.copy(os.path.join(mask_src_dir, file), full_mask_val_dir)
