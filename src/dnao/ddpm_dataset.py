"""Dataset/dataloader construction for the segmentation-guided DDPM, factored out of
the thesis's ``code/cond_ddpm/main.py::run_main``."""
import os

import datasets
import torch
from torchvision import transforms


def _list_dir(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def build_datasets(mode, img_dir, seg_dir, segmentation_guided):
    """Build HF `datasets.Dataset`s of (image, segmentation) pairs.

    For `mode == "train"` both a train and an eval (val) split are built from
    `img_dir`/{train,val}. For eval-only modes, only the eval dataset is built,
    either from `img_dir` directly or - if segmentation_guided and no img_dir -
    from `seg_dir` alone (mask-only conditioning).
    """
    seg_types = [""]

    if segmentation_guided:
        seg_paths_train, seg_paths_eval = {}, {}
        dataset_train = None

        if mode == "train":
            img_dir_train, img_dir_eval = os.path.join(img_dir, "train"), os.path.join(img_dir, "val")
            img_paths_train, img_paths_eval = _list_dir(img_dir_train), _list_dir(img_dir_eval)
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(img_dir_train)]
                seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, "val", f) for f in os.listdir(img_dir_eval)]

            dset_dict_train = {
                "image": img_paths_train,
                **{f"seg_{t}": seg_paths_train[t] for t in seg_types},
            }
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train[f"seg_{seg_types[0]}"]]
            dataset_train = datasets.Dataset.from_dict(dset_dict_train).cast_column("image", datasets.Image())
            for seg_type in seg_types:
                dataset_train = dataset_train.cast_column(f"seg_{seg_type}", datasets.Image())
        else:
            img_paths_eval = _list_dir(img_dir) if img_dir else []
            for seg_type in seg_types:
                seg_paths_eval[seg_type] = _list_dir(os.path.join(seg_dir, seg_type))

        dset_dict_eval = {
            **({"image": img_paths_eval} if img_dir else {}),
            **{f"seg_{t}": seg_paths_eval[t] for t in seg_types},
        }
        dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval[f"seg_{seg_types[0]}"]]
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)
        if img_dir:
            dataset_eval = dataset_eval.cast_column("image", datasets.Image())
        for seg_type in seg_types:
            dataset_eval = dataset_eval.cast_column(f"seg_{seg_type}", datasets.Image())
    else:
        img_dir_train, img_dir_eval = os.path.join(img_dir, "train"), os.path.join(img_dir, "val")
        img_paths_train, img_paths_eval = _list_dir(img_dir_train), _list_dir(img_dir_eval)

        dset_dict_train = {"image": img_paths_train, "image_filename": [os.path.basename(f) for f in img_paths_train]}
        dset_dict_eval = {"image": img_paths_eval, "image_filename": [os.path.basename(f) for f in img_paths_eval]}

        dataset_train = datasets.Dataset.from_dict(dset_dict_train).cast_column("image", datasets.Image())
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval).cast_column("image", datasets.Image())

    return dataset_train, dataset_eval


def attach_transforms(dataset_train, dataset_eval, image_size, segmentation_guided, has_images):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if segmentation_guided:
        preprocess_segmentation = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        def transform(examples):
            out = {"image_filenames": examples["image_filename"]}
            if has_images:
                out["images"] = [preprocess(image.convert("L")) for image in examples["image"]]
            for key in examples:
                if key.startswith("seg_"):
                    out[key] = [preprocess_segmentation(image.convert("L")) for image in examples[key]]
            return out
    else:
        def transform(examples):
            return {
                "images": [preprocess(image.convert("L")) for image in examples["image"]],
                "image_filenames": examples["image_filename"],
            }

    if dataset_train is not None:
        dataset_train.set_transform(transform)
    if dataset_eval is not None:
        dataset_eval.set_transform(transform)


def build_dataloaders(mode, dataset_train, dataset_eval, train_batch_size, eval_batch_size, image_size,
                       eval_shuffle_dataloader=True):
    train_dataloader = None
    if mode == "train":
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset_eval, batch_size=eval_batch_size,
                                                        shuffle=eval_shuffle_dataloader)
    elif dataset_eval is not None:
        eval_dataloader = torch.utils.data.DataLoader(dataset_eval, batch_size=eval_batch_size,
                                                        shuffle=eval_shuffle_dataloader)
    else:
        # placeholder dataloader to iterate through when sampling from a fully unconditional model
        eval_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(eval_batch_size, 1, image_size, image_size)),
            batch_size=eval_batch_size, shuffle=eval_shuffle_dataloader)

    return train_dataloader, eval_dataloader
