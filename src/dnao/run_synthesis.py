"""Orchestrates dataset loading, model construction and train/eval/eval_many modes
for the segmentation-guided DDPM. Ported from ``code/cond_ddpm/main.py::run_main``."""
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

from dnao.ddpm_config import TrainingConfig
from dnao.ddpm_dataset import build_datasets, attach_transforms, build_dataloaders
from dnao.ddpm_model import build_unet, build_noise_scheduler
from dnao.ddpm_training import train_loop
from dnao.ddpm_sampling import evaluate_generation, evaluate_sample_many


def run_synthesis(
        mode,
        img_size,
        dataset,
        img_dir,
        seg_dir,
        model_type,
        segmentation_guided,
        num_segmentation_classes,
        train_batch_size,
        eval_batch_size,
        num_epochs,
        resume_epoch=None,
        eval_shuffle_dataloader=True,
        eval_blank_mask=False,
        eval_sample_size=1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = f"models/{model_type.lower()}-{dataset}"
    if segmentation_guided:
        assert seg_dir is not None, "must provide segmentation directory for segmentation-guided training/sampling"
    if eval_blank_mask:
        output_dir += "-blank-mask"
    if mode == "train":
        assert img_dir is not None, "must provide image directory for training"

    config = TrainingConfig(
        image_size=img_size, dataset=dataset, segmentation_guided=segmentation_guided,
        num_segmentation_classes=num_segmentation_classes, train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size, num_epochs=num_epochs, output_dir=output_dir,
        model_type=model_type, resume_epoch=resume_epoch,
    )

    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert config.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)"

    dataset_train, dataset_eval = build_datasets(mode, img_dir, seg_dir, segmentation_guided)
    has_images = img_dir is not None
    attach_transforms(dataset_train, dataset_eval, config.image_size, segmentation_guided, has_images)
    train_dataloader, eval_dataloader = build_dataloaders(
        mode, dataset_train, dataset_eval, config.train_batch_size, config.eval_batch_size, config.image_size,
        eval_shuffle_dataloader)

    model = build_unet(config, mode, resume_epoch)
    model.to(device)
    noise_scheduler = build_noise_scheduler(model_type)

    if mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * config.num_epochs)
        train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler,
                   device=device)
    elif mode == "eval":
        evaluate_generation(config, model, noise_scheduler, eval_dataloader, eval_blank_mask=eval_blank_mask,
                             device=device)
    elif mode == "eval_many":
        evaluate_sample_many(eval_sample_size, config, model, noise_scheduler, eval_dataloader, device=device)
    else:
        raise ValueError(f'mode "{mode}" not supported.')
