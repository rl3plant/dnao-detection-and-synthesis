"""U-Net construction and checkpoint loading for the segmentation-guided DDPM."""
import os

import diffusers
from torch import nn


def build_unet(config, mode, resume_epoch=None):
    in_channels = 2 if config.segmentation_guided else 1

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,
        in_channels=in_channels,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )

    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        if mode == "train":
            print(f"--- resuming from model at training epoch {resume_epoch}")
        model = model.from_pretrained(os.path.join(config.output_dir + f"/epochs_{resume_epoch}", "unet"),
                                       use_safetensors=True)

    # overwrite the loaded model's sample size with the currently configured one
    model.config.sample_size = config.image_size
    return nn.DataParallel(model)


def build_noise_scheduler(model_type):
    if model_type == "DDPM":
        return diffusers.DDPMScheduler(num_train_timesteps=1000)
    if model_type == "DDIM":
        return diffusers.DDIMScheduler(num_train_timesteps=1000)
    raise ValueError(f"model_type must be 'DDPM' or 'DDIM', got {model_type!r}")
