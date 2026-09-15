"""Segmentation-guided sampling: custom diffusers pipelines plus the evaluate/
evaluate_generation/evaluate_sample_many entry points used by training and the CLI.

Ported from the thesis's ``code/cond_ddpm/eval.py``.
"""
import math
import os
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

import diffusers
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchvision.utils import save_image

from dnao.ddpm_utils import make_grid


def convert_segbatch_to_multiclass(imgs_shape, segmentations_batch, device):
    # NOTE: assumes that segs don't overlap - all segs are put on the same channel
    segs = torch.zeros(imgs_shape).to(device)
    for k, seg in segmentations_batch.items():
        if k.startswith("seg_"):
            seg = seg.to(device)
            segs[segs == 0] = seg[segs == 0]
    return segs


def add_segmentations_to_noise(noisy_images, segmentations_batch, device):
    """Concatenate the segmentation-mask channel onto the (noisy) image channel."""
    segs = convert_segbatch_to_multiclass(noisy_images.shape, segmentations_batch, device)
    return torch.cat((noisy_images, segs), dim=1)


def evaluate(config, epoch, pipeline, seg_batch=None):
    if config.segmentation_guided:
        images = pipeline(batch_size=config.eval_batch_size, seg_batch=seg_batch).images
    else:
        images = pipeline(batch_size=config.eval_batch_size).images

    cols = 4
    rows = math.ceil(len(images) / cols)
    image_grid = make_grid(images, rows=rows, cols=cols)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    if config.segmentation_guided:
        for seg_type in seg_batch:
            if seg_type.startswith("seg_"):
                save_image(seg_batch[seg_type], f"{test_dir}/{epoch:04d}_cond_{seg_type}.png", normalize=True, nrow=cols)
        save_image(seg_batch["images"], f"{test_dir}/{epoch:04d}_orig.png", normalize=True, nrow=cols)


def _build_pipeline(config, model, noise_scheduler, eval_dataloader):
    if config.model_type == "DDPM":
        if config.segmentation_guided:
            return SegGuidedDDPMPipeline(unet=model.module, scheduler=noise_scheduler,
                                          eval_dataloader=eval_dataloader, external_config=config)
        return diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
    if config.model_type == "DDIM":
        if config.segmentation_guided:
            return SegGuidedDDIMPipeline(unet=model.module, scheduler=noise_scheduler,
                                          eval_dataloader=eval_dataloader, external_config=config)
        return diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)
    raise ValueError(f"model_type must be 'DDPM' or 'DDIM', got {config.model_type!r}")


def evaluate_generation(config, model, noise_scheduler, eval_dataloader, eval_blank_mask=False, device='cuda'):
    """Sample once from a (possibly mask-guided) trained model and save a preview grid."""
    eval_dataloader = iter(eval_dataloader)
    seg_batch = None

    if config.segmentation_guided:
        seg_batch = next(eval_dataloader)
        if eval_blank_mask:
            for k, v in seg_batch.items():
                if k.startswith("seg_"):
                    seg_batch[k] = torch.zeros_like(v)

    pipeline = _build_pipeline(config, model, noise_scheduler, eval_dataloader)
    evaluate(config, -1, pipeline, seg_batch)


def evaluate_sample_many(sample_size, config, model, noise_scheduler, eval_dataloader, device='cuda'):
    """Sample `sample_size` images (mask-guided if configured) and save each individually."""
    pipeline = _build_pipeline(config, model, noise_scheduler, eval_dataloader)

    sample_dir = os.path.join(config.output_dir, f"samples_epoch_{config.resume_epoch}_size_{config.image_size}")
    os.makedirs(sample_dir, exist_ok=True)

    num_sampled = 0
    for bidx, seg_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        if num_sampled >= sample_size:
            break

        if config.segmentation_guided:
            current_batch_size = next(v for k, v in seg_batch.items() if k.startswith("seg_")).shape[0]
            images = pipeline(batch_size=current_batch_size, seg_batch=seg_batch).images
        else:
            images = pipeline(batch_size=config.eval_batch_size).images

        for i, img in enumerate(images):
            if config.segmentation_guided:
                img_fname = f"{sample_dir}/{seg_batch['image_filenames'][i]}"
            else:
                img_fname = f"{sample_dir}/{num_sampled + i:04d}.png"
            img.save(img_fname)

        num_sampled += len(images)


####################
# custom diffusers pipelines for segmentation-guided sampling
####################

class SegGuidedDDPMPipeline(DiffusionPipeline):
    r"""Segmentation-guided image generation, modified from `diffusers.DDPMPipeline`.

    Args:
        unet ([`UNet2DModel`]): denoises the (image + segmentation) latents.
        scheduler: a `DDPMScheduler` or `DDIMScheduler`.
        eval_dataloader: yields batches of segmentations to condition generation on.
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.eval_dataloader = eval_dataloader
        self.external_config = external_config

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            seg_batch: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        img_channel_ct = self.unet.config.in_channels - 1

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (batch_size, img_channel_ct, self.unet.config.sample_size, self.unet.config.sample_size)
        else:
            image_shape = (batch_size, self.unet.config.in_channels - 1, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator).to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            image = add_segmentations_to_noise(image, seg_batch, self.device)
            model_output = self.unet(image, t).sample
            # only denoise the image channel, not the seg channel
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


class SegGuidedDDIMPipeline(DiffusionPipeline):
    r"""Segmentation-guided image generation, modified from `diffusers.DDIMPipeline`."""
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, eval_dataloader=eval_dataloader,
                               external_config=external_config)
        self.eval_dataloader = eval_dataloader
        self.external_config = external_config
        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            seg_batch: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        img_channel_ct = self.unet.config.in_channels - 1

        if seg_batch is not None and batch_size != len(seg_batch['images']):
            batch_size = len(seg_batch['images'])

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (batch_size, img_channel_ct, self.unet.config.sample_size, self.unet.config.sample_size)
        else:
            image_shape = (batch_size, self.unet.config.in_channels - 1, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators.")

        image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            image = add_segmentations_to_noise(image, seg_batch, self.device)
            model_output = self.unet(image, t).sample
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output,
                generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
