"""Training loop for the segmentation-guided DDPM/DDIM. Ported from
``code/cond_ddpm/training.py``."""
import os

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dnao.ddpm_sampling import evaluate, add_segmentations_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler,
               device='cuda'):
    global_step = 0

    run_name = f"models/{config.model_type.lower()}-{config.dataset}-{config.image_size}"
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "runs"), comment=run_name)

    iter_eval_dataloader = iter(eval_dataloader)

    start_epoch = config.resume_epoch if config.resume_epoch is not None else 0

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(device)

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),
                                       device=clean_images.device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(noisy_images, batch, device)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            writer.add_scalar("loss", loss.detach().item(), global_step)
            progress_bar.set_postfix(**logs)
            global_step += 1

        is_image_epoch = (epoch + 1) % config.save_image_epochs == 0
        is_model_epoch = (epoch + 1) % config.save_model_epochs == 0
        is_last_epoch = (epoch + 1) == config.num_epochs

        if is_image_epoch or is_model_epoch or is_last_epoch:
            if config.model_type == "DDPM":
                pipeline_cls = SegGuidedDDPMPipeline if config.segmentation_guided else None
                pipeline = (pipeline_cls(unet=model.module, scheduler=noise_scheduler,
                                          eval_dataloader=iter_eval_dataloader, external_config=config)
                            if pipeline_cls else diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler))
            elif config.model_type == "DDIM":
                pipeline_cls = SegGuidedDDIMPipeline if config.segmentation_guided else None
                pipeline = (pipeline_cls(unet=model.module, scheduler=noise_scheduler,
                                          eval_dataloader=iter_eval_dataloader, external_config=config)
                            if pipeline_cls else diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler))
            else:
                raise ValueError(f"model_type must be 'DDPM' or 'DDIM', got {config.model_type!r}")

            model.eval()

            if is_image_epoch or is_model_epoch:
                if config.segmentation_guided:
                    try:
                        seg_batch = next(iter_eval_dataloader)
                    except StopIteration:
                        iter_eval_dataloader = iter(eval_dataloader)
                        seg_batch = next(iter_eval_dataloader)
                    evaluate(config, epoch + 1, pipeline, seg_batch)
                else:
                    evaluate(config, epoch + 1, pipeline)

            if is_model_epoch or is_last_epoch:
                pipeline.save_pretrained(config.output_dir + f"/epochs_{epoch + 1}", safe_serialization=True)
