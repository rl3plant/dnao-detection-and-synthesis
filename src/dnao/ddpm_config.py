"""Training/sampling configuration for the segmentation-guided DDPM/DDIM model."""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 200
    save_model_epochs: int = 200
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    num_segmentation_classes: int = None  # INCLUDING background
    dataset: str = None
    resume_epoch: int = None
