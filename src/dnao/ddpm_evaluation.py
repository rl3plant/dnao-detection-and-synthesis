"""Synthetic-vs-real image quality metrics (FID, SSIM) for the trained generator."""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def calc_fid(synth_data, real_data, feature=64):
    fid = FrechetInceptionDistance(feature=feature)
    fid.update(synth_data, real=False)
    fid.update(real_data, real=True)
    return fid.compute()


def calc_ssim(synth_data, real_data):
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0))
    return ssim(synth_data, real_data)


def eval_datasets(synth_data, real_data):
    synth_data = torch.tensor(synth_data)[:, None, ...].expand(-1, 3, -1, -1)
    real_data = torch.tensor(real_data)[:, None, ...].expand(-1, 3, -1, -1)

    fid = calc_fid(synth_data, real_data)
    ssim = calc_ssim(synth_data, real_data)
    return fid, ssim
