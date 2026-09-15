# DNAO Detection and Synthesis

Classical computer-vision shape detection and mask-conditioned diffusion
image synthesis for AFM scans of DNA origami. From Noah Ruhmer's MSc thesis
(TU Graz, 2024).

- Detection: threshold -> contour -> idealized shape mask (rectangles,
  triangles, z-shapes).
- Synthesis: segmentation-guided DDPM/DDIM trained on the detected masks.

## Install

```bash
pip install -e .
pip install --no-deps pySPM        # raw .spm scan support
pip install -e ".[synthesis]"      # DDPM training/sampling (pulls in torch)
```

## Run

```bash
python scripts/quickstart.py
```

Or use the pipeline directly:

```bash
dnao preprocess <raw_dir> <out_dir>
dnao detect <preprocessed_dir> <out_dir> --shape rectangles
dnao prepare <detected_dir> <out_dir>
dnao train --dataset NAME --img-dir <dir> --seg-dir <dir> --num-segmentation-classes 2
dnao sample --dataset NAME --img-dir <dir> --seg-dir <dir> --num-segmentation-classes 2 --resume-epoch N
```

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
