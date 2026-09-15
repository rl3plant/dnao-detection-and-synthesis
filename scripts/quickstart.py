"""End-to-end smoke test of the whole pipeline, using only what's already in this repo.

Part A - detection: reads the example raw AFM scan in examples/, preprocesses it,
and runs shape detection for all three shapes, writing results to out/detection/.

Part B - synthesis: takes a small subset of the already-published rectangle dataset
in data/extracted/rectangles/, patches it, and runs a *tiny* (CPU-feasible, a few
seconds) DDPM training + sampling run, writing results to out/synthesis/. This is
just enough to prove the synthesis code runs end to end - it is nowhere near enough
data/epochs to produce a good generator. For a real training run, see README.md.

Usage: python scripts/quickstart.py [--skip-synthesis]
"""
import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_detection(out_dir: Path):
    from dnao.detection import preprocess_directory, detect_directory

    print("\n=== Part A: detection ===")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    example = next((REPO_ROOT / "examples").glob("*.spm"))
    shutil.copy(example, raw_dir / example.name)

    preprocessed_dir = out_dir / "preprocessed"
    print(f"preprocessing {example.name} ...")
    preprocess_directory(str(raw_dir), str(preprocessed_dir), image_size=1024)

    for shape in ("rectangles", "triangles", "z_shapes"):
        target = out_dir / "extracted" / shape
        print(f"detecting {shape} ...")
        detect_directory(str(preprocessed_dir), str(target), shape)
        n = len(list((target / "images").glob("*.png")))
        print(f"  -> {target} ({n} image(s))")


def run_synthesis(out_dir: Path):
    from dnao.patch_extraction import prepare_patch_dataset
    from dnao.run_synthesis import run_synthesis as run_synthesis_pipeline

    print("\n=== Part B: synthesis (tiny CPU smoke test) ===")
    subset_dir = out_dir / "subset"
    (subset_dir / "images").mkdir(parents=True, exist_ok=True)
    (subset_dir / "masks").mkdir(parents=True, exist_ok=True)

    src = REPO_ROOT / "data" / "extracted" / "rectangles"
    files = sorted((src / "images").glob("*.png"))[:6]
    assert files, f"no images found under {src}/images"
    for f in files:
        shutil.copy(f, subset_dir / "images" / f.name)
        shutil.copy(src / "masks" / f.name, subset_dir / "masks" / f.name)

    patched_dir = out_dir / "patched"
    print(f"patching {len(files)} images into {patched_dir} ...")
    prepare_patch_dataset(str(subset_dir), str(patched_dir), tile_shape=(128, 128),
                           val_split=0.2, test_split=0.2, patches_per_image=4, patch_extraction="random")

    dataset_name = "quickstart"
    img_dir, seg_dir = patched_dir / "images", patched_dir / "masks"

    print("training for 1 epoch (this is a smoke test, not a real training run) ...")
    run_synthesis_pipeline(
        mode="train", img_size=32, dataset=dataset_name, img_dir=str(img_dir), seg_dir=str(seg_dir),
        model_type="DDIM", segmentation_guided=True, num_segmentation_classes=2,
        train_batch_size=2, eval_batch_size=2, num_epochs=1)

    print("sampling from the trained checkpoint ...")
    run_synthesis_pipeline(
        mode="eval", img_size=32, dataset=dataset_name, img_dir=str(img_dir / "val"), seg_dir=str(seg_dir / "val"),
        model_type="DDIM", segmentation_guided=True, num_segmentation_classes=2,
        train_batch_size=0, eval_batch_size=2, num_epochs=0, resume_epoch=1)

    samples_dir = Path(f"models/{('ddim-' + dataset_name)}/samples")
    print(f"  -> generated samples in {samples_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="out/quickstart")
    parser.add_argument("--skip-synthesis", action="store_true",
                         help="only run the detection half (skip torch/diffusers)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_detection(out_dir)
    if not args.skip_synthesis:
        run_synthesis(out_dir)

    print("\nQuickstart finished OK.")


if __name__ == "__main__":
    main()
