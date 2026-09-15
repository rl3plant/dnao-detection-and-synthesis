"""Command-line entry point for the DNAO detection + synthesis pipeline.

    dnao preprocess  <raw_dir> <preprocessed_dir> [--image-size 1024]
    dnao detect      <preprocessed_dir> <extracted_dir> --shape rectangles|triangles|z_shapes
    dnao prepare     <extracted_shape_dir> <patched_dir> [--tile-size 128] [--val-split 0.05] [--test-split 0.1]
    dnao train       --dataset NAME --img-dir DIR --seg-dir DIR --num-segmentation-classes 2 [...]
    dnao sample      --dataset NAME --seg-dir DIR --num-segmentation-classes 2 --resume-epoch N [...]
"""
import argparse

from dnao.detection import preprocess_directory, detect_directory
from dnao.patch_extraction import prepare_patch_dataset
from dnao.run_synthesis import run_synthesis


def _add_synthesis_args(parser, mode):
    parser.add_argument('--dataset', required=True, help='name for this model/dataset (used to name the output dir)')
    parser.add_argument('--img-dir', default=None, help='directory with train/ and val/ subfolders of clean images')
    parser.add_argument('--seg-dir', required=True, help='directory with the matching segmentation masks')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--model-type', choices=['DDPM', 'DDIM'], default='DDIM')
    parser.add_argument('--num-segmentation-classes', type=int, required=True,
                         help='number of segmentation classes, including background')
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--resume-epoch', type=int, default=None)
    if mode == 'train':
        parser.add_argument('--train-batch-size', type=int, default=32)
        parser.add_argument('--num-epochs', type=int, default=200)
    else:
        parser.add_argument('--eval-sample-size', type=int, default=1000)
        parser.add_argument('--eval-many', action='store_true', help='sample many images instead of one preview grid')
        parser.add_argument('--eval-blank-mask', action='store_true')


def build_parser():
    parser = argparse.ArgumentParser(prog='dnao', description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('preprocess', help='raw AFM scans -> cleaned/normalized PNGs')
    p.add_argument('raw_dir')
    p.add_argument('target_dir')
    p.add_argument('--image-size', type=int, default=1024)

    p = sub.add_parser('detect', help='preprocessed images -> shape contours + idealized masks')
    p.add_argument('src_dir')
    p.add_argument('target_dir')
    p.add_argument('--shape', required=True, choices=['rectangles', 'triangles', 'z_shapes'])

    p = sub.add_parser('prepare', help='detected images+masks -> patched train/val/test dataset for synthesis')
    p.add_argument('src_dir', help='directory with images/ and masks/ subfolders (e.g. an extracted/<shape> dir)')
    p.add_argument('target_dir')
    p.add_argument('--tile-size', type=int, default=128)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--test-split', type=float, default=0.1)
    p.add_argument('--patches-per-image', type=int, default=32)
    p.add_argument('--patch-extraction', choices=['random', 'tiled'], default='random')
    p.add_argument('--augment', action='store_true')

    p = sub.add_parser('train', help='train the segmentation-guided DDPM/DDIM image generator')
    _add_synthesis_args(p, 'train')

    p = sub.add_parser('sample', help='sample images from a trained generator')
    _add_synthesis_args(p, 'sample')

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == 'preprocess':
        preprocess_directory(args.raw_dir, args.target_dir, args.image_size)
    elif args.command == 'detect':
        detect_directory(args.src_dir, args.target_dir, args.shape)
    elif args.command == 'prepare':
        prepare_patch_dataset(
            args.src_dir, args.target_dir, (args.tile_size, args.tile_size), args.val_split, args.test_split,
            args.patches_per_image, args.patch_extraction, args.augment)
    elif args.command == 'train':
        run_synthesis(
            'train', args.img_size, args.dataset, args.img_dir, args.seg_dir, args.model_type, True,
            args.num_segmentation_classes, args.train_batch_size, args.eval_batch_size, args.num_epochs)
    elif args.command == 'sample':
        run_synthesis(
            'eval_many' if args.eval_many else 'eval', args.img_size, args.dataset, args.img_dir, args.seg_dir,
            args.model_type, True, args.num_segmentation_classes, 0, args.eval_batch_size, 0, args.resume_epoch,
            eval_blank_mask=args.eval_blank_mask, eval_sample_size=args.eval_sample_size)


if __name__ == '__main__':
    main()
