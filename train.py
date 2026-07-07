# -*- coding: utf-8 -*-
"""Train or evaluate LACM on nnUNet-style MR-PDFF datasets."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import NnUNetLACMPatchDataset, build_case_list, parse_label_values
from trainer import LACMTrainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def resolve_subdir(root: Path, requested: str, fallbacks: tuple[str, ...] = ()) -> str:
    if (root / requested).is_dir():
        return requested
    for fallback in fallbacks:
        if (root / fallback).is_dir():
            print(f"[Data] Folder '{requested}' not found, using '{fallback}' instead.")
            return fallback
    return requested


def build_data_loaders(args):
    root = Path(args.data_root)
    args.train_image_dir = resolve_subdir(root, args.train_image_dir)
    args.train_label_dir = resolve_subdir(root, args.train_label_dir)
    args.train_init_dir = resolve_subdir(root, args.train_init_dir)
    args.test_image_dir = resolve_subdir(root, args.test_image_dir)
    args.test_label_dir = resolve_subdir(root, args.test_label_dir, ("labelsTs_all", "labelsVal"))
    args.test_init_dir = resolve_subdir(root, args.test_init_dir, ("imferTs", "inferTs", "inferVal"))

    label_values = parse_label_values(args.label_values, args.classes)

    train_cases = []
    train_loader = []
    if args.is_train:
        train_cases = build_case_list(
            root=root,
            image_subdir=args.train_image_dir,
            label_subdir=args.train_label_dir,
            initial_mask_subdir=args.train_init_dir,
            require_label=True,
        )
        train_dataset = NnUNetLACMPatchDataset(
            cases=train_cases,
            label_values=label_values,
            patch_size=args.patch,
            patches_per_epoch=args.patches_per_epoch,
            foreground_probability=args.foreground_probability,
            normalize=args.normalize,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_cases = []
    if args.is_test:
        test_cases = build_case_list(
            root=root,
            image_subdir=args.test_image_dir,
            label_subdir=args.test_label_dir,
            initial_mask_subdir=args.test_init_dir,
            require_label=True,
        )

    reference_cases = train_cases if train_cases else test_cases
    if not reference_cases:
        raise RuntimeError("No train/test cases were built.")

    detected_channels = len(reference_cases[0].image_paths)
    if args.in_channels <= 0:
        args.in_channels = detected_channels
    elif args.in_channels != detected_channels:
        print(
            f"[Warn] --in_channels={args.in_channels}, but detected {detected_channels} "
            "image channels from nnUNet-style files."
        )

    print(f"[Data] train cases: {len(train_cases)}")
    print(f"[Data] test cases: {len(test_cases)}")
    print(f"[Data] image channels: {args.in_channels}")
    print(f"[Data] label values: {label_values}")
    return train_loader, test_cases, label_values


def main(args):
    if args.val_patch is None:
        args.val_patch = args.patch
    if args.val_patch_step is None:
        args.val_patch_step = args.patch_step
    args.patch = tuple(args.patch)
    args.patch_step = tuple(args.patch_step)
    args.val_patch = tuple(args.val_patch)
    args.val_patch_step = tuple(args.val_patch_step)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available; using CPU.")
        args.device = "cpu"

    train_loader, test_cases, label_values = build_data_loaders(args)
    trainer = LACMTrainer(args, train_loader, test_cases, label_values)
    trainer.run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learnable Active Contours Model for nnUNet-initialized MR-PDFF segmentation."
    )

    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--train_image_dir", type=str, default="imagesTr")
    parser.add_argument("--train_label_dir", type=str, default="labelsTr")
    parser.add_argument("--train_init_dir", type=str, default="inferTr")
    parser.add_argument("--test_image_dir", type=str, default="imagesTs")
    parser.add_argument("--test_label_dir", type=str, default="labelsTs")
    parser.add_argument("--test_init_dir", type=str, default="inferTs")
    parser.add_argument("--label_values", type=str, default=None)

    parser.add_argument("--classes", type=int, default=11)
    parser.add_argument("--in_channels", type=int, default=0)
    parser.add_argument("--patch", type=int, nargs=3, default=[256, 256, 48])
    parser.add_argument("--patch_step", type=int, nargs=3, default=[256, 256, 24])
    parser.add_argument("--val_patch", type=int, nargs=3, default=None)
    parser.add_argument("--val_patch_step", type=int, nargs=3, default=None)
    parser.add_argument("--patches_per_epoch", type=int, default=250)
    parser.add_argument("--foreground_probability", type=float, default=0.7)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--models", type=int, default=3, help="Number of unrolled LACM iterations.")
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--local_sigma", type=float, default=3.0)
    parser.add_argument("--local_kernel_radius", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0, help="Boundary regularization sigma.")
    parser.add_argument("--kernel_radius", type=int, default=3, help="Boundary regularization radius.")
    parser.add_argument("--init_lambda", type=float, default=0.035)
    parser.add_argument("--lambda_values", type=str, default=None)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--term_source", type=str, choices=["init", "iter"], default="init")

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--include_background", type=str2bool, default=False)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--decay_start_epoch", type=int, default=50)
    parser.add_argument("--decay_epoch", type=int, default=10)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.8)

    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--is_test", type=str2bool, default=True)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--is_save", type=str2bool, default=True)
    parser.add_argument("--is_load", type=str2bool, default=False)
    parser.add_argument("--load_epoch", type=int, default=1)
    parser.add_argument("--continue_epoch", type=str2bool, default=True)
    parser.add_argument("--resume_optimizer", type=str2bool, default=False)

    parser.add_argument("--save_path", type=str, default="runs/lacm_params")
    parser.add_argument("--save_path_nii", type=str, default="runs/lacm_predictions")
    parser.add_argument("--metrics_path", type=str, default="runs/lacm_metrics")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    start = time.time()
    main(args)
    print(f"total time is: {time.time() - start:.1f}s")
