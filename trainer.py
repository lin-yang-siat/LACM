# -*- coding: utf-8 -*-
"""Training and evaluation utilities for LACM."""
from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from data import CaseRecord, decode_label, load_case_arrays
from model import LACMNet


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice and cross-entropy loss for multi-class segmentation."""

    def __init__(
        self,
        n_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().clamp_(0, self.n_classes - 1)
        ce = F.cross_entropy(logits, target)

        target_one_hot = F.one_hot(target, num_classes=self.n_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).contiguous().float()
        start = 0 if self.include_background else 1
        prob_fg = prob[:, start:]
        target_fg = target_one_hot[:, start:]

        dims = (0, 2, 3, 4)
        intersection = (prob_fg * target_fg).sum(dim=dims)
        denominator = prob_fg.sum(dim=dims) + target_fg.sum(dim=dims)
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def dice_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    n_classes: int,
    include_background: bool = False,
) -> np.ndarray:
    start = 0 if include_background else 1
    scores = np.full(n_classes, np.nan, dtype=np.float64)
    for cls in range(start, n_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator == 0:
            continue
        scores[cls] = 2.0 * np.logical_and(pred_mask, target_mask).sum() / denominator
    return scores


def nanmean(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def nanmean_axis0(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    valid = np.sum(~np.isnan(array), axis=0)
    sums = np.nansum(array, axis=0)
    out = np.full(array.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, valid, out=out, where=valid > 0)
    return out


def make_grad_scaler(enabled: bool, device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, device_type: str):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def compute_sliding_starts(size: int, patch: int, step: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, max(1, step)))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_for_sliding_window(
    image: torch.Tensor,
    initial_mask: torch.Tensor,
    patch_size: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    spatial_shape = tuple(int(v) for v in image.shape[2:])
    padded_shape = tuple(max(size, int(patch)) for size, patch in zip(spatial_shape, patch_size))
    pad_after = [padded - size for size, padded in zip(spatial_shape, padded_shape)]
    if any(v > 0 for v in pad_after):
        pad = (0, pad_after[2], 0, pad_after[1], 0, pad_after[0])
        image = F.pad(image, pad, value=0)
        initial_mask = F.pad(initial_mask.unsqueeze(1).float(), pad, value=0).squeeze(1).long()
    return image, initial_mask, spatial_shape


@torch.no_grad()
def sliding_window_inference(
    model: LACMNet,
    image: torch.Tensor,
    initial_mask: torch.Tensor,
    patch_size: Sequence[int],
    patch_step: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    image = image.to(device, non_blocking=True)
    initial_mask = initial_mask.to(device, non_blocking=True)
    image, initial_mask, original_shape = pad_for_sliding_window(image, initial_mask, patch_size)
    _, _, depth, height, width = image.shape
    pd, ph, pw = (int(v) for v in patch_size)
    sd, sh, sw = (int(v) for v in patch_step)

    prob_sum = torch.zeros((1, model.n_classes, depth, height, width), dtype=torch.float32)
    count = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)

    for d0 in compute_sliding_starts(depth, pd, sd):
        for h0 in compute_sliding_starts(height, ph, sh):
            for w0 in compute_sliding_starts(width, pw, sw):
                ds = slice(d0, d0 + pd)
                hs = slice(h0, h0 + ph)
                ws = slice(w0, w0 + pw)
                out = model(image[:, :, ds, hs, ws], initial_mask[:, ds, hs, ws])
                prob_sum[:, :, ds, hs, ws] += out["prob"].detach().cpu()
                count[:, :, ds, hs, ws] += 1.0

    prob = prob_sum / count.clamp_min(1.0)
    od, oh, ow = original_shape
    return prob[:, :, :od, :oh, :ow]


class LACMTrainer:
    """Trainer for nnUNet-initialized LACM refinement."""

    def __init__(self, args, train_loader, test_cases: Sequence[CaseRecord], label_values: Sequence[int]):
        self.args = args
        self.train_loader = train_loader
        self.test_cases = list(test_cases)
        self.label_values = list(label_values)
        self.device = torch.device(args.device)

        self.n_classes = int(args.classes)
        self.model = LACMNet(
            n_iterations=args.models,
            n_classes=self.n_classes,
            image_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            region_sigma=args.local_sigma,
            region_kernel_radius=args.local_kernel_radius,
            regularization_sigma=args.sigma,
            regularization_kernel_radius=args.kernel_radius,
            initial_lambda=args.init_lambda,
            lambda_values=args.lambda_values,
            initial_epsilon=args.init_epsilon,
            update_source=args.term_source,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.criterion = DiceCrossEntropyLoss(
            n_classes=self.n_classes,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            include_background=args.include_background,
        )
        self.scaler = make_grad_scaler(args.amp and self.device.type == "cuda", self.device.type)

        self.save_path = Path(args.save_path)
        self.save_path_nii = Path(args.save_path_nii)
        self.metrics_path = Path(args.metrics_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)

    def lambda_summary(self) -> tuple[float, float, float]:
        values = self.model.lambda_weight.detach().cpu().flatten().numpy()
        return float(values.mean()), float(values.min()), float(values.max())

    def save_checkpoint(self, epoch: int, val_mean_dice: float | None = None) -> None:
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "label_values": self.label_values,
            "val_mean_dice": val_mean_dice,
            "args": vars(self.args),
        }
        path = self.save_path / f"Params_{epoch}_epoch.ckpt"
        torch.save(checkpoint, path)
        print(f"[Save] epoch {epoch}: {path}")

    def load_checkpoint(self, epoch: int) -> None:
        path = self.save_path / f"Params_{epoch}_epoch.ckpt"
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint and self.args.resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[Load] epoch {epoch}: {path}")

    def decay_learning_rate(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.args.lr_decay_gamma
        print(f"[LR] current lr: {self.optimizer.param_groups[0]['lr']:.6g}")

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.num_epochs}", mininterval=0.3)
        for image, label, initial_mask in progress:
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            initial_mask = initial_mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.args.amp and self.device.type == "cuda", self.device.type):
                out = self.model(image, initial_mask)
                loss = self.criterion(out["logits"], out["prob"], label)

            self.scaler.scale(loss).backward()
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(float(loss.detach().cpu()))
            lam_mean, lam_min, lam_max = self.lambda_summary()
            progress.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                lam=f"{lam_mean:.4f}/{lam_min:.4f}-{lam_max:.4f}",
                eps=f"{float(self.model.epsilon.detach().cpu()):.4f}",
            )
        return float(np.mean(losses)) if losses else float("nan")

    def save_prediction(
        self,
        pred: np.ndarray,
        affine: np.ndarray,
        header: nib.Nifti1Header,
        case_id: str,
        epoch: int | None,
    ) -> None:
        save_dir = self.save_path_nii / (f"epoch_{epoch:03d}" if epoch is not None else "test")
        save_dir.mkdir(parents=True, exist_ok=True)
        raw_pred = decode_label(pred, self.label_values).astype(np.int16)
        out_header = header.copy()
        out_header.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(raw_pred, affine, out_header), str(save_dir / f"{case_id}.nii.gz"))

    def write_metrics_csv(self, epoch: int | None, rows: list[dict[str, object]]) -> None:
        suffix = f"epoch_{epoch:03d}" if epoch is not None else "test"
        path = self.metrics_path / f"metrics_{suffix}.csv"
        fieldnames = ["epoch", "case_id", "class", "label_value", "lacm_dice", "nnunet_dice", "delta"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Metrics] {path}")

    def evaluate(self, epoch: int | None = None) -> tuple[float, float]:
        if not self.test_cases:
            print("[Eval] no test cases; skipped")
            return float("nan"), float("nan")

        print(f"[Eval] Start validation on {len(self.test_cases)} cases")
        all_lacm: list[np.ndarray] = []
        all_nnunet: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        t0 = time.time()

        for case in self.test_cases:
            image_np, label_np, initial_np, affine, header = load_case_arrays(
                case,
                self.label_values,
                normalize=self.args.normalize,
            )
            if label_np is None:
                continue

            image = torch.from_numpy(np.ascontiguousarray(image_np)).unsqueeze(0).float()
            initial_mask = torch.from_numpy(np.ascontiguousarray(initial_np)).unsqueeze(0).long()
            prob = sliding_window_inference(
                self.model,
                image,
                initial_mask,
                patch_size=self.args.val_patch,
                patch_step=self.args.val_patch_step,
                device=self.device,
            )
            pred = prob.argmax(dim=1)[0].numpy().astype(np.int16)

            lacm_scores = dice_per_class(
                pred,
                label_np,
                self.n_classes,
                include_background=self.args.include_background,
            )
            nnunet_scores = dice_per_class(
                initial_np,
                label_np,
                self.n_classes,
                include_background=self.args.include_background,
            )
            all_lacm.append(lacm_scores)
            all_nnunet.append(nnunet_scores)

            case_lacm = nanmean(lacm_scores)
            case_nnunet = nanmean(nnunet_scores)
            print(
                f"[Eval] {case.case_id}: LACM Dice={case_lacm:.4f}, "
                f"nnUNet Dice={case_nnunet:.4f}, delta={case_lacm - case_nnunet:+.4f}"
            )

            for cls in range(self.n_classes):
                if not self.args.include_background and cls == 0:
                    continue
                rows.append(
                    {
                        "epoch": epoch if epoch is not None else "test",
                        "case_id": case.case_id,
                        "class": cls,
                        "label_value": self.label_values[cls],
                        "lacm_dice": lacm_scores[cls],
                        "nnunet_dice": nnunet_scores[cls],
                        "delta": lacm_scores[cls] - nnunet_scores[cls],
                    }
                )

            if self.args.is_save:
                self.save_prediction(pred, affine, header, case.case_id, epoch)

        self.write_metrics_csv(epoch, rows)
        lacm_array = np.vstack(all_lacm)
        nnunet_array = np.vstack(all_nnunet)
        per_class_lacm = nanmean_axis0(lacm_array)
        per_class_nnunet = nanmean_axis0(nnunet_array)
        mean_lacm = nanmean(per_class_lacm)
        mean_nnunet = nanmean(per_class_nnunet)

        print("[Eval] Per-class Dice:")
        for cls in range(self.n_classes):
            if not self.args.include_background and cls == 0:
                continue
            print(
                f"  class {cls} (label {self.label_values[cls]}): "
                f"LACM={per_class_lacm[cls]:.4f}, "
                f"nnUNet={per_class_nnunet[cls]:.4f}, "
                f"delta={per_class_lacm[cls] - per_class_nnunet[cls]:+.4f}"
            )

        elapsed = time.time() - t0
        print(
            f"[Eval] Mean Dice: LACM={mean_lacm:.4f}, nnUNet={mean_nnunet:.4f}, "
            f"delta={mean_lacm - mean_nnunet:+.4f}, time={elapsed:.1f}s"
        )
        return mean_lacm, mean_nnunet

    def run(self) -> None:
        if self.args.is_load:
            self.load_checkpoint(self.args.load_epoch)

        if not self.args.is_train:
            if self.args.is_test:
                self.model.eval()
                self.evaluate(None)
            return

        best_dice = -1.0
        start_epoch = self.args.load_epoch + 1 if self.args.is_load and self.args.continue_epoch else 1
        end_epoch = start_epoch + self.args.num_epochs - 1

        for epoch in range(start_epoch, end_epoch + 1):
            val_dice = None
            train_loss = self.train_one_epoch(epoch)
            print(f"[Train] epoch {epoch}: loss={train_loss:.5f}")
            if epoch > self.args.decay_start_epoch and epoch % self.args.decay_epoch == 0:
                self.decay_learning_rate()

            if self.args.is_test and epoch % self.args.test_num == 0:
                self.model.eval()
                val_dice, _ = self.evaluate(epoch)
                if val_dice == val_dice and val_dice > best_dice:
                    best_dice = val_dice
                    self.save_checkpoint(epoch, val_mean_dice=val_dice)

            if self.args.save_every > 0 and epoch % self.args.save_every == 0:
                self.save_checkpoint(epoch, val_mean_dice=val_dice)
