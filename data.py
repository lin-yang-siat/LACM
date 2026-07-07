# -*- coding: utf-8 -*-
"""nnUNet-style data loading utilities for LACM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


NIFTI_SUFFIXES = (".nii.gz", ".nii")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_paths: tuple[Path, ...]
    label_path: Path | None
    initial_mask_path: Path


def strip_nifti_suffix(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def case_id_from_image(path: Path) -> str:
    stem = strip_nifti_suffix(path)
    if len(stem) > 5 and stem[-5] == "_" and stem[-4:].isdigit():
        return stem[:-5]
    return stem


def list_nifti_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in NIFTI_SUFFIXES:
        files.extend(folder.glob(f"*{suffix}"))
    return sorted(set(files))


def find_exact_nifti(folder: Path, stem: str) -> Path | None:
    for suffix in NIFTI_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_case_images(image_dir: Path, case_id: str) -> tuple[Path, ...]:
    modality_files: list[Path] = []
    for suffix in NIFTI_SUFFIXES:
        modality_files.extend(image_dir.glob(f"{case_id}_*{suffix}"))

    def modality_key(path: Path) -> tuple[int, str]:
        stem = strip_nifti_suffix(path)
        if len(stem) > 5 and stem[-5] == "_" and stem[-4:].isdigit():
            return int(stem[-4:]), path.name
        return 9999, path.name

    modality_files = [
        path
        for path in modality_files
        if len(strip_nifti_suffix(path)) > 5
        and strip_nifti_suffix(path)[-5] == "_"
        and strip_nifti_suffix(path)[-4:].isdigit()
    ]
    modality_files = sorted(set(modality_files), key=modality_key)
    if modality_files:
        return tuple(modality_files)

    exact = find_exact_nifti(image_dir, case_id)
    if exact is not None:
        return (exact,)
    return tuple()


def build_case_list(
    root: str | Path,
    image_subdir: str,
    label_subdir: str | None,
    initial_mask_subdir: str,
    require_label: bool = True,
) -> list[CaseRecord]:
    root = Path(root)
    image_dir = root / image_subdir
    initial_mask_dir = root / initial_mask_subdir
    label_dir = root / label_subdir if label_subdir else None

    if not image_dir.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_dir}")
    if not initial_mask_dir.is_dir():
        raise FileNotFoundError(f"initial-mask folder not found: {initial_mask_dir}")
    if require_label and (label_dir is None or not label_dir.is_dir()):
        raise FileNotFoundError(f"label folder not found: {label_dir}")

    if label_dir is not None and label_dir.is_dir():
        case_ids = [strip_nifti_suffix(path) for path in list_nifti_files(label_dir)]
    else:
        case_ids = sorted({case_id_from_image(path) for path in list_nifti_files(image_dir)})

    cases: list[CaseRecord] = []
    missing: list[str] = []
    for case_id in case_ids:
        image_paths = find_case_images(image_dir, case_id)
        label_path = find_exact_nifti(label_dir, case_id) if label_dir is not None else None
        initial_mask_path = find_exact_nifti(initial_mask_dir, case_id)
        if not image_paths or initial_mask_path is None or (require_label and label_path is None):
            missing.append(case_id)
            continue
        cases.append(
            CaseRecord(
                case_id=case_id,
                image_paths=image_paths,
                label_path=label_path,
                initial_mask_path=initial_mask_path,
            )
        )

    if not cases:
        raise RuntimeError("No paired cases were found.")
    if missing:
        preview = ", ".join(missing[:5])
        print(f"[Data] Skipped {len(missing)} incomplete cases, e.g. {preview}")
    return cases


def parse_label_values(label_values: str | None, n_classes: int) -> list[int]:
    if label_values is None or label_values.strip() == "":
        return list(range(n_classes))
    values = [int(v.strip()) for v in label_values.split(",") if v.strip()]
    if len(values) != n_classes:
        raise ValueError(f"--label_values has {len(values)} values but --classes is {n_classes}")
    return values


def encode_label(array: np.ndarray, label_values: Sequence[int]) -> np.ndarray:
    rounded = np.rint(array).astype(np.int64)
    encoded = np.zeros(rounded.shape, dtype=np.int16)
    for encoded_value, raw_value in enumerate(label_values):
        encoded[rounded == int(raw_value)] = encoded_value
    return encoded


def decode_label(encoded: np.ndarray, label_values: Sequence[int]) -> np.ndarray:
    values = np.asarray(label_values, dtype=np.int16)
    encoded = np.asarray(encoded, dtype=np.int64)
    encoded = np.clip(encoded, 0, len(values) - 1)
    return values[encoded]


def normalize_image(image: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    image = image.astype(np.float32, copy=True)
    for channel in range(image.shape[0]):
        volume = image[channel]
        finite = np.isfinite(volume)
        if not np.any(finite):
            image[channel] = 0
            continue
        values = volume[finite]
        lo, hi = np.percentile(values, percentiles)
        if hi > lo:
            volume = np.clip(volume, lo, hi)
        mean = float(volume[finite].mean())
        std = float(volume[finite].std())
        image[channel] = (volume - mean) / max(std, 1e-6)
        image[channel][~finite] = 0
    return image


def load_image_stack(image_paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    arrays: list[np.ndarray] = []
    affine: np.ndarray | None = None
    header: nib.Nifti1Header | None = None
    shape: tuple[int, ...] | None = None
    for path in image_paths:
        nii = nib.load(str(path))
        data = nii.get_fdata(dtype=np.float32)
        if shape is None:
            shape = data.shape
            affine = nii.affine
            header = nii.header.copy()
        elif data.shape != shape:
            raise ValueError(f"image modalities for one case have different shapes: {path}")
        arrays.append(data.astype(np.float32, copy=False))
    if affine is None or header is None:
        raise RuntimeError("empty image stack")
    return np.stack(arrays, axis=0), affine, header


def load_case_arrays(
    case: CaseRecord,
    label_values: Sequence[int],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, nib.Nifti1Header]:
    image, affine, header = load_image_stack(case.image_paths)
    if normalize:
        image = normalize_image(image)

    label = None
    if case.label_path is not None:
        label_nii = nib.load(str(case.label_path))
        label = encode_label(label_nii.get_fdata(dtype=np.float32), label_values)

    initial_nii = nib.load(str(case.initial_mask_path))
    initial_mask = encode_label(initial_nii.get_fdata(dtype=np.float32), label_values)

    if label is not None and label.shape != image.shape[1:]:
        raise ValueError(f"label shape does not match image for case {case.case_id}")
    if initial_mask.shape != image.shape[1:]:
        raise ValueError(f"initial mask shape does not match image for case {case.case_id}")
    return image, label, initial_mask, affine, header


def pad_to_patch(
    image: np.ndarray,
    label: np.ndarray,
    initial_mask: np.ndarray,
    patch_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_shape = image.shape[1:]
    pads = [(0, 0)]
    label_pads = []
    for size, patch in zip(spatial_shape, patch_size):
        after = max(int(patch) - int(size), 0)
        pads.append((0, after))
        label_pads.append((0, after))
    if any(after > 0 for _, after in label_pads):
        image = np.pad(image, pads, mode="constant", constant_values=0)
        label = np.pad(label, label_pads, mode="constant", constant_values=0)
        initial_mask = np.pad(initial_mask, label_pads, mode="constant", constant_values=0)
    return image, label, initial_mask


def random_patch_slices(
    label: np.ndarray,
    patch_size: Sequence[int],
    foreground_probability: float,
) -> tuple[slice, slice, slice]:
    shape = label.shape
    patch_size = tuple(int(v) for v in patch_size)
    if np.random.rand() < foreground_probability and np.any(label > 0):
        foreground = np.argwhere(label > 0)
        center = foreground[np.random.randint(0, len(foreground))]
        starts = [
            int(np.clip(center[dim] - patch_size[dim] // 2, 0, shape[dim] - patch_size[dim]))
            for dim in range(3)
        ]
    else:
        starts = [
            0 if shape[dim] == patch_size[dim] else np.random.randint(0, shape[dim] - patch_size[dim] + 1)
            for dim in range(3)
        ]
    return tuple(slice(start, start + patch) for start, patch in zip(starts, patch_size))


class NnUNetLACMPatchDataset(Dataset):
    """Random-patch dataset for LACM training."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        label_values: Sequence[int],
        patch_size: Sequence[int],
        patches_per_epoch: int | None = None,
        foreground_probability: float = 0.7,
        normalize: bool = True,
    ):
        self.cases = list(cases)
        self.label_values = list(label_values)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.patches_per_epoch = patches_per_epoch
        self.foreground_probability = float(foreground_probability)
        self.normalize = normalize

    def __len__(self) -> int:
        if self.patches_per_epoch is not None and self.patches_per_epoch > 0:
            return int(self.patches_per_epoch)
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.patches_per_epoch is not None and self.patches_per_epoch > 0:
            case = self.cases[np.random.randint(0, len(self.cases))]
        else:
            case = self.cases[index % len(self.cases)]

        image, label, initial_mask, _, _ = load_case_arrays(
            case,
            self.label_values,
            normalize=self.normalize,
        )
        if label is None:
            raise RuntimeError("training requires labels")

        image, label, initial_mask = pad_to_patch(image, label, initial_mask, self.patch_size)
        slices = random_patch_slices(label, self.patch_size, self.foreground_probability)
        image_patch = image[(slice(None),) + slices]
        label_patch = label[slices]
        initial_patch = initial_mask[slices]

        return (
            torch.from_numpy(np.ascontiguousarray(image_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(label_patch)).long(),
            torch.from_numpy(np.ascontiguousarray(initial_patch)).long(),
        )
