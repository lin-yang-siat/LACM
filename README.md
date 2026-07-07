# A Lightweight Learnable Active Contour Model for Annotation-Efficient Whole-Body Composition Analysis Using MR Proton Density Fat Fraction Images

This repository contains a lightweight Learnable Active Contours Model (LACM)
for nnUNet-initialized whole-body MR-PDFF segmentation.

## Files

```text
model.py    LACMNet and unrolled contour-evolution layers
data.py     nnUNet-style NIfTI data pairing and patch sampling
trainer.py  training, sliding-window inference, Dice comparison
train.py    command-line entry point
```

## Data Layout

All folders are configured under `--data_root`.

```text
TaskXXX/
  imagesTr/
    case_0000.nii.gz
  labelsTr/
    case.nii.gz
  inferTr/
    case.nii.gz
  imagesTs/
    case_0000.nii.gz
  labelsTs/
    case.nii.gz
  inferTs/
    case.nii.gz
```

`inferTr` and `inferTs` are nnUNet predictions used as the initial contour
`u0`.

## Train

```powershell
python LACM\train.py \
  --data_root "D:\your\nnUNet_raw\TaskXXX" \
  --train_image_dir imagesTr \
  --train_label_dir labelsTr \
  --train_init_dir inferTr \
  --test_image_dir imagesTs \
  --test_label_dir labelsTs \
  --test_init_dir imferTs \
  --classes 11 \
  --models 3 \
  --patch 256 256 48 \
  --patch_step 256 256 24 \
  --local_sigma 3.0 \
  --local_kernel_radius 5 \
  --sigma 1.0 \
  --kernel_radius 3 \
  --term_source init \
  --test_num 5 \
  --device cuda:0
```

## Test Only

```powershell
python LACM\train.py \
  --data_root "D:\your\nnUNet_raw\TaskXXX" \
  --is_train false \
  --is_test true \
  --is_load true \
  --load_epoch 100 \
  --test_init_dir imferTs \
  --classes 11 \
  --device cuda:0
```

## Outputs

```text
runs/lacm_params/       checkpoints
runs/lacm_predictions/  refined masks
runs/lacm_metrics/      per-case and per-class Dice CSV files
```
