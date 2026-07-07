# -*- coding: utf-8 -*-
"""Learnable Active Contours Model (LACM).

This module implements the energy-based segmentation model described in the
paper:

    phi_theta = F_theta(I, Lambda_t) + lambda * R_theta(I, u_t)
    u_{t+1} = softmax(-phi_theta / epsilon)

The implementation keeps the current project behavior while using paper-facing
names suitable for release. The analytical region potential uses local
PDFF-aware class statistics and the regularization balance is class-wise.
"""
from __future__ import annotations

import math
from typing import Dict, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


UpdateSource = Literal["init", "iter"]


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def parse_class_values(values: str | None, n_classes: int, default: float) -> list[float]:
    """Parse comma-separated per-class hyperparameters."""
    if values is None or values.strip() == "":
        return [float(default)] * n_classes
    parsed = [float(v.strip()) for v in values.split(",") if v.strip()]
    if len(parsed) != n_classes:
        raise ValueError(f"expected {n_classes} comma-separated values, got {len(parsed)}")
    return parsed


class ConvNormActivation(nn.Module):
    """3D convolution block used in learnable LACM potentials."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable convolution used by D_Ftheta and D_Rtheta."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class LearnablePotentialBranch(nn.Module):
    """Residual learner for either F_theta or R_theta."""

    def __init__(self, in_channels: int, n_classes: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormActivation(in_channels, hidden_channels),
            DepthwiseSeparableConv3d(hidden_channels),
            ConvNormActivation(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, n_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LACMIteration(nn.Module):
    """One unrolled LACM contour-evolution step."""

    def __init__(self, image_channels: int, n_classes: int, hidden_channels: int):
        super().__init__()
        branch_channels = image_channels + 2 * n_classes
        self.region_residual = LearnablePotentialBranch(
            branch_channels,
            n_classes,
            hidden_channels,
        )
        self.regularization_residual = LearnablePotentialBranch(
            branch_channels,
            n_classes,
            hidden_channels,
        )

    def forward(self, image: Tensor, u0_one_hot: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.cat([image, u0_one_hot, u], dim=1)
        delta_region = self.region_residual(features)
        delta_regularization = self.regularization_residual(features)
        return delta_region, delta_regularization


class LACMNet(nn.Module):
    """Lightweight Learnable Active Contours Model for MR-PDFF segmentation."""

    def __init__(
        self,
        n_iterations: int,
        n_classes: int,
        image_channels: int = 1,
        hidden_channels: int = 16,
        region_sigma: float = 3.0,
        region_kernel_radius: int = 5,
        regularization_sigma: float = 1.0,
        regularization_kernel_radius: int = 3,
        initial_lambda: float = 0.035,
        lambda_values: str | None = None,
        initial_epsilon: float = 1.0,
        update_source: UpdateSource = "init",
    ):
        super().__init__()
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if update_source not in ("init", "iter"):
            raise ValueError("update_source must be either 'init' or 'iter'")

        self.n_iterations = int(n_iterations)
        self.n_class = int(n_classes)
        self.n_classes = int(n_classes)
        self.image_channels = int(image_channels)
        self.region_sigma = float(max(region_sigma, 1e-6))
        self.region_kernel_radius = int(region_kernel_radius)
        self.regularization_sigma = float(max(regularization_sigma, 1e-6))
        self.regularization_kernel_radius = int(regularization_kernel_radius)
        self.update_source = update_source

        lambda_init = parse_class_values(lambda_values, self.n_classes, initial_lambda)
        raw_lambda = [inverse_softplus(v) for v in lambda_init]
        self.raw_lambda = nn.Parameter(torch.tensor(raw_lambda).view(1, self.n_classes, 1, 1, 1))
        self.raw_epsilon = nn.Parameter(torch.tensor(inverse_softplus(initial_epsilon)))
        self.iterations = nn.ModuleList(
            [
                LACMIteration(self.image_channels, self.n_classes, hidden_channels)
                for _ in range(self.n_iterations)
            ]
        )

    @property
    def lambda_weight(self) -> Tensor:
        """Class-wise balance between region and regularization potentials."""
        return F.softplus(self.raw_lambda) + 1e-6

    @property
    def epsilon(self) -> Tensor:
        """Positive Softmax temperature."""
        return F.softplus(self.raw_epsilon) + 1e-6

    def labels_to_one_hot(self, labels: Tensor) -> Tensor:
        if labels.dim() == 5 and labels.size(1) == 1:
            labels = labels[:, 0]
        if labels.dim() != 4:
            raise ValueError("labels must have shape [B, D, H, W] or [B, 1, D, H, W]")
        labels = labels.long().clamp_(0, self.n_classes - 1)
        one_hot = F.one_hot(labels, num_classes=self.n_classes)
        return one_hot.permute(0, 4, 1, 2, 3).contiguous().float()

    @staticmethod
    def gaussian_kernel(
        sigma: float,
        radius: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if radius <= 0:
            return torch.ones((1, 1, 1, 1, 1), device=device, dtype=dtype)
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square() + zz.square()) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel.view(1, 1, *kernel.shape)

    def smooth_channels(self, x: Tensor, sigma: float, radius: int) -> Tensor:
        channels = x.size(1)
        kernel = self.gaussian_kernel(sigma, radius, x.device, x.dtype)
        weight = kernel.repeat(channels, 1, 1, 1, 1)
        return F.conv3d(x, weight, padding=radius, groups=channels)

    def pdff_region_potential(self, image: Tensor, u_reference: Tensor) -> Tensor:
        """Analytical PDFF-aware region potential F(I, Lambda_t)."""
        batch_size, modalities, depth, height, width = image.shape

        local_denominator = self.smooth_channels(
            u_reference,
            self.region_sigma,
            self.region_kernel_radius,
        )
        image_by_class = image.unsqueeze(1) * u_reference.unsqueeze(2)
        numerator = image_by_class.view(
            batch_size,
            self.n_classes * modalities,
            depth,
            height,
            width,
        )
        numerator = self.smooth_channels(numerator, self.region_sigma, self.region_kernel_radius)
        numerator = numerator.view(batch_size, self.n_classes, modalities, depth, height, width)
        local_mean = numerator / local_denominator.unsqueeze(2).clamp_min(1e-6)

        global_count = u_reference.sum(dim=(2, 3, 4), keepdim=True).unsqueeze(2)
        global_sum = image_by_class.sum(dim=(3, 4, 5), keepdim=True)
        global_mean = global_sum / global_count.clamp_min(1e-6)
        image_mean = image.mean(dim=(2, 3, 4), keepdim=True).view(
            batch_size,
            1,
            modalities,
            1,
            1,
            1,
        )
        global_mean = torch.where(global_count > 1.0, global_mean, image_mean)
        local_mean = torch.where(
            local_denominator.unsqueeze(2) > 1e-4,
            local_mean,
            global_mean.expand_as(local_mean),
        )

        return (image.unsqueeze(1) - local_mean).square().mean(dim=2)

    def boundary_regularization_potential(self, u_reference: Tensor) -> Tensor:
        """Analytical boundary regularization potential from threshold dynamics."""
        regularization = self.smooth_channels(
            1.0 - 2.0 * u_reference,
            self.regularization_sigma,
            self.regularization_kernel_radius,
        )
        return math.sqrt(math.pi / self.regularization_sigma) * regularization

    def forward(
        self,
        image: Tensor,
        initial_mask: Tensor,
        return_dict: bool = True,
    ) -> Dict[str, Tensor] | Tensor:
        if image.dim() != 5:
            raise ValueError("image must have shape [B, C, D, H, W]")
        if image.size(1) != self.image_channels:
            raise ValueError(
                f"model expects {self.image_channels} image channels, got {image.size(1)}"
            )

        u0 = self.labels_to_one_hot(initial_mask)
        u = u0
        logits = None
        phi = None
        region_potential = None
        regularization_potential = None

        for iteration in self.iterations:
            if self.update_source == "iter" and logits is not None:
                u_reference = self.labels_to_one_hot(u.argmax(dim=1))
            else:
                u_reference = u0

            analytical_region = self.pdff_region_potential(image, u_reference)
            analytical_regularization = self.boundary_regularization_potential(u_reference)
            learned_region, learned_regularization = iteration(image, u0, u)

            region_potential = analytical_region + learned_region
            regularization_potential = analytical_regularization + learned_regularization
            phi = region_potential + self.lambda_weight * regularization_potential
            logits = -phi / self.epsilon
            u = F.softmax(logits, dim=1)

        if not return_dict:
            return u

        return {
            "prob": u,
            "logits": logits,
            "phi": phi,
            "region_potential": region_potential,
            "regularization_potential": regularization_potential,
            "lambda": self.lambda_weight.detach(),
            "epsilon": self.epsilon.detach(),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LACMNet(n_iterations=2, n_classes=4, image_channels=1, hidden_channels=4).to(device)
    image = torch.randn(1, 1, 24, 24, 12, device=device)
    initial_mask = torch.randint(0, 4, (1, 24, 24, 12), device=device)
    output = model(image, initial_mask)
    print(output["prob"].shape, output["lambda"].flatten())
