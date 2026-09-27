"""Hayyi's radiomics-conditioned U-Net for loading his trained checkpoint.

The parameter names and bottleneck fusion match the model in Hayyi's branch.
This class is separate from Zaq's unchanged model architecture.
"""

from typing import Optional
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .multitask import MultiTaskOutput, MultiTaskUNet3D
from .unet3d import UNet3D


class HayyiUNet3D(UNet3D):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 base_features: int = 16, radiomics_dim: int = 386):
        super().__init__(in_channels, out_channels, base_features)
        self.radiomics_dim = int(radiomics_dim)
        self.radiomics_norm = nn.LayerNorm(self.radiomics_dim)
        self.radiomics_proj = nn.Linear(self.radiomics_dim, self.bottleneck_channels)

    def forward_features(self, x: torch.Tensor, radiomics: torch.Tensor):
        if radiomics.ndim == 1:
            radiomics = radiomics.unsqueeze(0)
        if radiomics.shape != (x.shape[0], self.radiomics_dim):
            raise ValueError(
                f"Expected radiomics shape {(x.shape[0], self.radiomics_dim)}, "
                f"got {tuple(radiomics.shape)}"
            )

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        r = self.radiomics_proj(self.radiomics_norm(radiomics))
        b = b + r.view(r.size(0), self.bottleneck_channels, 1, 1, 1)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1), b


class HayyiMultiTaskUNet3D(MultiTaskUNet3D):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 base_features: int = 16, radiomics_dim: int = 386):
        backbone = HayyiUNet3D(in_channels, out_channels, base_features, radiomics_dim)
        super().__init__(segmentation=backbone)
        self.radiomics_dim = radiomics_dim

    def forward(self, x: torch.Tensor, tumor_mask: Optional[torch.Tensor] = None,
                radiomics: Optional[torch.Tensor] = None) -> MultiTaskOutput:
        if radiomics is None:
            raise ValueError("Hayyi's model requires the matching radiomics vector")
        seg_logits, bottleneck = self.segmentation.forward_features(x, radiomics)
        if tumor_mask is None:
            tumor_mask = torch.sigmoid(seg_logits)
        survival = self.survival_head(bottleneck, tumor_mask.detach())
        return MultiTaskOutput(seg_logits=seg_logits, survival=survival)


def load_case_radiomics(csv_path: Path, stats_path: Path, case_id: str) -> torch.Tensor:
    """Return Hayyi's saved, normalized whole-tumor vector for one known case.

    The supplied CSV was extracted using reference masks. It must not be used
    to claim blind performance on new patients.
    """
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty radiomics CSV: {csv_path}")
        names = [name for name in reader.fieldnames if name not in {"case_id", "roi", "split"}]
        if names != stats["feature_names"]:
            raise ValueError("Radiomics CSV columns differ from Hayyi's saved training statistics")
        row = next((item for item in reader if item["case_id"] == case_id), None)

    if row is None:
        raise ValueError(
            f"No Hayyi radiomics row for {case_id}. This model cannot run for "
            "this patient with the supplied CSV."
        )
    if row.get("roi") != "whole":
        raise ValueError(f"{case_id}: expected whole-tumor radiomics, got {row.get('roi')!r}")

    raw = np.asarray([float(row[name]) if row[name] else 0.0 for name in names], dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != raw.shape or std.shape != raw.shape or np.any(std <= 0):
        raise ValueError("Invalid radiomics normalization dimensions")
    return torch.from_numpy(((raw - mean) / std).copy()).unsqueeze(0)
