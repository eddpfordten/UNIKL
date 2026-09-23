"""
Simple 3D U-Net for volumetric brain tumor segmentation.

Architecture overview:
  Encoder (downsampling) -> Bottleneck (+ optional radiomics) -> Decoder

Input:  (batch, 4, D, H, W)   — 4 MRI modalities
Output: (batch, 1, D, H, W)   — tumor probability map

When radiomics_dim > 0, a normalised PyRadiomics vector is projected and
reshaped to the bottleneck tensor ``b`` and added before the decoder.
CSV z-score normalisation is applied in the dataloader; a LayerNorm here
stabilises the fused scale inside the network.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net with 3 encoder levels and 3 decoder levels.

    This is a lightweight version suitable for intermediate-level projects
    and GPUs with limited memory.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 16,
        radiomics_dim: int = 0,
    ):
        super().__init__()

        # Width of the bottleneck feature map, published so heads attached to it
        # (see models/multitask.py) do not have to re-derive the arithmetic.
        self.bottleneck_channels = base_features * 8
        self.radiomics_dim = int(radiomics_dim)

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_features * 4, base_features * 8)

        # Optional radiomics → bottleneck fusion (same shape as b).
        # CSV values are already z-scored in RadiomicsFeatureTable; LayerNorm
        # re-centres each sample before the linear map to C channels.
        if self.radiomics_dim > 0:
            self.radiomics_norm = nn.LayerNorm(self.radiomics_dim)
            self.radiomics_proj = nn.Linear(self.radiomics_dim, self.bottleneck_channels)
        else:
            self.radiomics_norm = None
            self.radiomics_proj = None

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_features * 2, base_features)

        # Final 1x1x1 convolution -> single output channel
        self.out_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def _fuse_radiomics(self, b: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        """
        Normalise radiomics, project to bottleneck channels, reshape to ``b``.

        Args:
            b:          Bottleneck map (batch, C, D, H, W).
            radiomics:  (batch, radiomics_dim) — CSV features (preferably
                        already z-scored on the training split).

        Returns:
            ``b`` with a residual radiomics map of the same shape added.
        """
        if radiomics.dim() == 1:
            radiomics = radiomics.unsqueeze(0)

        # Per-sample normalisation (CSV z-score is train-cohort; this is local).
        r = self.radiomics_norm(radiomics)
        r = self.radiomics_proj(r)  # (batch, C)

        # Reshape / broadcast to match b: (batch, C, D, H, W).
        r = r.view(r.size(0), self.bottleneck_channels, 1, 1, 1).expand_as(b)
        return b + r

    def forward_features(
        self,
        x: torch.Tensor,
        radiomics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full U-Net and also hand back the bottleneck activations.

        The bottleneck is the most compressed view the network has. The
        survival head pools it *inside the tumor mask*, so the per-patient
        prediction is based on the mass rather than the whole brain.

        Args:
            x:          Input volume of shape (batch, in_channels, D, H, W).
            radiomics:  Optional (batch, radiomics_dim) vector fused into ``b``.

        Returns:
            (segmentation logits, bottleneck features of shape
            (batch, bottleneck_channels, D/8, H/8, W/8)).
        """
        # Encoder path (save skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Radiomics: after normalisation, reshape to match the same shape as b.
        if (
            radiomics is not None
            and self.radiomics_dim > 0
            and self.radiomics_proj is not None
        ):
            b = self._fuse_radiomics(b, radiomics)

        # Decoder path (concatenate skip connections)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1), b

    def forward(
        self,
        x: torch.Tensor,
        radiomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_features(x, radiomics=radiomics)[0]
