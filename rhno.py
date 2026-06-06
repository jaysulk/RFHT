"""
Regularized Hartley Neural Operator (RHNO) — CORRECTED (v2)
===========================================================
Both HNO and RHNO now share the SAME diagonal-capable spectral conv.
The only difference: RHNO enables a zero-initialized structured correction.

  make_hno()  -> RFHTSpectralConv2d(use_butterfly_correction=False)
                 == exact HNO from the paper (dense per-mode weights)
  make_rhno() -> RFHTSpectralConv2d(use_butterfly_correction=True)
                 == HNO + zero-init butterfly correction (regularizer)

At initialization, RHNO and HNO produce IDENTICAL outputs (gamma=0).
This guarantees RHNO can never start worse than HNO, and only adds
structured cross-mode coupling where it reduces loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from rfht import RFHTSpectralConv2d, dht2d, idht2d


# Kept for backward-compatibility / direct comparison; identical to
# RFHTSpectralConv2d(use_butterfly_correction=False).
class HNOSpectralConv2d(nn.Module):
    """Original dense HNO spectral conv (paper Eq. 9)."""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.conv = RFHTSpectralConv2d(
            in_channels, out_channels, modes,
            use_butterfly_correction=False
        )

    def forward(self, x):
        return self.conv(x)


class SpectralBlock(nn.Module):
    """One spectral conv block with residual bypass (paper Figure 1)."""
    def __init__(self, channels: int, modes: int,
                 use_rfht: bool = True,
                 num_butterfly_stages: Optional[int] = None):
        super().__init__()
        # use_rfht == True  -> enable the structured correction (RHNO)
        # use_rfht == False -> diagonal-only, == HNO
        self.spectral_conv = RFHTSpectralConv2d(
            channels, channels, modes,
            use_butterfly_correction=use_rfht,
            num_butterfly_stages=num_butterfly_stages
        )
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral_conv(x) + self.bypass(x)))


class HartleyNeuralOperator(nn.Module):
    """
    Hartley Neural Operator — base for HNO and RHNO.

    use_rfht=False -> HNO  (diagonal-capable dense weights only)
    use_rfht=True  -> RHNO (HNO + zero-init structured correction)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        num_blocks: int = 3,
        use_rfht: bool = True,
        num_butterfly_stages: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.num_blocks = num_blocks
        self.use_rfht = use_rfht

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList([
            SpectralBlock(width, modes, use_rfht=use_rfht,
                          num_butterfly_stages=num_butterfly_stages)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h.permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.permute(0, 2, 3, 1)
        return self.output_proj(h)

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        spectral = sum(
            p.numel()
            for block in self.blocks
            for _, p in block.spectral_conv.named_parameters()
        )
        # Correction params specifically (the added regularizer cost)
        corr = 0
        for block in self.blocks:
            if block.spectral_conv.correction is not None:
                for c in block.spectral_conv.correction:
                    corr += sum(p.numel() for p in c.parameters())
        return {
            'total': total,
            'spectral': spectral,
            'correction': corr,
            'non_spectral': total - spectral,
            'spectral_fraction': spectral / total,
        }


def make_hno(in_channels=4, out_channels=1, width=32, modes=16,
             num_blocks=3) -> HartleyNeuralOperator:
    """Exact HNO from paper: diagonal-capable dense weights, no correction."""
    return HartleyNeuralOperator(
        in_channels=in_channels, out_channels=out_channels,
        width=width, modes=modes, num_blocks=num_blocks,
        use_rfht=False
    )


def make_rhno(in_channels=4, out_channels=1, width=32, modes=16,
              num_blocks=3, num_butterfly_stages=None) -> HartleyNeuralOperator:
    """RHNO: HNO + zero-init structured (RFHT-inspired) correction."""
    return HartleyNeuralOperator(
        in_channels=in_channels, out_channels=out_channels,
        width=width, modes=modes, num_blocks=num_blocks,
        use_rfht=True, num_butterfly_stages=num_butterfly_stages
    )