"""
Regularized Hartley Neural Operator (RHNO)
==========================================
Extends the HNO from the Allerton/TMLR paper with RFHT-inspired
butterfly-factorized spectral convolution layers.

Architecture mirrors the paper exactly:
- Same input projection, spectral blocks, residual bypass, output projection
- Same 3 blocks for time-dependent PDEs, 4 for elliptic
- GELU activations, 1x1 residual convolutions
- Only the spectral convolution is replaced

Comparison:
  HNO  (paper) : RFHTSpectralConv2d with dense Weven/Wodd
  RHNO (this)  : RFHTSpectralConv2d with butterfly-factorized weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transforms.rfht import RFHTSpectralConv2d, dht2d, idht2d


# ---------------------------------------------------------------------------
# Also include the original dense HNO spectral conv for fair comparison
# (matches paper implementation exactly)
# ---------------------------------------------------------------------------

class HNOSpectralConv2d(nn.Module):
    """
    Original dense HNO spectral convolution from the paper.
    Equation 9: y = H^{-1}[Weven * Heven[x] + Wodd * Hodd[x]]
    Four corners, dense weights per quadrant.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # 4 quadrants × 2 (even/odd) = 8 weight tensors
        # Each: (out_channels, in_channels, modes, modes)
        scale = 1.0 / (in_channels * out_channels)
        self.W_even = nn.ParameterList([
            nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, modes))
            for _ in range(4)
        ])
        self.W_odd = nn.ParameterList([
            nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, modes))
            for _ in range(4)
        ])
    
    def _compl_mul2d(self, W, x):
        # einsum: (out, in, kh, kw), (batch, in, kh, kw) -> (batch, out, kh, kw)
        return torch.einsum('oikj,bikj->bokj', W, x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W_size = x.shape
        m = self.modes
        
        x_ht = dht2d(x)
        x_flip = torch.roll(torch.flip(x_ht, dims=[-1, -2]),
                            shifts=(1, 1), dims=[-1, -2])
        x_even = (x_ht + x_flip) / 2.0
        x_odd  = (x_ht - x_flip) / 2.0
        
        quads_e = [x_even[:, :, :m, :m], x_even[:, :, :m, -m:],
                   x_even[:, :, -m:, :m], x_even[:, :, -m:, -m:]]
        quads_o = [x_odd[:, :, :m, :m],  x_odd[:, :, :m, -m:],
                   x_odd[:, :, -m:, :m],  x_odd[:, :, -m:, -m:]]
        
        out_ht = torch.zeros(B, self.out_channels, H, W_size,
                             device=x.device, dtype=x.dtype)
        
        slots = [
            (slice(None, m),  slice(None, m)),
            (slice(None, m),  slice(-m, None)),
            (slice(-m, None), slice(None, m)),
            (slice(-m, None), slice(-m, None)),
        ]
        for q, (sh, sw) in enumerate(slots):
            out = (self._compl_mul2d(self.W_even[q], quads_e[q]) +
                   self._compl_mul2d(self.W_odd[q],  quads_o[q]))
            out_ht[:, :, sh, sw] = out
        
        N = H * W_size
        return dht2d(out_ht) / N


# ---------------------------------------------------------------------------
# Shared spectral block (used by both HNO and RHNO)
# ---------------------------------------------------------------------------

class SpectralBlock(nn.Module):
    """
    One spectral convolution block with residual bypass.
    Matches paper Figure 1 exactly.
    """
    def __init__(self, channels: int, modes: int,
                 use_rfht: bool = True,
                 num_butterfly_stages: Optional[int] = None):
        super().__init__()
        
        if use_rfht:
            self.spectral_conv = RFHTSpectralConv2d(
                channels, channels, modes,
                num_butterfly_stages=num_butterfly_stages
            )
        else:
            self.spectral_conv = HNOSpectralConv2d(channels, channels, modes)
        
        # 1x1 residual bypass (paper: "1×1 convolutions on flattened spatial dims")
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral_conv(x) + self.bypass(x)))


# ---------------------------------------------------------------------------
# Main RHNO model (and HNO for comparison)
# ---------------------------------------------------------------------------

class HartleyNeuralOperator(nn.Module):
    """
    Hartley Neural Operator — base class for HNO and RHNO.
    
    Parameters
    ----------
    in_channels : int
        Input channels. Paper uses 4 for time-dep [u0,x,y,t], 3 for elliptic [f,x,y]
    out_channels : int
        Output channels (typically 1)
    width : int
        Hidden channel width (paper searches {16, 32, 48})
    modes : int
        Spectral modes retained per quadrant edge
    num_blocks : int
        Spectral conv blocks (paper: 3 time-dep, 4 elliptic)
    use_rfht : bool
        If True: RFHT butterfly-factorized weights (RHNO)
        If False: original dense weights (HNO, reproduces paper)
    num_butterfly_stages : int, optional
        Butterfly depth. Default: log2(modes). Jones radix-4: log4(modes)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
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
        
        # Input projection P: R^{d_in} -> R^{width}
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        
        # Spectral blocks
        self.blocks = nn.ModuleList([
            SpectralBlock(width, modes, use_rfht=use_rfht,
                         num_butterfly_stages=num_butterfly_stages)
            for _ in range(num_blocks)
        ])
        
        # Output projection Q: R^{width} -> R^{d_out}
        self.output_proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, H, W, in_channels) — channels last, matching paper convention
        Returns: (batch, H, W, out_channels)
        """
        # Input projection (applied pointwise)
        h = self.input_proj(x)             # (B, H, W, width)
        h = h.permute(0, 3, 1, 2)         # (B, width, H, W) for conv layers
        
        # Spectral blocks
        for block in self.blocks:
            h = block(h)
        
        # Output projection
        h = h.permute(0, 2, 3, 1)         # (B, H, W, width)
        return self.output_proj(h)          # (B, H, W, out_channels)
    
    def parameter_count(self) -> dict:
        """Breakdown of parameter counts for analysis."""
        total = sum(p.numel() for p in self.parameters())
        spectral = sum(
            p.numel()
            for block in self.blocks
            for name, p in block.spectral_conv.named_parameters()
        )
        return {
            'total': total,
            'spectral': spectral,
            'non_spectral': total - spectral,
            'spectral_fraction': spectral / total,
        }


def make_hno(in_channels=4, out_channels=1, width=32, modes=12,
             num_blocks=3) -> HartleyNeuralOperator:
    """Reproduce original HNO from paper (dense weights)."""
    return HartleyNeuralOperator(
        in_channels=in_channels, out_channels=out_channels,
        width=width, modes=modes, num_blocks=num_blocks,
        use_rfht=False
    )


def make_rhno(in_channels=4, out_channels=1, width=32, modes=12,
              num_blocks=3, num_butterfly_stages=None) -> HartleyNeuralOperator:
    """RFHT-regularized HNO (this work)."""
    return HartleyNeuralOperator(
        in_channels=in_channels, out_channels=out_channels,
        width=width, modes=modes, num_blocks=num_blocks,
        use_rfht=True,
        num_butterfly_stages=num_butterfly_stages
    )
