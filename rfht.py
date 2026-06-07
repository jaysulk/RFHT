"""
Regularized Fast Hartley Transform (RFHT) for Neural Operators — CORRECTED
==========================================================================
Inspired by Jones (2022), "The Regularized Fast Hartley Transform."

CORRECTED DESIGN (v2):
----------------------
The previous version made a conceptual error: it replaced the *learned*
spectral weights with a butterfly factorization. This crippled the diagonal
operator that elliptic PDEs require (pointwise mult by 1/|k|^2), hurting
exactly the Poisson/biharmonic case where HNO should dominate.

This version separates the two distinct ideas in Jones' book:

  (A) RFHT as a FAST TRANSFORM
      The butterfly with FIXED twiddle factors computes the DHT efficiently.
      In PyTorch this is mathematically identical to the Re-Im FFT trick
      (validated in the paper); the hardware speedup needs a custom kernel.
      We provide both a reference fixed-butterfly FHT and the fast Re-Im path.

  (B) Diagonal-capable LEARNED operator (preserves paper's elliptic win)
      The learned spectral weights stay dense per-mode (exactly as HNO),
      so the diagonal Green's-function multiplier remains representable.

  (C) Optional structured CORRECTION as a regularizer (the new contribution)
      A 2D butterfly-structured, cross-mode coupling term, GATED by a scalar
      initialized to ZERO. At init, RHNO == HNO exactly. The correction can
      only add structured coupling if it reduces loss. This is the honest
      "regularization" story: structure is added on top of, not in place of,
      the diagonal operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. DHT (Re-Im FFT path) — fast, validated in paper
# ---------------------------------------------------------------------------

def dht2d(x: torch.Tensor) -> torch.Tensor:
    """2D DHT via FFT: H{f} = Re{F{f}} - Im{F{f}}."""
    X = torch.fft.fft2(x)
    return X.real - X.imag


def idht2d(H: torch.Tensor) -> torch.Tensor:
    """Inverse 2D DHT (self-inverse up to 1/N)."""
    N = H.shape[-1] * H.shape[-2]
    return dht2d(H) / N


# ---------------------------------------------------------------------------
# 2. Dibit-reversal permutation (Jones Ch. 4) — valid permutation for any n
# ---------------------------------------------------------------------------

def dibit_reverse_indices(n: int) -> torch.Tensor:
    """
    Valid permutation for any n.
    Dibit-reversal for power-of-4, bit-reversal for power-of-2, identity else.
    (The reordering only has structural meaning for power-of-2/4 sizes.)
    """
    if n == 1:
        return torch.tensor([0], dtype=torch.long)
    log4 = math.log(n, 4)
    is_pow4 = abs(log4 - round(log4)) < 1e-9
    log2 = math.log2(n)
    is_pow2 = abs(log2 - round(log2)) < 1e-9

    if is_pow4:
        num_dibits = int(round(log4))
        result = []
        for i in range(n):
            rev = 0; val = i
            for _ in range(num_dibits):
                rev = (rev << 2) | (val & 0x3); val >>= 2
            result.append(rev)
        return torch.tensor(result, dtype=torch.long)
    elif is_pow2:
        num_bits = int(round(log2))
        result = []
        for i in range(n):
            rev = int(bin(i)[2:].zfill(num_bits)[::-1], 2)
            result.append(rev)
        return torch.tensor(result, dtype=torch.long)
    else:
        return torch.arange(n, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. Fixed-twiddle FHT (reference for the RFHT "fast transform" path)
#
# This computes the DHT using the radix-2 butterfly recurrence with FIXED
# (non-learned) cas twiddle factors. It is mathematically identical to dht2d
# but exposes the butterfly compute graph that maps to Jones' FPGA architecture.
# Used as a reference / for the hardware-deployment story, NOT as a learned op.
# ---------------------------------------------------------------------------

def fht1d_fixed(x: torch.Tensor) -> torch.Tensor:
    """
    1D FHT along the last dimension using fixed cas twiddles.
    Matches torch DHT (Re-Im) up to floating point. O(N log N) structure,
    though implemented here with full matrices for clarity (reference only).
    """
    N = x.shape[-1]
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = n.view(-1, 1)
    cas = torch.cos(2 * math.pi * k * n / N) + torch.sin(2 * math.pi * k * n / N)
    return torch.einsum('...n,kn->...k', x, cas)


# ---------------------------------------------------------------------------
# 4. Optional 2D Butterfly Correction (the regularizer) — ZERO-INITIALIZED
#
# Adds structured cross-mode coupling on top of the diagonal operator.
# Separable: a butterfly along height-modes and one along width-modes.
# Gated by a learnable scalar `gamma` initialized to 0, so at init this is
# a no-op and RHNO == HNO exactly.
# ---------------------------------------------------------------------------

class ButterflyCorrection2d(nn.Module):
    """
    Zero-initialized structured correction. Couples modes within a quadrant
    via learned 2x2 butterfly blocks along each spatial-frequency axis.
    """
    def __init__(self, channels: int, modes: int, num_stages: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.modes = modes
        if num_stages is None:
            self.num_stages = max(1, int(math.log2(modes))) if modes > 1 else 1
        else:
            self.num_stages = num_stages

        half = modes // 2
        # Butterfly 2x2 blocks for height axis and width axis, per stage.
        # Initialize near the identity butterfly [[1,1],[1,-1]]/sqrt(2) so the
        # correction produces a MEANINGFUL signal h. The no-op property at init
        # comes solely from gamma=0 (ReZero/LayerScale-style), which keeps the
        # gradient dL/dgamma = <upstream, h> at a usable magnitude rather than
        # vanishing (the bug from a tiny-weight init).
        base = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
        init_h = base.view(1, 1, 1, 2, 2).repeat(self.num_stages, half, channels, 1, 1)
        init_w = init_h.clone()
        init_h = init_h + 0.05 * torch.randn_like(init_h)
        init_w = init_w + 0.05 * torch.randn_like(init_w)
        self.bfly_h = nn.Parameter(init_h)
        self.bfly_w = nn.Parameter(init_w)
        # Zero-initialized gate — at init, correction contributes nothing,
        # but its gradient is non-vanishing because h is meaningful.
        self.gamma = nn.Parameter(torch.zeros(1))

        self.register_buffer('perm', dibit_reverse_indices(modes))

    def _butterfly_axis(self, x: torch.Tensor, weights: torch.Tensor,
                        axis: int) -> torch.Tensor:
        """
        Apply paired butterfly mixing along `axis` (either -2 or -1).
        x: (B, C, m, m). weights: (num_stages, half, C, 2, 2).
        """
        h = x
        m = self.modes
        half = m // 2
        for s in range(self.num_stages):
            if axis == -1:
                lo = h[..., :half]            # (B, C, m, half)
                hi = h[..., half:]
                pairs = torch.stack([lo, hi], dim=-1)        # (B,C,m,half,2)
                # contract channel + pair-component with per-position 2x2
                # weights[s]: (half, C, 2, 2)
                out = torch.einsum('bcmhp,hcpq->bcmhq', pairs, weights[s])
                h = torch.cat([out[..., 0], out[..., 1]], dim=-1)
                h = h[..., self.perm]
            else:  # axis == -2
                lo = h[:, :, :half, :]        # (B, C, half, m)
                hi = h[:, :, half:, :]
                pairs = torch.stack([lo, hi], dim=-1)        # (B,C,half,m,2)
                out = torch.einsum('bchmp,hcpq->bchmq', pairs, weights[s])
                h = torch.cat([out[..., 0], out[..., 1]], dim=2)
                h = h[:, :, self.perm, :]
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, modes, modes). Returns same shape, gated by gamma."""
        h = self._butterfly_axis(x, self.bfly_w, axis=-1)
        h = self._butterfly_axis(h, self.bfly_h, axis=-2)
        return self.gamma * h


# ---------------------------------------------------------------------------
# 5. Corrected RFHT Spectral Convolution
#
# Main path: diagonal-capable dense per-mode weights (IDENTICAL to HNO),
#            preserving the elliptic Green's-function advantage.
# Optional:  zero-init butterfly correction adding structured coupling.
# ---------------------------------------------------------------------------

class RFHTSpectralConv2d(nn.Module):
    """
    Corrected Hartley spectral convolution.

    Parameters
    ----------
    in_channels, out_channels : int
    modes : int
        Modes retained per quadrant edge.
    use_butterfly_correction : bool
        If False, this is EXACTLY the HNO dense spectral conv.
        If True, adds a zero-initialized structured correction (RHNO).
    num_butterfly_stages : int, optional
        Depth of the correction's butterfly (default log2(modes)).
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 use_butterfly_correction: bool = True,
                 num_butterfly_stages: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.use_correction = use_butterfly_correction

        # Diagonal-capable dense per-mode weights (4 quadrants x even/odd).
        # einsum keeps modes (kh,kw) independent -> can represent any diagonal.
        scale = 1.0 / (in_channels * out_channels)
        self.W_even = nn.ParameterList([
            nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, modes))
            for _ in range(4)
        ])
        self.W_odd = nn.ParameterList([
            nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, modes))
            for _ in range(4)
        ])

        # Optional structured correction (operates in out_channel space)
        if use_butterfly_correction:
            self.correction = nn.ModuleList([
                ButterflyCorrection2d(out_channels, modes, num_butterfly_stages)
                for _ in range(4)
            ])
        else:
            self.correction = None

    def _diag_mul(self, W, x):
        # (out,in,kh,kw),(B,in,kh,kw) -> (B,out,kh,kw): per-mode channel mix
        return torch.einsum('oikj,bikj->bokj', W, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m = self.modes

        x_ht = dht2d(x)
        x_flip = torch.roll(torch.flip(x_ht, dims=[-1, -2]),
                            shifts=(1, 1), dims=[-1, -2])
        x_even = (x_ht + x_flip) / 2.0
        x_odd  = (x_ht - x_flip) / 2.0

        slots = [
            (slice(None, m),  slice(None, m)),
            (slice(None, m),  slice(-m, None)),
            (slice(-m, None), slice(None, m)),
            (slice(-m, None), slice(-m, None)),
        ]
        quads_e = [x_even[:, :, sh, sw] for sh, sw in slots]
        quads_o = [x_odd[:, :, sh, sw]  for sh, sw in slots]

        out_ht = torch.zeros(B, self.out_channels, H, W,
                             device=x.device, dtype=x.dtype)

        for q, (sh, sw) in enumerate(slots):
            # Diagonal-capable main operator (preserves elliptic advantage)
            out = (self._diag_mul(self.W_even[q], quads_e[q]) +
                   self._diag_mul(self.W_odd[q],  quads_o[q]))
            # Optional zero-init structured correction
            if self.correction is not None:
                out = out + self.correction[q](out)
            out_ht[:, :, sh, sw] = out

        return idht2d(out_ht)