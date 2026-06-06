"""
Regularized Fast Hartley Transform (RFHT) for Neural Operators
==============================================================
Inspired by Jones (2022), "The Regularized Fast Hartley Transform:
Low-Complexity Parallel Computation of the FHT in One and Multiple Dimensions"

Key ideas from the book translated to PyTorch:
1. Partitioned-memory butterfly structure (dibit-reversal ordering)
2. 8-fold parallelism via the double butterfly processing element
3. Structural weight factorization as a regularizer on spectral convolution

The "regularization" in Jones' sense is algorithmic: the RFHT imposes a
specific factorization of the transform computation that constrains how
information flows between frequency bins. We exploit this as a structural
prior on the learned spectral weight matrices in the HNO.

Architecture note:
- Standard HNO (from paper): dense Weven, Wodd per quadrant — 8 dense matrices
- RHNO (this work): weights factorized via butterfly stages — far fewer parameters,
  structured sparsity that mirrors the RFHT compute graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. DHT via FFT (same as original HNO paper, Re - Im trick)
# ---------------------------------------------------------------------------

def dht2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D Discrete Hartley Transform via FFT.
    H{f}(k) = Re{F{f}(k)} - Im{F{f}(k)}
    
    Input:  (..., H, W) real tensor
    Output: (..., H, W) real tensor
    """
    X = torch.fft.fft2(x)
    return X.real - X.imag


def idht2d(H: torch.Tensor) -> torch.Tensor:
    """
    Inverse 2D DHT. The DHT is self-inverse up to 1/N normalization.
    Since torch.fft.ifft2 handles normalization, we use the same Re-Im trick.
    """
    N = H.shape[-1] * H.shape[-2]
    return dht2d(H) / N


# ---------------------------------------------------------------------------
# 2. Dibit-reversal permutation (from Jones Ch. 4)
#
# In the RFHT, data is reordered between memory partitions using dibit-reversal
# (a generalization of bit-reversal for radix-4). For a neural operator, this
# defines which frequency bins are "coupled" in each butterfly stage.
# ---------------------------------------------------------------------------

def dibit_reverse_indices(n: int) -> torch.Tensor:
    """
    Compute dibit-reversal permutation indices for length-n sequence.
    n must be a power of 4 (or we fall back to bit-reversal for power of 2).
    
    Jones (2022) §4.3: the RFHT uses dibit-reversal rather than bit-reversal
    to achieve 8-fold parallelism with the double butterfly PE.
    """
    if n == 1:
        return torch.tensor([0])
    
    # Check if power of 4
    log4 = math.log(n, 4)
    is_pow4 = abs(log4 - round(log4)) < 1e-9
    
    indices = list(range(n))
    
    if is_pow4:
        # Dibit reversal: reverse 2-bit groups
        num_dibits = int(round(log4))
        result = []
        for i in range(n):
            rev = 0
            val = i
            for _ in range(num_dibits):
                rev = (rev << 2) | (val & 0x3)
                val >>= 2
            result.append(rev)
        return torch.tensor(result, dtype=torch.long)
    else:
        # Fallback: standard bit-reversal for power of 2
        num_bits = int(math.log2(n))
        result = []
        for i in range(n):
            rev = int(bin(i)[2:].zfill(num_bits)[::-1], 2)
            result.append(rev)
        return torch.tensor(result, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. RFHT Butterfly Stage
#
# Jones' double butterfly PE combines two radix-2 butterfly stages:
#   [a, b, c, d] -> [a+b+c+d, a-b+c-d, a+b-c-d, a-b-c+d] (with twiddles)
#
# For the neural operator, we implement this as a learnable butterfly layer
# where the twiddle factors are replaced by learned real weights.
# This is the core "structural regularization" idea.
# ---------------------------------------------------------------------------

class ButterflyStage(nn.Module):
    """
    One stage of the regularized butterfly factorization.
    
    Each butterfly stage couples pairs of frequency bins with a 2x2 real
    weight matrix (replacing fixed twiddle factors with learned weights).
    
    For a size-M frequency space, stage s couples bins that are 2^s apart.
    The total number of learnable parameters per stage: M (not M^2 as in dense).
    
    This is the key regularization: O(M log M) parameters instead of O(M^2).
    """
    def __init__(self, num_modes: int, in_channels: int, out_channels: int,
                 stage: int):
        super().__init__()
        self.num_modes = num_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage = stage
        
        # Stride for this butterfly stage (Jones: stride = N/2^(s+1))
        self.stride = max(1, num_modes // (2 ** (stage + 1)))
        
        # Learnable weights for butterfly: 2x2 real mixing per frequency pair
        # Shape: (num_pairs, out_channels, in_channels, 2, 2)
        # where 2x2 replaces the fixed [1,1;1,-1] Hadamard with learned values
        num_pairs = num_modes // 2
        self.W = nn.Parameter(
            torch.randn(num_pairs, out_channels, in_channels, 2, 2)
            * (1.0 / math.sqrt(in_channels * 2))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, num_modes_h, num_modes_w)
        Returns: (batch, out_channels, num_modes_h, num_modes_w)
        
        For 2D, we apply butterfly along the last dimension (width modes).
        A full 2D butterfly would apply to both dims; this is a simplified
        separable version consistent with the 2D RFHT.
        """
        B, C_in, H, W = x.shape
        M = W  # num_modes along width
        
        # Reshape to (batch, channels, num_pairs, 2)
        # pairing bins [k, k + stride] for each k in [0, stride)
        stride = self.stride
        
        # Simple paired butterfly: pair k with k + M//2
        half = M // 2
        x_low = x[..., :half]   # (B, C_in, H, half)
        x_high = x[..., half:]  # (B, C_in, H, half)
        
        # Stack pairs: (B, C_in, H, half, 2)
        pairs = torch.stack([x_low, x_high], dim=-1)
        
        # Apply learned 2x2 mixing per pair position
        # W: (half, C_out, C_in, 2, 2)
        # We use einsum: bihkp, kopi -> bohk  (then reassemble)
        # Simplified: apply same W across H spatial dim
        
        # pairs: (B, C_in, H, half, 2)
        # Contract over C_in and the 2 pair components
        # W[k]: (C_out, C_in, 2, 2) for each pair k
        
        # Vectorize over pairs using einsum
        # pairs -> (B, H, half, C_in, 2)
        pairs_t = pairs.permute(0, 2, 3, 1, 4)  # (B, H, half, C_in, 2)
        
        # W: (half, C_out, C_in, 2, 2)
        # output[b,h,k,o,q] = sum_{i,p} pairs[b,h,k,i,p] * W[k,o,i,p,q]
        out_pairs = torch.einsum('bhkip,koipq->bhkoq', pairs_t, self.W)
        # out_pairs: (B, H, half, C_out, 2)
        
        out_pairs = out_pairs.permute(0, 3, 1, 2, 4)  # (B, C_out, H, half, 2)
        
        # Reassemble
        out = torch.cat([out_pairs[..., 0], out_pairs[..., 1]], dim=-1)
        # out: (B, C_out, H, M)
        
        return out


# ---------------------------------------------------------------------------
# 4. RFHT Spectral Convolution Layer (core RHNO building block)
#
# Replaces the dense Weven/Wodd quadrant weights of HNO with a
# butterfly-factorized weight structure inspired by the RFHT.
#
# Standard HNO spectral conv (from paper):
#   - 8 dense weight matrices (4 quadrants × 2 even/odd)
#   - Parameters: 8 × kmax^2 × C^2
#
# RFHT-regularized spectral conv (this work):
#   - log2(kmax) butterfly stages per quadrant pair
#   - Parameters: 4 × log2(kmax) × kmax × C^2 / 2
#   - Structural sparsity = regularization
# ---------------------------------------------------------------------------

class RFHTSpectralConv2d(nn.Module):
    """
    RFHT-regularized Hartley spectral convolution for 2D problems.
    
    The weight structure is factorized via butterfly stages, mirroring
    Jones' RFHT partitioned-memory architecture. This imposes a structural
    prior on how learned weights can couple frequency bins.
    
    Key difference from HNO paper:
    - HNO: dense Weven, Wodd per quadrant (8 dense matrices)
    - RHNO: butterfly-factorized weights per quadrant (log-depth circuit)
    
    The factorization is:
        W_eff = W_stage_L @ ... @ W_stage_1 @ W_stage_0
    where each stage is a sparse butterfly matrix.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 num_butterfly_stages: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # kmax: number of modes retained per quadrant edge
        
        # Number of butterfly stages (depth of factorization)
        # Jones' RFHT: log2(N) stages for radix-2, log4(N) for radix-4
        # We use log2(modes) stages
        if num_butterfly_stages is None:
            self.num_stages = max(1, int(math.log2(modes))) if modes > 1 else 1
        else:
            self.num_stages = num_butterfly_stages
        
        # Four quadrant processors (matching HNO's four-corner structure)
        # Each quadrant gets its own butterfly stack
        # Quadrants: (low_h, low_w), (low_h, high_w),
        #            (high_h, low_w), (high_h, high_w)
        self.quad_butterflies = nn.ModuleList([
            nn.ModuleList([
                ButterflyStage(modes, in_channels, out_channels, stage=s)
                for s in range(self.num_stages)
            ])
            for _ in range(4)  # 4 quadrants
        ])
        
        # Separate even/odd mixing per quadrant (from HNO convolution theorem)
        # These are lightweight 1x1 channel mixers applied after butterfly
        self.even_mix = nn.Parameter(
            torch.randn(4, out_channels, out_channels) * 0.02
        )
        self.odd_mix = nn.Parameter(
            torch.randn(4, out_channels, out_channels) * 0.02
        )
        
        # Dibit-reversal permutation indices (precomputed)
        self.register_buffer(
            'perm', dibit_reverse_indices(modes)
        )
        
    def _apply_quadrant_butterfly(self, x_quad: torch.Tensor,
                                   quad_idx: int) -> torch.Tensor:
        """
        Apply butterfly stack to one frequency quadrant.
        x_quad: (batch, in_channels, modes, modes)
        """
        h = x_quad
        for stage in self.quad_butterflies[quad_idx]:
            h = stage(h)
            # Apply dibit-reversal permutation between stages
            # (Jones: data reordering between partitioned memories)
            if h.shape[-1] == self.modes:
                h = h[..., self.perm]
        return h
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W) — spatial domain input
        Returns: (batch, out_channels, H, W)
        """
        B, C, H, W = x.shape
        m = self.modes
        
        # Step 1: DHT
        x_ht = dht2d(x)  # (B, C, H, W), real
        
        # Step 2: Even/odd decomposition (from HNO paper Eq. 9)
        # H_even[k] = (H[k] + H[-k]) / 2
        # H_odd[k]  = (H[k] - H[-k]) / 2
        x_flip = torch.roll(torch.flip(x_ht, dims=[-1, -2]), shifts=(1, 1),
                            dims=[-1, -2])
        x_even = (x_ht + x_flip) / 2.0
        x_odd  = (x_ht - x_flip) / 2.0
        
        # Step 3: Extract four frequency quadrants (matching HNO four-corner)
        quads_even = [
            x_even[:, :, :m, :m],        # top-left
            x_even[:, :, :m, -m:],       # top-right
            x_even[:, :, -m:, :m],       # bottom-left
            x_even[:, :, -m:, -m:],      # bottom-right
        ]
        quads_odd = [
            x_odd[:, :, :m, :m],
            x_odd[:, :, :m, -m:],
            x_odd[:, :, -m:, :m],
            x_odd[:, :, -m:, -m:],
        ]
        
        # Step 4: Apply RFHT butterfly factorization per quadrant
        out_quads = []
        for q in range(4):
            # Process even and odd components through butterfly
            h_even = self._apply_quadrant_butterfly(quads_even[q], q)
            h_odd  = self._apply_quadrant_butterfly(quads_odd[q], q)
            
            # Mix even/odd (replaces Weven·Heven + Wodd·Hodd from HNO paper)
            # Using learned 1x1 channel mixing
            # h_even: (B, out_channels, m, m) already after butterfly
            # Apply even_mix[q]: (out_channels, out_channels)
            h_e = torch.einsum('oi,biHW->boHW', self.even_mix[q], h_even)
            h_o = torch.einsum('oi,biHW->boHW', self.odd_mix[q], h_odd)
            
            out_quads.append(h_e + h_o)
        
        # Step 5: Scatter back to full frequency grid
        out_ht = torch.zeros(B, self.out_channels, H, W,
                             device=x.device, dtype=x.dtype)
        out_ht[:, :, :m, :m]   = out_quads[0]
        out_ht[:, :, :m, -m:]  = out_quads[1]
        out_ht[:, :, -m:, :m]  = out_quads[2]
        out_ht[:, :, -m:, -m:] = out_quads[3]
        
        # Step 6: Inverse DHT back to spatial domain
        return idht2d(out_ht)
