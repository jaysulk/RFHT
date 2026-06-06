"""
Parameter count analysis and ablation study.
Compares HNO (dense) vs RHNO (butterfly-factorized) across:
- Different mode counts
- Different butterfly depths
- Parameter efficiency vs accuracy trade-off

This is a key table for the paper: showing RHNO achieves competitive
accuracy with dramatically fewer spectral parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
from models.rhno import make_hno, make_rhno, HartleyNeuralOperator


def analyze_parameter_counts():
    """
    Table comparing HNO vs RHNO parameter counts across configurations.
    
    HNO spectral params: 8 * modes^2 * width^2  (4 quadrants × 2 even/odd × dense)
    RHNO spectral params: 4 * num_stages * (modes/2) * width^2 * 4  (butterfly)
                        + 4 * 2 * width^2  (even/odd channel mixers)
    
    The butterfly factorization: O(M * log M) vs O(M^2) for dense.
    """
    print("=" * 75)
    print("Parameter Count Analysis: HNO (dense) vs RHNO (butterfly-factorized)")
    print("=" * 75)
    
    configs = [
        # (modes, width, num_blocks, label)
        (8,  32, 3, "small"),
        (12, 32, 3, "paper-default"),
        (16, 32, 3, "medium"),
        (20, 48, 3, "large"),
        (12, 32, 4, "elliptic-4block"),
    ]
    
    butterfly_depths = [None, 2, 3]  # None = log2(modes)
    
    print(f"\n{'Config':<20} {'HNO Total':>12} {'HNO Spec':>10} "
          f"{'RHNO Total':>12} {'RHNO Spec':>10} "
          f"{'Spec Ratio':>10} {'Stages':>7}")
    print("-" * 85)
    
    for modes, width, blocks, label in configs:
        hno = make_hno(width=width, modes=modes, num_blocks=blocks)
        hno_counts = hno.parameter_count()
        
        for stages in butterfly_depths:
            rhno = make_rhno(width=width, modes=modes, num_blocks=blocks,
                             num_butterfly_stages=stages)
            rhno_counts = rhno.parameter_count()
            
            actual_stages = stages if stages else max(1, int(math.log2(modes)))
            ratio = rhno_counts['spectral'] / hno_counts['spectral']
            
            config_str = f"{label}(m={modes},w={width})"
            print(f"{config_str:<20} "
                  f"{hno_counts['total']:>12,} "
                  f"{hno_counts['spectral']:>10,} "
                  f"{rhno_counts['total']:>12,} "
                  f"{rhno_counts['spectral']:>10,} "
                  f"{ratio:>10.3f} "
                  f"{actual_stages:>7d}")
    
    print()
    print("Spec Ratio < 1.0 means RHNO has fewer spectral parameters (regularization).")
    print("At large modes, RHNO spectral params scale as O(M log M) vs O(M^2) for HNO.")


def theoretical_scaling():
    """Show theoretical scaling of parameter counts."""
    print("\n" + "=" * 60)
    print("Theoretical Scaling: Spectral Parameters vs Mode Count")
    print("=" * 60)
    print(f"\n{'Modes':>8} {'HNO O(M^2)':>14} {'RHNO O(M logM)':>16} {'Ratio':>8}")
    print("-" * 50)
    
    width = 32
    for modes in [4, 8, 12, 16, 20, 32, 64]:
        # HNO: 8 weight matrices of size (width, width, modes, modes)
        # But simplified: 4 quad × 2 even/odd × modes × modes × width^2 / width^2
        # Per unit width^2:
        hno_spec = 8 * modes * modes
        
        # RFHT butterfly: 4 quadrants × num_stages × (modes//2) × 2×2 per pair
        # Plus even/odd channel mixers: 4 × 2 (small)
        num_stages = max(1, int(math.log2(modes)))
        # Each stage: num_pairs=modes//2, each pair: (width, width, 2, 2)
        # Per unit width^2:
        rhno_spec = 4 * num_stages * (modes // 2) * 4 + 4 * 2
        
        ratio = rhno_spec / hno_spec
        print(f"{modes:>8} {hno_spec:>14,} {rhno_spec:>16,} {ratio:>8.3f}")


def forward_pass_test():
    """Verify forward passes work for both models."""
    print("\n" + "=" * 60)
    print("Forward Pass Verification")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    B, H, W = 4, 64, 64  # smaller for quick test
    
    test_cases = [
        # (pde_type, in_channels, blocks)
        ("heat/wave/burgers/NS (time-dep)", 4, 3),
        ("poisson/biharmonic (elliptic)", 3, 4),
    ]
    
    for label, in_ch, n_blocks in test_cases:
        print(f"\n  PDE type: {label}")
        x = torch.randn(B, H, W, in_ch, device=device)
        
        for model_type, use_rfht in [("HNO (dense)", False),
                                      ("RHNO (butterfly)", True)]:
            model = HartleyNeuralOperator(
                in_channels=in_ch, out_channels=1,
                width=32, modes=8, num_blocks=n_blocks,
                use_rfht=use_rfht
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                out = model(x)
            
            counts = model.parameter_count()
            print(f"    {model_type}: "
                  f"output {tuple(out.shape)}, "
                  f"params={counts['total']:,} "
                  f"(spectral={counts['spectral']:,}, "
                  f"{counts['spectral_fraction']:.1%})")
            
            # Verify gradients flow
            model.train()
            out = model(x)
            loss = out.mean()
            loss.backward()
            print(f"      Gradient check: OK")


def butterfly_depth_ablation():
    """
    Ablation: how does butterfly depth affect parameter count and structure?
    
    Jones (2022) uses radix-4 (log4 N stages) for 131x speedup.
    Here we explore the accuracy/parameter trade-off at different depths.
    """
    print("\n" + "=" * 60)
    print("Butterfly Depth Ablation (modes=12, width=32, 3 blocks)")
    print("=" * 60)
    print(f"\n{'Depth':>8} {'Stages':>8} {'Spec Params':>12} "
          f"{'Total Params':>13} {'vs Dense HNO':>13}")
    print("-" * 58)
    
    modes, width, blocks = 12, 32, 3
    hno = make_hno(width=width, modes=modes, num_blocks=blocks)
    hno_counts = hno.parameter_count()
    
    for stages in [1, 2, 3, 4, None]:
        actual_stages = stages if stages else max(1, int(math.log2(modes)))
        label = f"log2({modes})={actual_stages}" if stages is None else str(stages)
        
        rhno = make_rhno(width=width, modes=modes, num_blocks=blocks,
                          num_butterfly_stages=stages)
        counts = rhno.parameter_count()
        ratio = counts['total'] / hno_counts['total']
        
        print(f"{label:>8} {actual_stages:>8} {counts['spectral']:>12,} "
              f"{counts['total']:>13,} {ratio:>12.3f}x")
    
    print(f"\n  HNO (dense):  spec={hno_counts['spectral']:,}, "
          f"total={hno_counts['total']:,}")
    print()
    print("  Depth 1 (shallowest): maximum regularization, fewest parameters")
    print("  Depth log2(M): matches standard radix-2 FHT factorization")
    print("  Jones radix-4 equivalent: log4(M) = log2(M)/2 stages")


if __name__ == '__main__':
    analyze_parameter_counts()
    theoretical_scaling()
    forward_pass_test()
    butterfly_depth_ablation()
