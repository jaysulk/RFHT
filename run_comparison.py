"""
Quick comparison: HNO vs RHNO on Poisson equation (GRF ICs).

This is the best-case PDE from the paper (HNO achieves 0.06x vs FNO),
so it's the right place to first test whether RFHT regularization
hurts the elliptic advantage.

Expected result: RHNO should match or beat HNO with 4-6x fewer
spectral parameters, since the butterfly factorization is a natural
structural prior for the real symmetric Poisson Green's function.

Usage:
    python experiments/run_comparison.py --pde poisson --epochs 100
    python experiments/run_comparison.py --pde heat --epochs 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from models.rhno import make_hno, make_rhno
from utils.training import (
    PoissonDataset, HeatDataset, train_model
)


def run_experiment(pde: str = 'poisson',
                   ic_type: str = 'grf',
                   n_samples: int = 200,
                   resolution: int = 64,  # 128 for paper, 64 for quick
                   modes: int = 12,
                   width: int = 32,
                   n_epochs: int = 100,
                   batch_size: int = 8,
                   device: str = 'auto',
                   butterfly_stages: int = None):
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"RHNO vs HNO: {pde.upper()} equation, {ic_type} ICs")
    print(f"Resolution: {resolution}x{resolution}, Modes: {modes}, Width: {width}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # --- Build dataset ---
    if pde == 'poisson':
        dataset = PoissonDataset(
            n_samples=n_samples, resolution=resolution,
            ic_type=ic_type, device=device
        )
        in_channels = 3
        num_blocks = 4  # elliptic: 4 blocks (paper)
    elif pde == 'heat':
        dataset = HeatDataset(
            n_samples=n_samples, resolution=resolution,
            ic_type=ic_type, device=device
        )
        in_channels = 4
        num_blocks = 3  # time-dep: 3 blocks (paper)
    else:
        raise ValueError(f"Unknown PDE: {pde}. Use 'poisson' or 'heat'.")
    
    # Train/test split (paper: 160/40)
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False)
    
    print(f"Train: {n_train} samples, Test: {n_test} samples")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    results = {}
    
    # --- Train HNO (dense, reproduces paper) ---
    print("Training HNO (dense weights, paper baseline)...")
    hno = make_hno(in_channels=in_channels, width=width,
                    modes=modes, num_blocks=num_blocks)
    hno_counts = hno.parameter_count()
    print(f"  Parameters: {hno_counts['total']:,} "
          f"(spectral: {hno_counts['spectral']:,})")
    
    hno_history = train_model(
        hno, train_loader, test_loader,
        n_epochs=n_epochs,
        lr=3.8e-3,          # HNO-optimized (Table 3)
        weight_decay=1e-6,
        clip_grad=5.0,
        scheduler_type='step',
        device=device,
        verbose=True
    )
    results['HNO'] = hno_history
    
    # --- Train RHNO (butterfly-factorized) ---
    print(f"\nTraining RHNO (butterfly-factorized, stages={butterfly_stages})...")
    rhno = make_rhno(in_channels=in_channels, width=width,
                      modes=modes, num_blocks=num_blocks,
                      num_butterfly_stages=butterfly_stages)
    rhno_counts = rhno.parameter_count()
    print(f"  Parameters: {rhno_counts['total']:,} "
          f"(spectral: {rhno_counts['spectral']:,})")
    print(f"  Spectral param ratio vs HNO: "
          f"{rhno_counts['spectral']/hno_counts['spectral']:.3f}x")
    
    # RHNO may need slightly different LR due to even sparser gradients
    rhno_history = train_model(
        rhno, train_loader, test_loader,
        n_epochs=n_epochs,
        lr=3.8e-3,
        weight_decay=1e-6,
        clip_grad=5.0,
        scheduler_type='step',
        device=device,
        verbose=True
    )
    results['RHNO'] = rhno_history
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {pde.upper()} / {ic_type}")
    print(f"{'='*60}")
    
    hno_best = hno_history['best_test_rel_l2']
    rhno_best = rhno_history['best_test_rel_l2']
    ratio = rhno_best / hno_best
    
    print(f"\n  HNO  best test Rel-L2:  {hno_best:.6f}  "
          f"(params: {hno_counts['total']:,})")
    print(f"  RHNO best test Rel-L2:  {rhno_best:.6f}  "
          f"(params: {rhno_counts['total']:,})")
    print(f"\n  RHNO/HNO error ratio: {ratio:.3f}x")
    print(f"  RHNO/HNO param ratio: "
          f"{rhno_counts['total']/hno_counts['total']:.3f}x")
    
    if ratio < 1.0:
        print(f"\n  ✓ RHNO outperforms HNO with {1/ratio:.1f}x lower error "
              f"and {hno_counts['total']/rhno_counts['total']:.1f}x fewer params")
    elif ratio < 1.1:
        print(f"\n  ~ RHNO matches HNO within 10% with "
              f"{hno_counts['total']/rhno_counts['total']:.1f}x fewer params")
    else:
        print(f"\n  ✗ RHNO underperforms — may need tuned LR for butterfly depth")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', default='poisson',
                        choices=['poisson', 'heat'])
    parser.add_argument('--ic', default='grf',
                        choices=['grf', 'eigenfunction', 'gaussian_bump'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--modes', type=int, default=12)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--butterfly_stages', type=int, default=None)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    run_experiment(
        pde=args.pde,
        ic_type=args.ic,
        n_samples=args.n_samples,
        resolution=args.resolution,
        modes=args.modes,
        width=args.width,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        butterfly_stages=args.butterfly_stages,
    )
