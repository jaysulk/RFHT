"""
darcy_experiment.py -- SELF-CONTAINED. Does a learnable/Monarch transform beat
fixed bases on a NON-diagonal operator (Darcy), and does depth wash it out?
=====================================================================
This file defines the Monarch layer inline and patches make_operator itself,
so it depends only on your spectral_operators.py + darcy.py (no monarch_operators
import -> nothing to go stale in a Colab session).

Runs FNO / HNO / learnable-HNO / Monarch on Darcy flow across network depth,
parameter-matched and seed-averaged under the identical shared recipe
(align_lambda = 0). Darcy's solution operator is non-diagonal, so unlike
Poisson/heat/wave there is a real representational floor for the learnable
transforms to close (see b_sanity.py). Open question: does the advantage
survive depth, which lets the network compensate?

Read the output: edge = relL2(learnable) - min(relL2(FNO), relL2(HNO)) per depth.
  edge < 0 at usable depth  => learnable/Monarch genuinely wins (B alive)
  edge -> 0 as depth grows   => depth compensates; basis advantage washes out

  python darcy_experiment.py            # full: depths 1-4, 3 seeds
  python darcy_experiment.py --smoke    # tiny CPU check
=====================================================================
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn

import spectral_operators as _so
from spectral_operators import (HartleyConv2d, SpectralConv2d, NeuralOperator2d, count_params,
                                 match_width, TrainConfig, train_eval, DEVICE)
from darcy import make_darcy


# --------------------------------------------------------------------------- #
#  Order-2 Monarch spectral layer (inline; starts at Hartley, non-separable)
# --------------------------------------------------------------------------- #
class MonarchHartleyConv2d(HartleyConv2d):
    """HNO + learnable order-2 Monarch transform on the retained m x m block.
    B1, B2 are (m, m, m) block tensors (a distinct m x m matrix per row),
    initialized to identity so the layer starts exactly at Hartley but can
    represent non-separable transforms (DFT/DHT and beyond)."""

    def __init__(self, in_ch, out_ch, modes1, modes2, share_across_corners=True):
        super().__init__(in_ch, out_ch, modes1, modes2)
        assert modes1 == modes2, "Monarch layer uses a square m x m mode block"
        m = modes1
        self.m = m
        eye_blocks = torch.eye(m).unsqueeze(0).repeat(m, 1, 1).clone()
        self.B1 = nn.Parameter(eye_blocks.clone())
        self.B2 = nn.Parameter(eye_blocks.clone())

    def _monarch(self, X):                                   # X: (B, C, m, m)
        Y = torch.einsum("piq,bcpq->bcpi", self.B1, X)       # block-diag along rows
        Y = Y.transpose(-1, -2)                              # permutation
        Z = torch.einsum("ikp,bcip->bcik", self.B2, Y)       # block-diag along new rows
        return Z.transpose(-1, -2)

    def _mix(self, e, o, i):
        return self._monarch(e), self._monarch(o)

    def alignment_penalty(self):
        I = torch.eye(self.m, device=self.B1.device).unsqueeze(0)
        return ((self.B1 - I) ** 2).sum() + ((self.B2 - I) ** 2).sum()

    @torch.no_grad()
    def basis_deviation(self):
        I = torch.eye(self.m, device=self.B1.device).unsqueeze(0)
        num = (torch.linalg.matrix_norm(self.B1 - I).mean()
               + torch.linalg.matrix_norm(self.B2 - I).mean())
        return float(num / (2.0 * np.sqrt(self.m)))



class LearnableFourierConv2d(SpectralConv2d):
    """FNO + learnable complex separable mode-mixing, initialized at identity.
    Starts EXACTLY at FNO (the best fixed basis on Darcy) -> tests whether
    learning can improve even the strongest fixed basis."""

    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__(in_ch, out_ch, m1, m2)
        self.L1 = nn.Parameter(torch.stack([torch.eye(m1), torch.zeros(m1, m1)], dim=-1))
        self.L2 = nn.Parameter(torch.stack([torch.eye(m2), torch.zeros(m2, m2)], dim=-1))

    def _mixL(self, blk):                                   # blk: (B,C,m1,m2) complex
        L1 = torch.view_as_complex(self.L1.contiguous())
        L2 = torch.view_as_complex(self.L2.contiguous())
        blk = torch.einsum("mn,bcnj->bcmj", L1, blk)
        return torch.einsum("bcmj,jk->bcmk", blk, L2)

    def forward(self, x):
        B, _, H, W = x.shape
        xft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        w1, w2 = torch.view_as_complex(self.w1), torch.view_as_complex(self.w2)
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1, :m2] = self._cmul(self._mixL(xft[:, :, :m1, :m2]), w1)
        out[:, :, -m1:, :m2] = self._cmul(self._mixL(xft[:, :, -m1:, :m2]), w2)
        return torch.fft.irfft2(out, s=(H, W))

    def alignment_penalty(self):
        L1 = torch.view_as_complex(self.L1.contiguous()); L2 = torch.view_as_complex(self.L2.contiguous())
        I1 = torch.eye(self.modes1, device=L1.device, dtype=L1.dtype)
        I2 = torch.eye(self.modes2, device=L2.device, dtype=L2.dtype)
        return (L1 - I1).abs().pow(2).sum() + (L2 - I2).abs().pow(2).sum()

    @torch.no_grad()
    def basis_deviation(self):
        L1 = torch.view_as_complex(self.L1.contiguous()); L2 = torch.view_as_complex(self.L2.contiguous())
        I1 = torch.eye(self.modes1, device=L1.device, dtype=L1.dtype)
        I2 = torch.eye(self.modes2, device=L2.device, dtype=L2.dtype)
        num = (L1 - I1).abs().pow(2).sum().sqrt() + (L2 - I2).abs().pow(2).sum().sqrt()
        return float(num / (self.modes1 ** 0.5 + self.modes2 ** 0.5))



class RegSpectralConv2d(nn.Module):
    """Regularized real spectral conv: rfft transform (same as FNO) but REAL
    per-mode weights -> half params / half spectral FLOPs. Included here to
    test the BOUNDARY: it should LOSE to FNO off the aligned operators."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch + out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))

    @staticmethod
    def _rmul(xc, w):
        return torch.complex(torch.einsum("bixy,ioxy->boxy", xc.real, w),
                             torch.einsum("bixy,ioxy->boxy", xc.imag, w))

    def forward(self, x):
        B, _, H, W = x.shape
        xft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1, :m2] = self._rmul(xft[:, :, :m1, :m2], self.w1)
        out[:, :, -m1:, :m2] = self._rmul(xft[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))

    def alignment_penalty(self):
        return torch.zeros((), device=self.w1.device)


# --- monarch-aware make_operator; patches the global so train_eval/match_width see it ---
_ORIG_MAKE = getattr(_so, "_orig_make_operator", _so.make_operator)
_so._orig_make_operator = _ORIG_MAKE          # stash once (idempotent across re-runs)


def make_operator(kind, in_channels=3, width=32, nlayers=4, modes=12):
    if str(kind).lower() in ("monarch", "monarch_hno", "monarch_ours"):
        fac = lambda w: MonarchHartleyConv2d(w, w, modes, modes)
        return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)
    if str(kind).lower() in ("learnable_fno", "lfno", "learnable_fourier"):
        fac = lambda w: LearnableFourierConv2d(w, w, modes, modes)
        return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)
    if str(kind).lower() in ("reghno", "reg", "real_fno"):
        fac = lambda w: RegSpectralConv2d(w, w, modes, modes)
        return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)
    return _ORIG_MAKE(kind, in_channels, width, nlayers, modes)


_so.make_operator = make_operator             # train_eval / match_width resolve this at call time


# --------------------------------------------------------------------------- #
def run_darcy(operators=("fno", "hno", "learnable_hno", "monarch", "learnable_fno"),
              depths=(1, 2, 3, 4), n_train=512, n_test=128, s=128, modes=12,
              epochs=200, seeds=(0, 1, 2), fno_width=32, kind="threshold",
              save_dir="./darcy_results"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"device={DEVICE}; Darcy ({kind}) s={s} N={n_train}+{n_test} "
          f"modes={modes} ep={epochs} x{len(seeds)} seeds")

    data = {}
    for sd in seeds:
        a, u = make_darcy(n_train + n_test, s, seed=sd, kind=kind)
        data[sd] = (a[:n_train], u[:n_train], a[n_train:], u[n_train:])

    recs = []
    for depth in depths:
        target = count_params(make_operator("fno", 3, fno_width, depth, modes))
        row = {}
        for op in operators:
            w = fno_width if op == "fno" else match_width(op, target, modes, depth)
            bests, deltas = [], []
            for sd in seeds:
                xtr, ytr, xte, yte = data[sd]
                cfg = TrainConfig(operator=op, width=w, modes=modes, nlayers=depth,
                                  epochs=epochs, batch_size=16, lr=1e-3, weight_decay=1e-4,
                                  grad_clip=1.0, align_lambda=0.0, scheduler="step",
                                  early_stop=False, seed=sd)
                _, r = train_eval(xtr, ytr, xte, yte, cfg, dataset_name="darcy", verbose=False)
                bests.append(r["best_test_relL2"]); deltas.append(r["delta_at_best"])
            row[op] = dict(best=float(np.mean(bests)), std=float(np.std(bests)),
                           delta=float(np.mean(deltas)),
                           params=count_params(make_operator(op, 3, w, depth, modes)))
        fixed = min(row["fno"]["best"], row.get("hno", row["fno"])["best"])
        for op in operators:
            if op in ("learnable_hno", "monarch", "learnable_fno"):
                row[op]["edge"] = row[op]["best"] - fixed
        rec = dict(depth=depth, **{op: row[op] for op in operators})
        recs.append(rec)
        json.dump(rec, open(os.path.join(save_dir, f"darcy_depth{depth}.json"), "w"), indent=2)

    print("\n" + "=" * 92)
    print(f"DARCY ({kind})   relL2 mean+/-std per operator; edge = learnable - best fixed")
    print("=" * 92)
    print(f"{'depth':>6}" + "".join(f"{op:>16}" for op in operators))
    for rec in recs:
        line = f"{rec['depth']:>6}"
        for op in operators:
            line += f"{rec[op]['best']:>9.4f}+-{rec[op]['std']:<4.3f}"
        print(line)
    print("\n  deltas (basis movement) and edges over best fixed basis:")
    for rec in recs:
        parts = [f"d={rec['depth']}"]
        for op in ("learnable_hno", "monarch", "learnable_fno"):
            if op in rec:
                parts.append(f"{op}: delta={rec[op]['delta']:.3f} edge={rec[op]['edge']:+.4f}"
                             f"{' WIN' if rec[op]['edge'] < -0.005 else ''}")
        print("   " + " | ".join(parts))

    print("\n  verdict:")
    for op in ("learnable_hno", "monarch", "learnable_fno"):
        if op in recs[0]:
            edges = [r[op]["edge"] for r in recs]
            depths_ = [r["depth"] for r in recs]
            c = float(np.corrcoef(depths_, edges)[0, 1]) if len(edges) > 1 else float("nan")
            best = min(edges)
            tag = "WINS at some depth" if best < -0.005 else "never beats fixed"
            trend = ("falls further behind fixed as depth grows" if c > 0
                     else "gap to fixed shrinks with depth (compensation)")
            print(f"    {op:>14}: {tag}; best edge {best:+.4f}; {trend} (corr depth,edge={c:+.2f})")
    print("=" * 92)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--kind", default="threshold", choices=["threshold", "lognormal"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    if args.smoke:
        print(">>> SMOKE (tiny, CPU) <<<")
        run_darcy(depths=(1, 2), n_train=48, n_test=16, s=32, modes=6,
                  epochs=8, seeds=tuple(args.seeds[:1]), kind=args.kind)
    else:
        run_darcy(seeds=tuple(args.seeds), kind=args.kind)