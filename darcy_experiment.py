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
from spectral_operators import (HartleyConv2d, NeuralOperator2d, count_params,
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


# --- monarch-aware make_operator; patches the global so train_eval/match_width see it ---
_ORIG_MAKE = getattr(_so, "_orig_make_operator", _so.make_operator)
_so._orig_make_operator = _ORIG_MAKE          # stash once (idempotent across re-runs)


def make_operator(kind, in_channels=3, width=32, nlayers=4, modes=12):
    if str(kind).lower() in ("monarch", "monarch_hno", "monarch_ours"):
        fac = lambda w: MonarchHartleyConv2d(w, w, modes, modes)
        return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)
    return _ORIG_MAKE(kind, in_channels, width, nlayers, modes)


_so.make_operator = make_operator             # train_eval / match_width resolve this at call time


# --------------------------------------------------------------------------- #
def run_darcy(operators=("fno", "hno", "learnable_hno", "monarch"),
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
            if op in ("learnable_hno", "monarch"):
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
        for op in ("learnable_hno", "monarch"):
            if op in rec:
                parts.append(f"{op}: delta={rec[op]['delta']:.3f} edge={rec[op]['edge']:+.4f}"
                             f"{' WIN' if rec[op]['edge'] < -0.005 else ''}")
        print("   " + " | ".join(parts))

    print("\n  verdict:")
    for op in ("learnable_hno", "monarch"):
        if op in recs[0]:
            edges = [r[op]["edge"] for r in recs]
            depths_ = [r["depth"] for r in recs]
            c = float(np.corrcoef(depths_, edges)[0, 1]) if len(edges) > 1 else float("nan")
            best = min(edges)
            tag = "WINS at some depth" if best < -0.005 else "never beats fixed"
            trend = ("edge grows toward 0 with depth (compensation)" if c > 0
                     else "edge does not vanish with depth")
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