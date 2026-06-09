"""
conditioning_sweep.py
=====================================================================
Capacity / conditioning sweep to decide whether a learnable spectral
basis can BEAT the operator-aligned fixed basis (hypothesis B) or merely
RECOVERS it (hypothesis A).

For each one-factor-at-a-time setting of {depth, modes, n_train, epochs}
around a baseline, it trains FNO / HNO / learnable-HNO under an IDENTICAL,
shared-hyperparameter, parameter-matched recipe with align_lambda = 0 (the
learnable basis is fully untethered from its Hartley init), and records:

  * best test relL2 for each operator
  * the learnable basis deviation delta (how far L moved from identity)
  * data-side CONDITIONING of each basis, from the retained-mode
    coefficients of the training inputs:
       - Gram condition number  (lambda_max / lambda_min over the data subspace)
       - effective rank (participation ratio of the Gram spectrum)
       - spectral flatness (Wiener), Fourier and Hartley

Derived per configuration:
  gap_hno_fno      = relL2(HNO) - relL2(FNO)        (<0 => Hartley-favored cell)
  lhno_gain        = relL2(HNO) - relL2(LHNO)       (>0 => learning helped vs HNO)
  lhno_edge        = relL2(LHNO) - min(FNO, HNO)    (<0 => B win: beat BOTH fixed)
  log10_cond_ratio = log10(cond_Fourier / cond_Hartley)  (>0 => Fourier worse-cond.)

Theory (A) predicts:
  - |gap| is largest at LOW capacity and shrinks toward 0 as depth / modes /
    data / epochs grow (the network compensates for a sub-optimal basis);
  - sign(-gap) tracks sign(log10_cond_ratio) (the worse-conditioned basis loses);
  - lhno_edge stays >= ~0 everywhere (learning recovers, never beats, the
    aligned fixed basis).
A clearly negative lhno_edge anywhere is a (B) signal worth chasing.

Run:
  python conditioning_sweep.py                 # full sweep, poisson/grf
  python conditioning_sweep.py --pde biharmonic --ic eigenfunction
  python conditioning_sweep.py --smoke         # tiny CPU end-to-end check
=====================================================================
"""

import os, json, math, argparse
import numpy as np
import torch

from spectral_operators import (DEVICE, make_operator, count_params, match_width,
                                 TrainConfig, train_eval, dht2, spectral_flatness)
from elliptic_pdes import make_elliptic


# --------------------------------------------------------------------------- #
#  Conditioning metrics (data-side, model-free, basis-by-basis)
# --------------------------------------------------------------------------- #
def _fourier_feats(fields, m):
    """Retained low-freq Fourier coefficients (2 corners) as real features."""
    Xf = torch.fft.rfft2(fields)
    c1 = Xf[:, :m, :m]
    c2 = Xf[:, -m:, :m]
    return torch.cat([torch.view_as_real(c1).reshape(fields.shape[0], -1),
                      torch.view_as_real(c2).reshape(fields.shape[0], -1)], dim=1)


def _hartley_feats(fields, m):
    """Retained low-freq Hartley coefficients (4 corners) as real features."""
    H = dht2(fields)
    Hh, Ww = H.shape[-2], H.shape[-1]
    corners = [H[:, :m, :m], H[:, :m, Ww - m:], H[:, Hh - m:, :m], H[:, Hh - m:, Ww - m:]]
    return torch.cat([c.reshape(fields.shape[0], -1) for c in corners], dim=1)


def _gram_cond_rank(X, rel_floor=1e-10):
    """Condition number and effective rank of the data-subspace Gram (N x N).

    Using the N x N Gram X X^T / N is equivalent to the feature second-moment
    on the span of the data, so it stays well-defined when #features > N."""
    X = X.double()
    N = X.shape[0]
    G = (X @ X.t()) / N
    ev = torch.linalg.eigvalsh(G).clamp_min(0.0)          # ascending
    lmax = float(ev[-1])
    if lmax <= 0:
        return float("inf"), 0.0
    pos = ev[ev > rel_floor * lmax]
    cond = lmax / float(pos[0]) if pos.numel() else float("inf")
    eff_rank = float((ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30)).item())
    return cond, eff_rank


def _hartley_flatness(fields, n_modes=16):
    P = (dht2(fields).abs() ** 2)[..., :n_modes, :n_modes].reshape(fields.shape[0], -1) + 1e-12
    return float((torch.exp(torch.log(P).mean(1)) / P.mean(1)).mean())


def data_conditioning(fields, modes):
    fields = fields.detach().float().cpu()
    cf, rf = _gram_cond_rank(_fourier_feats(fields, modes))
    ch, rh = _gram_cond_rank(_hartley_feats(fields, modes))
    return dict(cond_fourier=cf, cond_hartley=ch,
                log10_cond_ratio=math.log10((cf + 1e-30) / (ch + 1e-30)),
                effrank_fourier=rf, effrank_hartley=rh,
                sflat_fourier=float(spectral_flatness(fields)),
                sflat_hartley=_hartley_flatness(fields))


# --------------------------------------------------------------------------- #
#  Shared, controlled training recipe (identical across operators)
# --------------------------------------------------------------------------- #
def _shared_cfg(op, width, modes, depth, epochs, seed):
    return TrainConfig(operator=op, width=width, modes=modes, nlayers=depth,
                       epochs=epochs, batch_size=16, lr=1e-3, weight_decay=1e-4,
                       grad_clip=1.0, align_lambda=0.0, scheduler="step",
                       early_stop=False, seed=seed)


def run_config(pde, ic, depth, modes, n_train, epochs,
               s=128, n_test=64, fno_width=32, seed=0,
               ops=("fno", "hno", "learnable_hno"), data_cache=None):
    key = (pde, ic, n_train + n_test, s, seed)
    if data_cache is not None and key in data_cache:
        f, u = data_cache[key]
    else:
        f, u = make_elliptic(pde, ic, n_train + n_test, s, seed=seed)
        if data_cache is not None:
            data_cache[key] = (f, u)
    xtr, ytr, xte, yte = f[:n_train], u[:n_train], f[n_train:n_train + n_test], u[n_train:n_train + n_test]

    cond = data_conditioning(xtr, modes)
    target = count_params(make_operator("fno", 3, fno_width, depth, modes))

    res = {}
    for op in ops:
        w = fno_width if op == "fno" else match_width(op, target, modes, depth)
        cfg = _shared_cfg(op, w, modes, depth, epochs, seed)
        _, r = train_eval(xtr, ytr, xte, yte, cfg, dataset_name=f"{pde}_{ic}", verbose=False)
        res[op] = r

    rec = dict(pde=pde, ic=ic, depth=depth, modes=modes, n_train=n_train, epochs=epochs)
    for op in ops:
        rec[f"{op}_best"] = res[op]["best_test_relL2"]
        rec[f"{op}_delta"] = res[op]["delta_at_best"]
        rec[f"{op}_params"] = res[op]["n_params"]
        rec[f"{op}_best_epoch"] = res[op]["best_epoch"]
    rec.update(cond)
    fno, hno, lh = rec["fno_best"], rec["hno_best"], rec["learnable_hno_best"]
    rec["gap_hno_fno"] = hno - fno
    rec["lhno_gain"] = hno - lh
    rec["lhno_edge"] = lh - min(fno, hno)
    return rec


# --------------------------------------------------------------------------- #
#  One-factor-at-a-time sweep
# --------------------------------------------------------------------------- #
DEFAULT_BASELINE = dict(depth=3, modes=12, n_train=256, epochs=200)
DEFAULT_AXES = dict(depth=[1, 2, 3, 4], modes=[4, 8, 12, 16],
                    n_train=[64, 128, 256, 512], epochs=[25, 50, 100, 200])

SMOKE_BASELINE = dict(depth=1, modes=4, n_train=32, epochs=5)
SMOKE_AXES = dict(depth=[1, 2], modes=[4, 8], n_train=[32, 64], epochs=[5, 10])


def sweep(pde="poisson", ic="grf", baseline=None, axes=None,
          s=128, n_test=64, seed=0, save_dir="./sweep_results"):
    baseline = baseline or DEFAULT_BASELINE
    axes = axes or DEFAULT_AXES
    os.makedirs(save_dir, exist_ok=True)
    data_cache = {}

    configs = []
    for axis, vals in axes.items():
        for v in vals:
            cfg = dict(baseline); cfg[axis] = v
            configs.append((axis, v, cfg))

    run_cache = {}   # signature -> rec, so the shared baseline trains once
    recs = []
    for i, (axis, v, cfg) in enumerate(configs):
        sig = (cfg["depth"], cfg["modes"], cfg["n_train"], cfg["epochs"])
        tag = "cached" if sig in run_cache else "train "
        print(f"[{i+1:2d}/{len(configs)}] {pde}/{ic}  {axis}={v:<4} [{tag}] "
              f"(depth={cfg['depth']} modes={cfg['modes']} N={cfg['n_train']} ep={cfg['epochs']})",
              flush=True)
        if sig in run_cache:
            rec = dict(run_cache[sig])
        else:
            rec = run_config(pde, ic, cfg["depth"], cfg["modes"], cfg["n_train"], cfg["epochs"],
                             s=s, n_test=n_test, seed=seed, data_cache=data_cache)
            run_cache[sig] = rec
        rec = dict(rec)
        rec["axis"], rec["axis_value"] = axis, v
        recs.append(rec)
        json.dump(rec, open(os.path.join(save_dir, f"sweep_{pde}_{ic}_{axis}_{v}.json"), "w"), indent=2)

    json.dump(recs, open(os.path.join(save_dir, f"sweep_{pde}_{ic}_ALL.json"), "w"), indent=2)
    summarize(recs, pde, ic)
    return recs


# --------------------------------------------------------------------------- #
#  Readout
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(recs, pde, ic):
    print("\n" + "=" * 100)
    print(f"SUMMARY  {pde}/{ic}   (gap<0 => Hartley-favored; Ledge<0 => learnable beats both fixed)")
    print("=" * 100)
    for axis in ["depth", "modes", "n_train", "epochs"]:
        rr = sorted([r for r in recs if r["axis"] == axis], key=lambda r: r["axis_value"])
        if not rr:
            continue
        print(f"\n-- axis: {axis} --")
        print(f"{axis:>8}{'FNO':>9}{'HNO':>9}{'LHNO':>9}{'gap(H-F)':>10}"
              f"{'Lgain':>8}{'Ledge':>8}{'delta':>7}{'log cF/cH':>11}{'sflatF':>8}{'sflatH':>8}")
        for r in rr:
            print(f"{r['axis_value']:>8}{r['fno_best']:>9.4f}{r['hno_best']:>9.4f}"
                  f"{r['learnable_hno_best']:>9.4f}{r['gap_hno_fno']:>10.4f}"
                  f"{r['lhno_gain']:>8.4f}{r['lhno_edge']:>8.4f}{r['learnable_hno_delta']:>7.3f}"
                  f"{r['log10_cond_ratio']:>11.2f}{r['sflat_fourier']:>8.3f}{r['sflat_hartley']:>8.3f}")

    print("\n" + "-" * 100)
    neg_gap = [-r["gap_hno_fno"] for r in recs]
    lcr = [r["log10_cond_ratio"] for r in recs]
    print(f"corr( -gap , log10(cond_F/cond_H) ) = {_pearson(neg_gap, lcr):+.3f}   "
          f"(>0 supports: the worse-conditioned basis loses)")

    for axis in ["depth", "modes", "n_train", "epochs"]:
        rr = sorted([r for r in recs if r["axis"] == axis], key=lambda r: r["axis_value"])
        if len(rr) >= 2:
            absgap = [abs(r["gap_hno_fno"]) for r in rr]
            c = _pearson([r["axis_value"] for r in rr], absgap)
            trend = "shrinks" if c < 0 else "grows"
            print(f"  |gap| vs {axis:>8}: corr={c:+.3f} ({trend} with capacity)  "
                  f"vals={['%.4f' % g for g in absgap]}")

    edges = [r["lhno_edge"] for r in recs]
    j = int(np.argmin(edges))
    b = recs[j]
    print(f"\n  best learnable edge over fixed basis: {edges[j]:+.4f}  "
          f"(depth={b['depth']} modes={b['modes']} N={b['n_train']} ep={b['epochs']})")
    if edges[j] < -0.005:
        print("  => (B) SIGNAL: learnable basis beats BOTH fixed bases in at least one regime.")
    else:
        print("  => (A): learnable basis does not clearly beat the aligned fixed basis anywhere.")
    print("=" * 100)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="poisson", choices=["poisson", "biharmonic"])
    ap.add_argument("--ic", default="grf", choices=["grf", "eigenfunction", "bump"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        print(">>> SMOKE MODE (tiny, CPU-friendly) <<<")
        sweep(args.pde, args.ic, baseline=SMOKE_BASELINE, axes=SMOKE_AXES,
              s=32, n_test=16, seed=args.seed)
    else:
        sweep(args.pde, args.ic, seed=args.seed)