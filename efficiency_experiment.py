"""
efficiency_experiment.py -- SELF-CONTAINED. The positive (Jones-spirit) result:
a computationally-regularized REAL spectral operator that matches FNO accuracy
on aligned (elliptic) operators at lower params / FLOPs / memory.
=====================================================================
RegHNO = FNO's transform (rfft/irfft, identical cost) with REAL per-mode
weights instead of complex. For a real-symmetric operator (elliptic Green's
function -- Allerton Thm 1) the complex weights' imaginary part vanishes at
the optimum, so RegHNO never allocates it:

  * half the spectral parameters        (W^2 m^2 real vs FNO's 2 W^2 m^2)
  * ~half the spectral-multiply FLOPs    (2 real matmuls vs a complex matmul)
  * identical transform cost             (rfft/irfft, NOT fftn like the 4/8-corner HNO)
  * real spectral coefficients           (half the spectral memory traffic)

The harness trains FNO / HNO / RegHNO at matched width on an elliptic PDE,
then profiles params, analytic spectral MACs, wall-clock/forward, and peak
GPU memory. The claim, if it holds: RegHNO matches FNO accuracy at ~2x lower
spectral cost on the aligned operators (and only there -- see Darcy/FNO).

  python efficiency_experiment.py                 # poisson/grf, depth 3
  python efficiency_experiment.py --smoke
=====================================================================
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

import spectral_operators as _so
from spectral_operators import (NeuralOperator2d, add_grid, count_params,
                                 TrainConfig, train_eval, DEVICE)
from elliptic_pdes import make_elliptic


# --------------------------------------------------------------------------- #
#  RegHNO: regularized real spectral conv (rfft transform, REAL weights)
# --------------------------------------------------------------------------- #
class RegSpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch + out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))

    @staticmethod
    def _rmul(xc, w):                      # xc complex (B,in,m,m); w real (in,out,m,m)
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


_ORIG_MAKE = getattr(_so, "_orig_make_operator", _so.make_operator)
_so._orig_make_operator = _ORIG_MAKE


def make_operator(kind, in_channels=3, width=32, nlayers=4, modes=12):
    if str(kind).lower() in ("reghno", "reg", "real_fno"):
        fac = lambda w: RegSpectralConv2d(w, w, modes, modes)
        return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)
    return _ORIG_MAKE(kind, in_channels, width, nlayers, modes)


_so.make_operator = make_operator


# --------------------------------------------------------------------------- #
#  Efficiency metrics
# --------------------------------------------------------------------------- #
def spectral_macs(op, width, modes, nlayers):
    """Real multiply-accumulates in the spectral-WEIGHT path per forward pass."""
    m2 = modes * modes
    per = {"fno": 8 * width * width * m2,        # 2 corners x complex matmul (4 real MACs)
           "reghno": 4 * width * width * m2,     # 2 corners x 2 real matmuls
           "hno": 8 * width * width * m2}        # 4 corners x (even+odd) real matmuls
    return per.get(op, per["fno"]) * nlayers


def transform_kind(op):
    return "fftn (full complex)" if op == "hno" else "rfft (half)"


@torch.no_grad()
def profile(op, width, modes, depth, s=128, batch=8, reps=20):
    model = make_operator(op, 3, width, depth, modes).to(DEVICE)
    x = add_grid(torch.randn(batch, s, s, device=DEVICE))
    cuda = DEVICE.type == "cuda"
    for _ in range(3):
        model(x)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps * 1e3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if cuda else float("nan")
    return dict(params=count_params(model), macs=spectral_macs(op, width, modes, depth),
                fwd_ms=dt, peak_mb=peak, transform=transform_kind(op))


# --------------------------------------------------------------------------- #
#  Experiment: accuracy + efficiency at matched width
# --------------------------------------------------------------------------- #
def run_efficiency(pde="poisson", ic="grf", width=32, modes=12, depth=3,
                   n_train=256, n_test=64, s=128, epochs=200, seeds=(0, 1, 2),
                   operators=("fno", "hno", "reghno"), save_dir="./efficiency_results"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"device={DEVICE}; efficiency on {pde}/{ic}  width={width} modes={modes} "
          f"depth={depth} N={n_train}+{n_test} ep={epochs} x{len(seeds)} seeds")

    data = {}
    for sd in seeds:
        f, u = make_elliptic(pde, ic, n_train + n_test, s, seed=sd)
        data[sd] = (f[:n_train], u[:n_train], f[n_train:], u[n_train:])

    res = {}
    for op in operators:
        bests = []
        for sd in seeds:
            xtr, ytr, xte, yte = data[sd]
            cfg = TrainConfig(operator=op, width=width, modes=modes, nlayers=depth,
                              epochs=epochs, batch_size=16, lr=1e-3, weight_decay=1e-4,
                              grad_clip=1.0, align_lambda=0.0, scheduler="step",
                              early_stop=False, seed=sd)
            _, r = train_eval(xtr, ytr, xte, yte, cfg, dataset_name=f"{pde}_{ic}", verbose=False)
            bests.append(r["best_test_relL2"])
        prof = profile(op, width, modes, depth, s=s)
        res[op] = dict(relL2=float(np.mean(bests)), std=float(np.std(bests)), **prof)
        json.dump(res[op], open(os.path.join(save_dir, f"eff_{pde}_{ic}_{op}.json"), "w"), indent=2)

    print("\n" + "=" * 100)
    print(f"EFFICIENCY  {pde}/{ic}  (matched width={width})")
    print("=" * 100)
    print(f"{'operator':>10}{'relL2':>11}{'params':>12}{'specMACs':>14}"
          f"{'fwd ms':>10}{'peak MB':>10}  transform")
    for op in operators:
        r = res[op]
        print(f"{op:>10}{r['relL2']:>8.4f}+-{r['std']:<3.3f}{r['params']:>12,}"
              f"{r['macs']:>14,}{r['fwd_ms']:>10.2f}{r['peak_mb']:>10.1f}  {r['transform']}")

    if "reghno" in res and "fno" in res:
        a, f = res["reghno"], res["fno"]
        d_acc = (a["relL2"] - f["relL2"]) / f["relL2"] * 100
        print("\n  RegHNO vs FNO  (the win, on this aligned operator):")
        print(f"    accuracy:      {d_acc:+.1f}%  relL2 ({'matched/better' if a['relL2'] <= f['relL2']*1.03 else 'worse'})")
        print(f"    parameters:    {f['params']/a['params']:.2f}x fewer  ({f['params']:,} -> {a['params']:,})")
        print(f"    spectral MACs: {f['macs']/a['macs']:.2f}x fewer")
        if DEVICE.type == "cuda":
            print(f"    wall-clock:    {f['fwd_ms']/a['fwd_ms']:.2f}x  | peak mem {f['peak_mb']/a['peak_mb']:.2f}x")
        verdict = ("WIN: matched accuracy at lower cost" if a["relL2"] <= f["relL2"] * 1.03
                   else "no win: RegHNO loses accuracy here")
        print(f"    => {verdict}")
    print("=" * 100)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="poisson", choices=["poisson", "biharmonic"])
    ap.add_argument("--ic", default="grf", choices=["grf", "eigenfunction", "bump"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(">>> SMOKE (tiny, CPU) <<<")
        run_efficiency(args.pde, args.ic, width=16, modes=6, depth=2,
                       n_train=48, n_test=16, s=32, epochs=8, seeds=tuple(args.seeds[:1]))
    else:
        run_efficiency(args.pde, args.ic, seeds=tuple(args.seeds))