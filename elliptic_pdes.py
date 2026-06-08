"""
elliptic_pdes.py  (faithful-reproduction build)
=====================================================================
Imports the v2 harness from spectral_operators.py.

This version replaces the earlier (degenerate) IC generators with the EXACT
generators from the validated Allerton pipeline, and adds a reproduction
runner wired to that paper's configuration. The goal is to first reproduce
the known elliptic result (HNO beats FNO) and thereby validate the harness,
*before* drawing any new conclusions.

Prior-pipeline facts encoded here:
  * Source families (each per-sample max-abs normalized):
      - grf   : Matern GRF, nu=2.5, length_scale=0.15, sigma=1.0
                spectrum (2 nu/l^2 + 4 pi^2 |k|^2)^(-(nu + d/2)), d=2
      - eigen : sum of 3-8 modes, k in [-8,8]^2, amp ~ U(0.5,2.0), random phase
                f += amp * cos(2 pi (kx x + ky y) + phase)
      - bump  : 2-4 Gaussian bumps, centers U(0.2,0.8), sigma U(0.06,0.15),
                amp U(0.5,1.0)  (positive)
  * Task: source f -> solution u ;  f zero-meaned for periodic solvability;
          f and u each per-sample max-abs normalized.
  * Reproduction configs (Optuna-tuned, 128x128, 200 train / 40 test, 200 ep):
      - FNO : modes 20, width 48, lr 9.4e-3, wd 1e-5, clip 0.5, cosine
      - HNO : modes  7, width 32, lr 3.8e-3, wd 1e-6, clip 5.0, step
      (iso-parametric match is via MODES, not width)

Prior reported elliptic relative-L2 (targets to reproduce):
  poisson  grf 0.0038  eigen 0.0301  bump 0.0023
  biharm   grf 0.0041  eigen 0.1451  bump 0.0032
  (FNO: poisson 0.063/0.051/0.038, biharm 0.081/0.296/0.045)

USAGE (Colab)
-------------
    from spectral_operators import *
    from elliptic_pdes import run_elliptic_reproduction
    run_elliptic_reproduction()          # should land near the targets above
=====================================================================
"""

import math
import numpy as np
import torch

from spectral_operators import (
    DEVICE, set_seed, make_operator, count_params, TrainConfig, train_eval,
    save_result, spectral_flatness, register_dataset, RESULTS_DIR,
)

# --------------------------- solvers (exact spectral) ----------------------- #

def _wavenumbers(s, device):
    k = torch.fft.fftfreq(s, d=1.0 / s, device=device)  # integer cycles over [0,1)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    return kx, ky


def solve_poisson(f):
    """-lap u = f on the unit torus (zero-mean solution)."""
    s = f.shape[-1]
    kx, ky = _wavenumbers(s, f.device)
    denom = (2 * math.pi) ** 2 * (kx ** 2 + ky ** 2)
    denom[0, 0] = 1.0
    uh = torch.fft.fft2(f) / denom
    uh[..., 0, 0] = 0.0
    return torch.fft.ifft2(uh).real


def solve_biharmonic(f):
    """lap^2 u = f on the unit torus (zero-mean solution)."""
    s = f.shape[-1]
    kx, ky = _wavenumbers(s, f.device)
    denom = (2 * math.pi) ** 4 * (kx ** 2 + ky ** 2) ** 2
    denom[0, 0] = 1.0
    uh = torch.fft.fft2(f) / denom
    uh[..., 0, 0] = 0.0
    return torch.fft.ifft2(uh).real


SOLVERS = {"poisson": solve_poisson, "biharmonic": solve_biharmonic}


# --------------------------- IC families (verbatim from prior pipeline) ----- #

def _maxnorm(x):
    return x / (x.abs().amax(dim=(-2, -1), keepdim=True) + 1e-8)


def ic_grf(n, s, device, nu=2.5, length_scale=0.15, sigma=1.0):
    """Matern GRF via spectral coloring of white noise (per-sample max-abs)."""
    kx, ky = _wavenumbers(s, device)
    k_sq = kx ** 2 + ky ** 2
    d = 2
    tau = 2 * nu / (length_scale ** 2)
    spectrum = (tau + 4 * math.pi ** 2 * k_sq) ** (-(nu + d / 2))
    spectrum[0, 0] = 0.0
    sqrt_spec = torch.sqrt(spectrum) * sigma * math.sqrt(s * s)
    noise = torch.randn(n, s, s, dtype=torch.cfloat, device=device)
    u = torch.fft.ifft2(sqrt_spec[None] * noise).real
    return _maxnorm(u)


def ic_eigen(n, s, device, k_max=8, n_modes_range=(3, 8), amp_range=(0.5, 2.0)):
    """Sum of 3-8 random integer Fourier modes with random phase (per-sample max-abs)."""
    xs = torch.linspace(0, 1, s, device=device)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    fields = []
    for _ in range(n):
        nm = int(torch.randint(n_modes_range[0], n_modes_range[1] + 1, (1,)))
        f = torch.zeros(s, s, device=device)
        for _ in range(nm):
            kx = int(torch.randint(-k_max, k_max + 1, (1,)))
            ky = int(torch.randint(-k_max, k_max + 1, (1,)))
            if kx == 0 and ky == 0:
                continue
            amp = float(torch.empty(1).uniform_(*amp_range))
            phase = float(torch.empty(1).uniform_(0, 2 * math.pi))
            f = f + amp * torch.cos(2 * math.pi * (kx * X + ky * Y) + phase)
        fields.append(f)
    return _maxnorm(torch.stack(fields))


def ic_bump(n, s, device, n_bumps=(2, 5), sigma_range=(0.06, 0.15), amp_range=(0.5, 1.0)):
    """Superposition of 2-4 positive Gaussian bumps (per-sample max-abs)."""
    xs = torch.linspace(0, 1, s, device=device)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    fields = []
    for _ in range(n):
        nb = int(torch.randint(n_bumps[0], n_bumps[1] + 1, (1,)))
        f = torch.zeros(s, s, device=device)
        for _ in range(nb):
            cx = float(torch.empty(1).uniform_(0.2, 0.8))
            cy = float(torch.empty(1).uniform_(0.2, 0.8))
            sg = float(torch.empty(1).uniform_(*sigma_range))
            amp = float(torch.empty(1).uniform_(*amp_range))
            f = f + amp * torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sg ** 2))
        fields.append(f)
    return _maxnorm(torch.stack(fields))


IC_FAMILIES = {"grf": ic_grf, "eigen": ic_eigen, "bump": ic_bump}


# --------------------------- dataset builder -------------------------------- #

def make_elliptic(pde, ic, n, s=128, device=DEVICE, seed=0):
    """source f -> solution u. f zero-meaned (solvability); both max-abs normalized."""
    assert pde in SOLVERS and ic in IC_FAMILIES
    set_seed(seed)
    f = IC_FAMILIES[ic](n, s, device)
    f = f - f.mean(dim=(-2, -1), keepdim=True)   # periodic solvability
    u = SOLVERS[pde](f)
    f = _maxnorm(f)
    u = _maxnorm(u)
    return f.float().cpu(), u.float().cpu()


def _split(fu, n_test=40):
    f, u = fu
    n_tr = f.shape[0] - n_test
    return f[:n_tr], u[:n_tr], f[n_tr:], u[n_tr:]


for _pde in SOLVERS:
    for _ic in IC_FAMILIES:
        register_dataset(
            f"{_pde}_{_ic}",
            (lambda p, c: (lambda n=240, s=128: _split(make_elliptic(p, c, n, s))))(_pde, _ic),
        )


# --------------------------- reproduction runner ---------------------------- #

# Exact per-method configs from the Allerton pipeline (Optuna-tuned, iso-param
# match via modes). LHNO mirrors HNO.
REPRO_CONFIGS = {
    "fno":           dict(modes=20, width=48, lr=9.4e-3, weight_decay=1e-5, grad_clip=0.5, scheduler="cosine"),
    "hno":           dict(modes=7,  width=32, lr=3.8e-3, weight_decay=1e-6, grad_clip=5.0, scheduler="step"),
    "learnable_hno": dict(modes=7,  width=32, lr=3.8e-3, weight_decay=1e-6, grad_clip=5.0, scheduler="step"),
}

# Prior reported relative-L2 (targets); ratio = HNO/FNO.
PRIOR_TARGETS = {
    ("poisson", "grf"):   {"FNO": 0.063, "HNO": 0.0038},
    ("poisson", "eigen"): {"FNO": 0.051, "HNO": 0.0301},
    ("poisson", "bump"):  {"FNO": 0.038, "HNO": 0.0023},
    ("biharmonic", "grf"):   {"FNO": 0.081, "HNO": 0.0041},
    ("biharmonic", "eigen"): {"FNO": 0.296, "HNO": 0.1451},
    ("biharmonic", "bump"):  {"FNO": 0.045, "HNO": 0.0032},
}


def run_elliptic_reproduction(s=128, n_train=200, n_test=40, epochs=200,
                              nlayers=4, batch_size=8, seed=42,
                              pdes=("poisson", "biharmonic"),
                              ics=("grf", "eigen", "bump"),
                              operators=("fno", "hno", "learnable_hno")):
    """Reproduce the Allerton elliptic result with the validated generators
    and per-method configs. Success = HNO best_relL2 << FNO, near PRIOR_TARGETS.
    """
    print(f"device={DEVICE}; elliptic REPRODUCTION (s={s}, N={n_train}+{n_test}, "
          f"{epochs} ep, seed {seed})")
    header = f"{'pde':>10} {'ic':>6} {'op':>14} {'relL2':>9} {'prior':>9} {'delta':>7}"
    print(header); print("-" * len(header))

    rows = []
    for pde in pdes:
        for ic in ics:
            f, u = make_elliptic(pde, ic, n_train + n_test, s=s, seed=seed)
            x_tr, y_tr = f[:n_train], u[:n_train]
            x_te, y_te = f[n_train:], u[n_train:]
            for op in operators:
                rc = REPRO_CONFIGS[op]
                cfg = TrainConfig(operator=op, nlayers=nlayers, epochs=epochs,
                                  batch_size=batch_size, seed=seed, **rc)
                _, res = train_eval(x_tr, y_tr, x_te, y_te, cfg,
                                    dataset_name=f"{pde}_{ic}", verbose=False)
                save_result(res, tag=f"repro_{pde}_{ic}_{op}")
                tgt = PRIOR_TARGETS.get((pde, ic), {})
                tgt_v = tgt.get("FNO" if op == "fno" else "HNO", float("nan"))
                rows.append((pde, ic, op, res["best_test_relL2"], tgt_v, res["delta_at_best"]))
                print(f"{pde:>10} {ic:>6} {op:>14} {res['best_test_relL2']:9.4f} "
                      f"{tgt_v:9.4f} {res['delta_at_best']:7.3f}")

    print("\n=== HNO vs FNO (prior: HNO wins all elliptic) ===")
    cells = {}
    for pde, ic, op, e, *_ in rows:
        cells.setdefault((pde, ic), {})[op] = e
    n_hno_win = 0
    for (pde, ic), d in cells.items():
        if "fno" in d and "hno" in d:
            win = "HNO" if d["hno"] < d["fno"] else "FNO"
            n_hno_win += (win == "HNO")
            ratio = d["hno"] / d["fno"]
            print(f"  {pde:>10} {ic:>6}: FNO {d['fno']:.4f}  HNO {d['hno']:.4f}  "
                  f"ratio {ratio:.2f}  -> {win}")
    print(f"\nHNO wins {n_hno_win}/{len(cells)} elliptic cells "
          f"(prior paper: {len(cells)}/{len(cells)}). "
          f"If HNO wins most and approaches the prior column, the harness is validated.")
    return rows


def collect_scatter(results_dir=RESULTS_DIR):
    """Read every result JSON -> rows for plotting (S_flat, relL2, delta)."""
    import os, json
    rows = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fn)) as fh:
            r = json.load(fh)
        rows.append({
            "tag": fn[:-5], "dataset": r.get("dataset"),
            "operator": r["config"]["operator"],
            "S_flat": r.get("spectral_flatness_train"),
            "best_relL2": r.get("best_test_relL2"),
            "delta_at_best": r.get("delta_at_best"),
        })
    return rows


if __name__ == "__main__":
    run_elliptic_reproduction()