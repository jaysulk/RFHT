"""
elliptic_pdes.py
=====================================================================
Elliptic control experiment for the learnable-basis project. Imports the
v2 harness from spectral_operators.py.

Why elliptic first: it is the discriminating test. Theorem 4 predicts
  (a) delta ~= 0 for elliptic at ANY initial condition (the Green's function
      is real + even, so Hartley is already the conditioning-optimal basis;
      the alignment pressure pins the learned basis there), and
  (b) HNO should BEAT FNO on elliptic (real-symmetric kernel),
contrasting with Navier-Stokes (time-dependent, low S_flat) where FNO wins
and the learned basis moves (delta > 0). That CONTRAST is the evidence --
not any single number.

PDEs (exact spectral solves, verified to ~1e-14 residual):
  poisson     :  -lap u = f   ->  u_hat = f_hat / ((2pi)^2 |k|^2)
  biharmonic  :  lap^2 u = f  ->  u_hat = f_hat / ((2pi)^4 |k|^4)

IC families for the source f (span a range of spectral flatness S_flat):
  grf    : broadband Gaussian random field        (high  S_flat)
  eigen  : sparse low-mode Sobolev-weighted field  (low   S_flat)
  bump   : sum of a few Gaussian bumps             (low   S_flat)

USAGE (Colab)
-------------
    from spectral_operators import *
    from elliptic_pdes import run_elliptic_control, collect_scatter
    run_elliptic_control(n_train=512, n_test=128, epochs=200)
    rows = collect_scatter()      # (dataset, operator, S_flat, relL2, delta) for plotting
=====================================================================
"""

import math
import numpy as np
import torch

from spectral_operators import (
    DEVICE, set_seed, grf_2d, make_operator, count_params, match_width,
    TrainConfig, train_eval, save_result, spectral_flatness, register_dataset,
    RESULTS_DIR,
)


# --------------------------- wavenumbers / solvers -------------------------- #

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


# --------------------------- IC families ------------------------------------ #

def _standardize(x):
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    return x / (x.std(dim=(-2, -1), keepdim=True) + 1e-8)


def ic_grf(n, s, device, alpha=2.5, tau=7.0):
    return grf_2d(n, s, alpha=alpha, tau=tau, device=device)


def ic_eigen(n, s, device, n_modes=6, sob=2.0):
    """Sparse low-mode periodic field: random Sobolev-weighted coeffs on |k|<=n_modes."""
    kx, ky = _wavenumbers(s, device)
    mask = ((kx.abs() <= n_modes) & (ky.abs() <= n_modes)).float()
    weight = (1.0 + kx ** 2 + ky ** 2) ** (-sob) * mask
    coeff = (torch.randn(n, s, s, device=device) + 1j * torch.randn(n, s, s, device=device))
    f = torch.fft.ifft2(coeff * weight[None]).real
    return _standardize(f)


def ic_bump(n, s, device, n_bumps=(2, 5), sigma=(0.06, 0.15)):
    """Sum of a few random Gaussian bumps (spatially localized)."""
    xx = torch.linspace(0, 1, s, device=device)
    X, Y = torch.meshgrid(xx, xx, indexing="ij")
    fields = []
    for _ in range(n):
        k = int(torch.randint(n_bumps[0], n_bumps[1] + 1, (1,)).item())
        f = torch.zeros(s, s, device=device)
        for _ in range(k):
            cx, cy = (torch.rand(2, device=device) * 0.6 + 0.2).tolist()
            sg = float(torch.rand(1, device=device) * (sigma[1] - sigma[0]) + sigma[0])
            amp = float(torch.randn(1, device=device))
            f = f + amp * torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sg ** 2))
        fields.append(f)
    return _standardize(torch.stack(fields))


IC_FAMILIES = {"grf": ic_grf, "eigen": ic_eigen, "bump": ic_bump}


# --------------------------- dataset builder -------------------------------- #

def make_elliptic(pde, ic, n, s=64, device=DEVICE, seed=0):
    """Returns (f, u) each (n, s, s) on CPU: input source f -> elliptic solution u."""
    assert pde in SOLVERS and ic in IC_FAMILIES
    set_seed(seed)
    f = IC_FAMILIES[ic](n, s, device)
    u = SOLVERS[pde](f)
    u = _standardize(u)          # relL2 is scale-invariant; standardize for optimization
    return f.float().cpu(), u.float().cpu()


# register the 6 elliptic cells so they're discoverable in DATASET_REGISTRY
for _pde in SOLVERS:
    for _ic in IC_FAMILIES:
        register_dataset(
            f"{_pde}_{_ic}",
            (lambda p, c: (lambda n=640, s=64: _split(make_elliptic(p, c, n, s))))(_pde, _ic),
        )


def _split(fu, n_test_frac=0.2):
    f, u = fu
    n = f.shape[0]
    n_te = int(round(n * n_test_frac))
    n_tr = n - n_te
    return f[:n_tr], u[:n_tr], f[n_tr:], u[n_tr:]


# --------------------------- runner ----------------------------------------- #

def run_elliptic_control(n_train=512, n_test=128, s=64, epochs=200, modes=12,
                         nlayers=4, fno_width=32, param_match=True,
                         pdes=("poisson", "biharmonic"),
                         ics=("grf", "eigen", "bump"),
                         operators=("fno", "hno", "learnable_hno"), seed=0):
    """Elliptic delta-control sweep. nlayers=4 (elliptic) per prior work.

    Prediction to check:
      * delta@best ~= 0 for learnable-HNO in EVERY cell (basis stays at Hartley)
      * HNO best_relL2 <= FNO best_relL2 (HNO advantage on elliptic)
    """
    target = count_params(make_operator("fno", 3, fno_width, nlayers, modes))
    print(f"device={DEVICE}; elliptic control, nlayers={nlayers}, "
          f"param target={target:,d}\n")
    header = f"{'pde':>10} {'ic':>6} {'operator':>14} {'S_flat':>7} {'relL2':>8} {'delta':>7}"
    print(header); print("-" * len(header))

    rows = []
    for pde in pdes:
        for ic in ics:
            f, u = make_elliptic(pde, ic, n_train + n_test, s=s, seed=seed)
            x_tr, y_tr = f[:n_train], u[:n_train]
            x_te, y_te = f[n_train:], u[n_train:]
            sflat = spectral_flatness(x_tr)
            for op in operators:
                width = (fno_width if (op == "fno" or not param_match)
                         else match_width(op, target, modes, nlayers))
                cfg = TrainConfig(operator=op, width=width, modes=modes, nlayers=nlayers,
                                  epochs=epochs, lr=(9e-3 if op == "fno" else 3e-3),
                                  grad_clip=(0.5 if op == "fno" else 5.0), seed=seed)
                _, res = train_eval(x_tr, y_tr, x_te, y_te, cfg,
                                    dataset_name=f"{pde}_{ic}", verbose=False)
                save_result(res, tag=f"{pde}_{ic}_{op}")
                rows.append((pde, ic, op, sflat, res["best_test_relL2"], res["delta_at_best"]))
                print(f"{pde:>10} {ic:>6} {op:>14} {sflat:7.3f} "
                      f"{res['best_test_relL2']:8.4f} {res['delta_at_best']:7.3f}")

    # quick per-cell HNO-vs-FNO verdict
    print("\n=== HNO vs FNO (elliptic should favor HNO) ===")
    by_cell = {}
    for pde, ic, op, sf, e, d in rows:
        by_cell.setdefault((pde, ic), {})[op] = e
    for (pde, ic), d in by_cell.items():
        if "fno" in d and "hno" in d:
            win = "HNO" if d["hno"] < d["fno"] else "FNO"
            print(f"  {pde:>10} {ic:>6}: FNO {d['fno']:.4f}  HNO {d['hno']:.4f}  -> {win}")
    return rows


def collect_scatter(results_dir=RESULTS_DIR):
    """Read every result JSON and return rows for the delta-vs-S_flat scatter."""
    import os, json
    rows = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fn)) as fh:
            r = json.load(fh)
        rows.append({
            "tag": fn[:-5],
            "dataset": r.get("dataset"),
            "operator": r["config"]["operator"],
            "S_flat": r.get("spectral_flatness_train"),
            "best_relL2": r.get("best_test_relL2"),
            "delta_at_best": r.get("delta_at_best"),
        })
    return rows


if __name__ == "__main__":
    run_elliptic_control(n_train=512, n_test=128, epochs=200)