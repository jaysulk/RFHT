"""
elliptic_pdes.py  (v3 -- faithful reproduction)
=====================================================================
Reproduces the Allerton/TMLR elliptic results using the *real* per-PDE
Optuna hyperparameters (extracted from {pde}_optuna_final_*.pkl) and the
verbatim data pipeline from the original notebooks:

  * IC generators (grf / eigenfunction / bump) copied exactly, incl. the
    [0,2pi] domain and randn/(k^2+1) amplitudes for the eigenfunction IC.
  * Spectral solvers: Poisson u_hat = f_hat / |k|^2, biharmonic / |k|^4.
  * GLOBAL max-abs normalization of stacked sources and solutions
    (NOT per-sample, NOT zero-meaned) -- matches gen_elliptic_data.
  * Training loss = plain MSE; Adam; StepLR(50, 0.5) or Cosine; grad-clip;
    batch_size 4; 200 epochs; per-sample relative-L2 test metric.

Models come from spectral_operators (layer/init/backbone already matched
to the validated repo code). LHNO == HNO at init plus a learnable basis.
=====================================================================
"""

import os, json, glob, pickle, random
import numpy as np
import torch
import torch.nn.functional as F

from spectral_operators import DEVICE, make_operator


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
#  Initial-condition generators  (verbatim from the original notebooks)
# --------------------------------------------------------------------------- #
def _grf_ic(Nx, Ny, nu=2.5, length_scale=0.15, sigma=1.0, device="cpu"):
    kx = torch.fft.fftfreq(Nx, d=1.0 / Nx, device=device)
    ky = torch.fft.fftfreq(Ny, d=1.0 / Ny, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    k_sq = KX ** 2 + KY ** 2
    tau = 2 * nu / (length_scale ** 2)
    spectrum = (tau + 4 * np.pi ** 2 * k_sq) ** (-(nu + 1))
    spectrum[0, 0] = 0
    sqrt_spec = torch.sqrt(spectrum) * sigma * np.sqrt(Nx * Ny)
    noise = torch.complex(torch.randn(Nx, Ny, device=device),
                          torch.randn(Nx, Ny, device=device))
    u = torch.fft.ifft2(sqrt_spec * noise).real
    return u / (u.abs().max() + 1e-8)


def _eigen_ic(Nx, Ny, n_modes=5, device="cpu"):
    x = torch.linspace(0, 2 * np.pi, Nx, device=device)
    y = torch.linspace(0, 2 * np.pi, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    ic = torch.zeros_like(X)
    for _ in range(n_modes):
        kx, ky = np.random.randint(1, 6), np.random.randint(1, 6)
        amp = np.random.randn() / (kx ** 2 + ky ** 2 + 1)
        ic += amp * torch.sin(kx * X + ky * Y + np.random.rand() * 2 * np.pi)
    return ic / (ic.abs().max() + 1e-8)


def _bump_ic(Nx, Ny, device="cpu"):
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u = torch.zeros(Nx, Ny, device=device)
    for _ in range(np.random.randint(2, 5)):
        cx, cy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
        sig = np.random.uniform(0.06, 0.15)
        u += np.random.uniform(0.5, 1.0) * torch.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))
    return u / (u.abs().max() + 1e-8)


IC_FAMILIES = {"grf": _grf_ic, "eigenfunction": _eigen_ic, "bump": _bump_ic}


# --------------------------------------------------------------------------- #
#  Spectral solvers  (verbatim)
# --------------------------------------------------------------------------- #
def solve_poisson(source):
    Nx, Ny = source.shape
    kx = torch.fft.fftfreq(Nx, d=1.0 / Nx, device=source.device) * 2 * np.pi
    ky = torch.fft.fftfreq(Ny, d=1.0 / Ny, device=source.device) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    Ks = KX ** 2 + KY ** 2
    Ks[0, 0] = 1
    uh = torch.fft.fft2(source) / Ks
    uh[0, 0] = 0
    return torch.fft.ifft2(uh).real


def solve_biharmonic(source):
    Nx, Ny = source.shape
    kx = torch.fft.fftfreq(Nx, d=1.0 / Nx, device=source.device) * 2 * np.pi
    ky = torch.fft.fftfreq(Ny, d=1.0 / Ny, device=source.device) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K4 = (KX ** 2 + KY ** 2) ** 2
    K4[0, 0] = 1
    uh = torch.fft.fft2(source) / K4
    uh[0, 0] = 0
    return torch.fft.ifft2(uh).real


SOLVERS = {"poisson": solve_poisson, "biharmonic": solve_biharmonic}


def make_elliptic(pde, ic, n, s=128, device=DEVICE, seed=0):
    """Generate n (source, solution) pairs with GLOBAL max-abs normalization."""
    set_seed(seed)
    gen, solver = IC_FAMILIES[ic], SOLVERS[pde]
    src, sol = [], []
    for _ in range(n):
        f = gen(s, s, device=device)
        src.append(f)
        sol.append(solver(f))
    sources = torch.stack(src)
    solutions = torch.stack(sol)
    sources = sources / (sources.abs().max() + 1e-8)        # global, matches repo
    solutions = solutions / (solutions.abs().max() + 1e-8)
    return sources.float(), solutions.float()


def _add_grid(f):
    """(N,H,W) -> (N,H,W,3) with channels [f, X, Y], X,Y on [0,1] (EllipticDataset)."""
    N, H, W = f.shape
    x = torch.linspace(0, 1, H, device=f.device)
    y = torch.linspace(0, 1, W, device=f.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X = X.unsqueeze(0).expand(N, -1, -1)
    Y = Y.unsqueeze(0).expand(N, -1, -1)
    return torch.stack([f, X, Y], dim=-1)


# --------------------------------------------------------------------------- #
#  Real per-PDE Optuna configs  (from *_optuna_final_*.pkl)
# --------------------------------------------------------------------------- #
OPTUNA_CONFIGS = {
    ("poisson", "fno"):    dict(modes=10, lr=2.1165e-3, weight_decay=1.016e-6, width=16, scheduler="step", grad_clip=5.0),
    ("poisson", "hno"):    dict(modes=16, lr=2.1329e-4, weight_decay=1.002e-6, width=32, scheduler="step", grad_clip=1.0),
    ("biharmonic", "fno"): dict(modes=20, lr=9.4450e-4, weight_decay=2.453e-6, width=32, scheduler="step", grad_clip=5.0),
    ("biharmonic", "hno"): dict(modes=14, lr=6.8941e-4, weight_decay=1.001e-6, width=48, scheduler="step", grad_clip=5.0),
}

# GRF-tuned achieved errors (the same HPs are applied to all ICs of a PDE)
PRIOR_TARGETS = {
    ("poisson", "grf"): 0.0041, ("poisson", "eigenfunction"): 0.0301, ("poisson", "bump"): 0.0023,
    ("biharmonic", "grf"): 0.0039, ("biharmonic", "eigenfunction"): 0.145, ("biharmonic", "bump"): 0.0032,
}


def load_optuna_configs(optuna_dir, verbose=True):
    """Override OPTUNA_CONFIGS from the MOST RECENT {pde}_optuna_final_*.pkl
    per PDE in optuna_dir. Falls back to the standard Drive mount path, and
    leaves the built-in defaults untouched for any PDE without a pkl."""
    keys = ["modes", "lr", "weight_decay", "width", "scheduler", "grad_clip"]
    cand_dirs = [optuna_dir,
                 optuna_dir.replace("/content/MyDrive", "/content/drive/MyDrive"),
                 "/content/drive/MyDrive/HNO_experiments",
                 "/content/MyDrive/HNO_experiments"]
    files, used_dir = [], None
    for dd in cand_dirs:
        files = glob.glob(os.path.join(dd, "*optuna_final*.pkl"))
        if files:
            used_dir = dd
            break
    if not files:
        if verbose:
            print(f"  [load_optuna_configs] no *_optuna_final_*.pkl found under "
                  f"{optuna_dir} -- using built-in defaults")
        return OPTUNA_CONFIGS

    # group by PDE prefix, pick the most recent (filename timestamp sorts last)
    by_pde = {}
    for p in files:
        pde = os.path.basename(p).split("_optuna")[0]
        by_pde.setdefault(pde, []).append(p)

    if verbose:
        print(f"  [load_optuna_configs] reading from {used_dir}")
    for pde, plist in sorted(by_pde.items()):
        p = sorted(plist)[-1]                       # most recent timestamp
        try:
            d = pickle.load(open(p, "rb"))
        except Exception as e:
            if verbose:
                print(f"    skip {os.path.basename(p)}: {e}")
            continue
        for op, k in [("fno", "fno_params"), ("hno", "hno_params")]:
            params = d.get(k) or d.get(f"{op}_best_params")
            if isinstance(params, dict):
                OPTUNA_CONFIGS[(pde, op)] = {kk: params.get(kk) for kk in keys}
        if verbose:
            print(f"    {pde:>11}: {os.path.basename(p)}")
            for op in ("fno", "hno"):
                c = OPTUNA_CONFIGS.get((pde, op))
                if c:
                    print(f"        {op.upper()}: modes={c['modes']} lr={c['lr']:.2e} "
                          f"width={c['width']} {c['scheduler']} clip={c['grad_clip']}")
    return OPTUNA_CONFIGS


# --------------------------------------------------------------------------- #
#  Training (replicates the repo train_model: MSE loss, Adam, step/cosine)
# --------------------------------------------------------------------------- #
def _rel_l2(pred, target):
    errs = [(torch.norm(pred[i] - target[i]) / torch.norm(target[i])).item()
            for i in range(pred.shape[0])]
    return float(np.mean(errs))


def train_repro(op, x_train, y_train, x_test, y_test, cfg,
                epochs=200, batch_size=4, align_lambda=1e-3, seed=42, verbose=False):
    set_seed(seed)
    model = make_operator(op, 3, cfg["width"], 4, cfg["modes"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    if cfg["scheduler"] == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xtr = _add_grid(x_train.to(DEVICE)); ytr = y_train.to(DEVICE)
    xte = _add_grid(x_test.to(DEVICE)); yte = y_test.to(DEVICE)
    ntr = xtr.shape[0]
    learnable = op in ("learnable_hno", "lhno", "ours")

    best, best_ep, final = float("inf"), 0, float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ntr, device=DEVICE)
        for i in range(0, ntr, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(xtr[idx])                       # (B,H,W)
            loss = F.mse_loss(pred, ytr[idx])
            if learnable and align_lambda > 0:
                pen = sum(m.alignment_penalty() for m in model.spectral)
                loss = loss + align_lambda * pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            preds = [model(xte[i:i + batch_size]) for i in range(0, xte.shape[0], batch_size)]
            rl2 = _rel_l2(torch.cat(preds, 0), yte)
        final = rl2
        if rl2 < best:
            best, best_ep = rl2, ep
        if verbose and ep % 50 == 0:
            print(f"        ep{ep:3d}: relL2={rl2:.4f}  (best {best:.4f})")

    delta = 0.0
    if learnable:
        with torch.no_grad():
            devs = [m.basis_deviation() for m in model.spectral
                    if hasattr(m, "basis_deviation")]
            delta = float(np.mean(devs)) if devs else 0.0
    return {"best_test_relL2": best, "final_test_relL2": final,
            "best_epoch": best_ep, "delta": delta}, model


# --------------------------------------------------------------------------- #
#  Reproduction driver
# --------------------------------------------------------------------------- #
def run_elliptic_reproduction(optuna_dir="/content/MyDrive/HNO_experiments",
                              pdes=("poisson", "biharmonic"),
                              ics=("grf",),
                              ops=("fno", "hno", "learnable_hno"),
                              n_train=200, n_test=40, epochs=200,
                              batch_size=4, seed=42, align_lambda=1e-3,
                              save_dir="./results", verbose=False):
    if optuna_dir:
        load_optuna_configs(optuna_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"device={DEVICE}; elliptic REPRODUCTION (REAL Optuna HPs, s=128, "
          f"N={n_train}+{n_test}, {epochs} ep, seed {seed})")
    print(f"{'pde':>10} {'ic':>14} {'op':>14} {'relL2':>9} {'prior':>9} {'delta':>7}")
    print("-" * 70)
    rows = []
    for pde in pdes:
        for ic in ics:
            f, u = make_elliptic(pde, ic, n_train + n_test, 128, seed=seed)
            xtr, ytr = f[:n_train], u[:n_train]
            xte, yte = f[n_train:], u[n_train:]
            for op in ops:
                base = "hno" if op in ("learnable_hno", "lhno", "ours") else op
                cfg = OPTUNA_CONFIGS.get((pde, base))
                if cfg is None:
                    print(f"  (no config for {pde}/{base}; skipping)")
                    continue
                res, _ = train_repro(op, xtr, ytr, xte, yte, cfg,
                                     epochs=epochs, batch_size=batch_size,
                                     align_lambda=align_lambda, seed=seed,
                                     verbose=verbose)
                prior = PRIOR_TARGETS.get((pde, ic), float("nan"))
                print(f"{pde:>10} {ic:>14} {op:>14} "
                      f"{res['best_test_relL2']:>9.4f} {prior:>9.4f} {res['delta']:>7.3f}")
                rec = dict(pde=pde, ic=ic, op=op, modes=cfg["modes"], lr=cfg["lr"],
                           **res, S_flat=None)
                rows.append(rec)
                with open(os.path.join(save_dir, f"repro_{pde}_{ic}_{op}.json"), "w") as fh:
                    json.dump(rec, fh, indent=2)
    return rows


def collect_scatter(save_dir="./results"):
    """Pull (S_flat, relL2, delta) rows from saved result JSONs."""
    rows = []
    for p in sorted(glob.glob(os.path.join(save_dir, "repro_*.json"))):
        rec = json.load(open(p))
        rows.append((rec.get("S_flat"), rec.get("best_test_relL2"),
                     rec.get("delta"), rec.get("op"), rec.get("pde"), rec.get("ic")))
    return rows


if __name__ == "__main__":
    run_elliptic_reproduction()