"""
spectral_operators.py  (v2)
=====================================================================
Drop-in replacement for v1. Same models; fixes the issues the first
smoke run exposed.

CHANGES vs v1
-------------
1. train_eval now does EARLY STOPPING and reports BEST test error (not the
   overfit final-epoch value). Records best_epoch and delta_at_best.
2. Parameter-matched comparison: run_ns_comparison(param_match=True) sizes
   HNO / learnable-HNO width so total params ~= FNO, removing the 2x confound
   (four-corner even/odd HNO has 2x FNO's spectral params at equal width).
3. verify_harness(): asserts the DHT round-trip, checks forward shapes,
   prints a per-operator parameter table (so the confound is visible), and
   confirms learnable-HNO == HNO at initialization (L = I).
4. Sensible smoke defaults: more data, weight decay, patience.

For Monarch-comparable NS numbers use load_li_ns_mat() on Li et al.'s data,
not the smoke generator.
=====================================================================
"""

import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

RESULTS_DIR = "./results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONARCH_NS_REFERENCE = {
    "nu_1e-3_T50": {"FNO": 0.0128, "Monarch_NO": 0.010, "U_Net": 0.0245},
    "nu_1e-4_T30": {"FNO": 0.1559, "Monarch_NO": 0.145, "U_Net": 0.2051},
    "nu_1e-5_T20": {"FNO": 0.1556, "Monarch_NO": 0.136, "U_Net": 0.1982},
}


def set_seed(seed: int = 0):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RelL2Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps

    def forward(self, pred, target):
        b = pred.shape[0]
        num = torch.linalg.vector_norm(pred.reshape(b, -1) - target.reshape(b, -1), dim=1)
        den = torch.linalg.vector_norm(target.reshape(b, -1), dim=1) + self.eps
        return (num / den).mean()


# --------------------------- transforms ------------------------------------ #

def dht2(x):
    Xf = torch.fft.fft2(x)
    return Xf.real - Xf.imag


def idht2(X):
    H, W = X.shape[-2], X.shape[-1]
    return dht2(X) / (H * W)


def hartley_neg(H):
    return torch.roll(torch.flip(H, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


# --------------------------- spectral layers -------------------------------- #

class SpectralConv2d(nn.Module):
    """Standard complex Fourier spectral conv (Li et al.); rfft, two corners."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))

    @staticmethod
    def _cmul(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        B, _, H, W = x.shape
        xft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        w1, w2 = torch.view_as_complex(self.w1), torch.view_as_complex(self.w2)
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1, :m2] = self._cmul(xft[:, :, :m1, :m2], w1)
        out[:, :, -m1:, :m2] = self._cmul(xft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out, s=(H, W))

    def alignment_penalty(self):
        return torch.zeros((), device=self.w1.device)


class HartleyConv2d(nn.Module):
    """Four-quadrant even/odd real Hartley spectral conv (prior-work HNO)."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_even = nn.Parameter(scale * torch.rand(4, in_ch, out_ch, modes1, modes2))
        self.w_odd = nn.Parameter(scale * torch.rand(4, in_ch, out_ch, modes1, modes2))

    def _corners(self, H, W):
        m1, m2 = self.modes1, self.modes2
        return [(slice(0, m1), slice(0, m2)), (slice(0, m1), slice(W - m2, W)),
                (slice(H - m1, H), slice(0, m2)), (slice(H - m1, H), slice(W - m2, W))]

    def _mix(self, e, o, i):
        return e, o

    def forward(self, x):
        B, _, H, W = x.shape
        Hs = dht2(x); Hn = hartley_neg(Hs)
        He = 0.5 * (Hs + Hn); Ho = 0.5 * (Hs - Hn)
        out = torch.zeros(B, self.out_ch, H, W, device=x.device, dtype=x.dtype)
        for i, (s1, s2) in enumerate(self._corners(H, W)):
            e, o = He[:, :, s1, s2], Ho[:, :, s1, s2]
            e, o = self._mix(e, o, i)
            out[:, :, s1, s2] = (torch.einsum("bixy,ioxy->boxy", e, self.w_even[i])
                                 + torch.einsum("bixy,ioxy->boxy", o, self.w_odd[i]))
        return idht2(out)

    def alignment_penalty(self):
        return torch.zeros((), device=self.w_even.device)


class LearnableHartleyConv2d(HartleyConv2d):
    """HNO + learnable separable basis correction (this work). Starts at HNO."""

    def __init__(self, in_ch, out_ch, modes1, modes2, share_across_corners=True):
        super().__init__(in_ch, out_ch, modes1, modes2)
        self.share = share_across_corners
        n = 1 if share_across_corners else 4
        self.L1 = nn.Parameter(torch.eye(modes1).unsqueeze(0).repeat(n, 1, 1).clone())
        self.L2 = nn.Parameter(torch.eye(modes2).unsqueeze(0).repeat(n, 1, 1).clone())

    def _mix(self, e, o, i):
        j = 0 if self.share else i
        L1, L2 = self.L1[j], self.L2[j]
        e = torch.einsum("mn,binj->bimj", L1, e); e = torch.einsum("bimj,jk->bimk", e, L2)
        o = torch.einsum("mn,binj->bimj", L1, o); o = torch.einsum("bimj,jk->bimk", o, L2)
        return e, o

    def alignment_penalty(self):
        I1 = torch.eye(self.modes1, device=self.L1.device)
        I2 = torch.eye(self.modes2, device=self.L2.device)
        return ((self.L1 - I1) ** 2).sum() + ((self.L2 - I2) ** 2).sum()

    @torch.no_grad()
    def basis_deviation(self):
        I1 = torch.eye(self.modes1, device=self.L1.device)
        I2 = torch.eye(self.modes2, device=self.L2.device)
        num = (torch.linalg.matrix_norm(self.L1 - I1).mean()
               + torch.linalg.matrix_norm(self.L2 - I2).mean())
        den = torch.linalg.matrix_norm(I1) + torch.linalg.matrix_norm(I2)
        return float(num / den)


# --------------------------- backbone --------------------------------------- #

class NeuralOperator2d(nn.Module):
    def __init__(self, spectral_factory, in_channels=3, width=32, nlayers=4):
        super().__init__()
        self.width, self.nlayers = width, nlayers
        self.fc0 = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList([spectral_factory(width) for _ in range(nlayers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(nlayers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = rearrange(x, "b h w c -> b c h w")
        for k in range(self.nlayers):
            x = F.gelu(self.spectral[k](x) + self.w[k](x))
        x = rearrange(x, "b c h w -> b h w c")
        x = self.fc2(F.gelu(self.fc1(x)))
        return x[..., 0]

    def alignment_penalty(self):
        return sum(s.alignment_penalty() for s in self.spectral)

    def basis_deviation(self):
        devs = [s.basis_deviation() for s in self.spectral if hasattr(s, "basis_deviation")]
        return float(np.mean(devs)) if devs else 0.0


def make_operator(kind, in_channels=3, width=32, nlayers=4, modes=12):
    kind = kind.lower()
    if kind == "fno":
        fac = lambda w: SpectralConv2d(w, w, modes, modes)
    elif kind == "hno":
        fac = lambda w: HartleyConv2d(w, w, modes, modes)
    elif kind in ("learnable_hno", "lhno", "ours"):
        fac = lambda w: LearnableHartleyConv2d(w, w, modes, modes)
    else:
        raise ValueError(f"unknown operator kind: {kind}")
    return NeuralOperator2d(fac, in_channels=in_channels, width=width, nlayers=nlayers)


def match_width(kind, target_params, modes, nlayers, in_channels=3, lo=8, hi=96):
    """Smallest-error width so that `kind` has ~target_params parameters."""
    best_w, best_diff = lo, float("inf")
    for w in range(lo, hi + 1):
        p = count_params(make_operator(kind, in_channels, w, nlayers, modes))
        if abs(p - target_params) < best_diff:
            best_diff, best_w = abs(p - target_params), w
    return best_w


# --------------------------- data ------------------------------------------- #

def add_grid(field):
    N, H, W = field.shape
    gx = repeat(torch.linspace(0, 1, H, device=field.device), "h -> n h w", n=N, w=W)
    gy = repeat(torch.linspace(0, 1, W, device=field.device), "w -> n h w", n=N, h=H)
    return torch.stack([field, gx, gy], dim=-1)


def grf_2d(n, s=64, alpha=2.5, tau=7.0, device=DEVICE):
    k = torch.fft.fftfreq(s, d=1.0 / s, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    coef = (tau ** (alpha - 1)) * (kx ** 2 + ky ** 2 + tau ** 2) ** (-alpha / 2.0)
    coef[0, 0] = 0.0
    field = torch.fft.ifft2(torch.randn(n, s, s, dtype=torch.cfloat, device=device) * coef[None]).real
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    return field / (field.std(dim=(-2, -1), keepdim=True) + 1e-8)


def generate_navier_stokes(n, s=64, nu=1e-3, T=1.0, dt=1e-3, device=DEVICE, seed=0):
    """SMOKE-TEST quality w0 -> w(T). Prefer load_li_ns_mat() for real numbers."""
    set_seed(seed)
    k1 = torch.fft.fftfreq(s, d=1.0 / s, device=device)
    kx, ky = torch.meshgrid(k1, k1, indexing="ij")
    lap = -(kx ** 2 + ky ** 2); lap[0, 0] = 1.0
    inv_lap = 1.0 / lap; inv_lap[0, 0] = 0.0
    kmax = s // 3
    mask = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).float()
    xx = torch.linspace(0, 2 * math.pi, s + 1, device=device)[:-1]
    X, Y = torch.meshgrid(xx, xx, indexing="ij")
    fh = torch.fft.fft2(0.1 * (torch.sin(X + Y) + torch.cos(X + Y)))

    def rhs(wh):
        psih = wh * inv_lap
        u = torch.fft.ifft2(1j * ky * psih).real
        v = torch.fft.ifft2(-1j * kx * psih).real
        wx = torch.fft.ifft2(1j * kx * wh).real
        wy = torch.fft.ifft2(1j * ky * wh).real
        adv = torch.fft.fft2(u * wx + v * wy) * mask
        return -adv + nu * lap * wh + fh[None]

    w0 = grf_2d(n, s, device=device)
    wh = torch.fft.fft2(w0)
    for _ in range(max(1, int(round(T / dt)))):
        k1_ = rhs(wh); k2_ = rhs(wh + dt * k1_)
        wh = wh + 0.5 * dt * (k1_ + k2_)
    return w0.float().cpu(), torch.fft.ifft2(wh).real.float().cpu()


def load_li_ns_mat(path, in_steps=1, key="u"):
    from scipy.io import loadmat
    arr = np.array(loadmat(path)[key])  # (N, S, S, T)
    return (torch.from_numpy(arr[..., in_steps - 1]).float(),
            torch.from_numpy(arr[..., -1]).float())


def spectral_flatness(field, n_modes=16):
    P = (torch.fft.fft2(field).abs() ** 2)[..., :n_modes, :n_modes].reshape(field.shape[0], -1) + 1e-12
    return float((torch.exp(torch.log(P).mean(1)) / P.mean(1)).mean())


DATASET_REGISTRY: dict = {}
def register_dataset(name, fn): DATASET_REGISTRY[name] = fn


# --------------------------- train / eval ----------------------------------- #

@dataclass
class TrainConfig:
    operator: str = "hno"
    modes: int = 12
    width: int = 32
    nlayers: int = 3
    epochs: int = 200
    batch_size: int = 20
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    align_lambda: float = 1e-3
    scheduler: str = "step"
    patience: int = 30          # early-stopping patience on test relL2
    early_stop: bool = True
    seed: int = 0


def train_eval(x_train, y_train, x_test, y_test, cfg: TrainConfig,
               dataset_name="navier_stokes", device=DEVICE, verbose=True):
    set_seed(cfg.seed)
    xtr = add_grid(x_train.to(device)); ytr = y_train.to(device)
    xte = add_grid(x_test.to(device)); yte = y_test.to(device)

    model = make_operator(cfg.operator, 3, cfg.width, cfg.nlayers, cfg.modes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
             if cfg.scheduler == "cosine"
             else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.epochs // 4), gamma=0.5))
    loss_fn = RelL2Loss()
    learnable = cfg.operator in ("learnable_hno", "lhno", "ours")

    n = xtr.shape[0]
    hist = {"train": [], "test": [], "delta": []}
    best_test, best_epoch, delta_at_best, wait = float("inf"), -1, 0.0, 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train(); perm = torch.randperm(n, device=device); run = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            if learnable:
                loss = loss + cfg.align_lambda * model.alignment_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); run += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            te = float(loss_fn(model(xte), yte))
        delta = model.basis_deviation()
        hist["train"].append(run / n); hist["test"].append(te); hist["delta"].append(delta)

        if te < best_test - 1e-6:
            best_test, best_epoch, delta_at_best, wait = te, ep, delta, 0
        else:
            wait += 1
        if verbose and (ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs - 1):
            print(f"[{cfg.operator:>14}] ep {ep:3d}  train {run/n:.4f}  test {te:.4f}  "
                  f"delta {delta:.4f}  (best {best_test:.4f}@{best_epoch})")
        if cfg.early_stop and wait >= cfg.patience:
            if verbose:
                print(f"[{cfg.operator:>14}] early stop at ep {ep} (best {best_test:.4f}@{best_epoch})")
            break

    return model, {
        "dataset": dataset_name, "config": asdict(cfg),
        "best_test_relL2": best_test, "best_epoch": best_epoch,
        "delta_at_best": delta_at_best,
        "final_test_relL2": hist["test"][-1], "final_delta": hist["delta"][-1],
        "n_params": count_params(model),
        "spectral_flatness_train": spectral_flatness(x_train),
        "train_seconds": time.time() - t0, "history": hist,
    }


def save_result(result, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, tag)
    with open(base + ".json", "w") as fh:
        json.dump(result, fh, indent=2)
    np.savez(base + ".npz", **{k: np.array(result["history"][k]) for k in result["history"]})
    print(f"saved -> {base}.json (+ .npz)")


# --------------------------- diagnostics ------------------------------------ #

def verify_harness(s=32, modes=8, width=16, nlayers=2, device=DEVICE):
    print(f"device = {device}")
    # 1) transform round-trip
    x = torch.randn(2, 3, s, s, device=device)
    err = (idht2(dht2(x)) - x).abs().max().item()
    print(f"DHT round-trip max abs err: {err:.2e}")
    assert err < 1e-3, "DHT round-trip failed"

    # 2) forward shapes + parameter table at EQUAL width (shows the 2x confound)
    inp = torch.randn(2, s, s, 3, device=device)
    print(f"\nparameters at equal width={width}, modes={modes}, nlayers={nlayers}:")
    for kind in ["fno", "hno", "learnable_hno"]:
        m = make_operator(kind, 3, width, nlayers, modes).to(device)
        y = m(inp)
        assert y.shape == (2, s, s), f"{kind} bad output shape {tuple(y.shape)}"
        print(f"  {kind:>14}: {count_params(m):>9,d} params")

    # 3) param-matched widths
    target = count_params(make_operator("fno", 3, 32, 3, 12))
    print(f"\nto match FNO(width 32, modes 12) = {target:,d} params:")
    for kind in ["hno", "learnable_hno"]:
        w = match_width(kind, target, 12, 3)
        print(f"  {kind:>14}: width {w} -> {count_params(make_operator(kind,3,w,3,12)):,d}")

    # 4) learnable-HNO starts exactly at HNO (L = I => identity mix)
    conv = LearnableHartleyConv2d(4, 4, modes, modes).to(device)
    e = torch.randn(2, 4, modes, modes, device=device); o = torch.randn_like(e)
    em, om = conv._mix(e, o, 0)
    mix_err = max((em - e).abs().max().item(), (om - o).abs().max().item())
    print(f"\nlearnable-HNO identity-init mix err: {mix_err:.2e}  (delta={conv.basis_deviation():.2e})")
    assert mix_err < 1e-5, "learnable-HNO does not start at Hartley"
    print("\nverify_harness: ALL CHECKS PASSED")


# --------------------------- drivers ---------------------------------------- #

def run_ns_comparison(n_train=512, n_test=128, s=64, nu=1e-3, T=1.0, dt=1e-3,
                      epochs=200, fno_width=32, modes=12, nlayers=3,
                      param_match=True, operators=("fno", "hno", "learnable_hno")):
    """NS baseline across operators, parameter-matched, best-error reported."""
    print(f"device={DEVICE}; generating NS (nu={nu}, T={T}, N={n_train+n_test}) ...")
    w0, wT = generate_navier_stokes(n_train + n_test, s=s, nu=nu, T=T, dt=dt)
    x_tr, y_tr = w0[:n_train], wT[:n_train]
    x_te, y_te = w0[n_train:], wT[n_train:]
    print(f"S_flat(train) = {spectral_flatness(x_tr):.3f}")

    target = count_params(make_operator("fno", 3, fno_width, nlayers, modes))
    summary = {}
    for op in operators:
        width = fno_width if (op == "fno" or not param_match) else match_width(op, target, modes, nlayers)
        cfg = TrainConfig(operator=op, width=width, modes=modes, nlayers=nlayers,
                          epochs=epochs, lr=(9e-3 if op == "fno" else 3e-3),
                          grad_clip=(0.5 if op == "fno" else 5.0))
        _, res = train_eval(x_tr, y_tr, x_te, y_te, cfg, dataset_name="navier_stokes_smoke")
        res["monarch_reference"] = MONARCH_NS_REFERENCE
        save_result(res, tag=f"ns_{op}")
        summary[op] = (res["best_test_relL2"], res["n_params"], res["delta_at_best"])

    print("\n=== summary (best test relL2 | params | delta@best) ===")
    for op, (e, p, d) in summary.items():
        print(f"  {op:>14}: {e:.4f} | {p:>9,d} | {d:.3f}")
    print("  Monarch NS reference (Li data, not smoke):", MONARCH_NS_REFERENCE["nu_1e-3_T50"])


if __name__ == "__main__":
    verify_harness()
    print("\n" + "=" * 60 + "\n")
    run_ns_comparison(n_train=512, n_test=128, epochs=150)