"""
b_sanity.py -- why is B dead? A controlled diagonal-vs-nondiagonal test.

Translation-invariant PDEs (Poisson, heat, wave, biharmonic) have solution
operators that are DIAGONAL in frequency (a per-mode multiplier -- the
convolution theorem). A fixed spectral basis that is already aligned is then
optimal, and a learnable cross-mode correction has nothing to learn: B cannot
win, by construction.

A learnable basis can only help when the operator is NON-diagonal (couples
modes), i.e. NOT translation-invariant. This test builds both kinds of target
as exact linear maps in the Hartley domain and checks whether learnable-HNO
beats HNO/FNO only on the non-diagonal one.

  diagonal     : y = IDHT( D (.) DHT(x) )           (a convolution; per-mode)
  nondiagonal  : y = IDHT( M1 @ DHT(x)[:m,:m] @ M2 ) (separable cross-mode mix)

Prediction:  diagonal -> LHNO edge ~ 0 ;  nondiagonal -> LHNO edge << 0.
"""
import numpy as np
import torch
from spectral_operators import (set_seed, dht2, idht2, grf_2d, make_operator,
                                 TrainConfig, train_eval, DEVICE)


def make_task(kind, N=160, s=32, m=8, seed=0):
    set_seed(seed)
    x = grf_2d(N, s, device="cpu")              # (N,s,s) real, unit-std
    Hx = dht2(x)
    g = torch.Generator().manual_seed(123)
    Hy = torch.zeros_like(Hx)
    if kind == "diagonal":
        D = torch.zeros(s, s); D[:m, :m] = 0.5 + torch.rand(m, m, generator=g)
        Hy = Hx * D
    elif kind == "nondiagonal":
        M1 = torch.randn(m, m, generator=g) / np.sqrt(m)
        M2 = torch.randn(m, m, generator=g) / np.sqrt(m)
        blk = Hx[:, :m, :m]
        blk = torch.einsum("mn,bnj->bmj", M1, blk)
        blk = torch.einsum("bmj,jk->bmk", blk, M2)
        Hy[:, :m, :m] = blk
    else:
        raise ValueError(kind)
    y = idht2(Hy).real
    y = y / (y.abs().max() + 1e-8)
    return x.float(), y.float()


def run(kind, m=8, s=32, N=160, n_test=40, epochs=120, depth=1, width=24, seed=0):
    x, y = make_task(kind, N + n_test, s, m, seed=seed)
    xtr, ytr, xte, yte = x[:N], y[:N], x[N:], y[N:]
    out = {}
    for op in ("fno", "hno", "learnable_hno"):
        cfg = TrainConfig(operator=op, width=width, modes=m, nlayers=depth,
                          epochs=epochs, batch_size=16, lr=1e-3, weight_decay=1e-4,
                          grad_clip=1.0, align_lambda=0.0, scheduler="step",
                          early_stop=False, seed=seed)
        _, r = train_eval(xtr, ytr, xte, yte, cfg, dataset_name=kind, verbose=False)
        out[op] = (r["best_test_relL2"], r["delta_at_best"])
    fno, hno, lh = out["fno"][0], out["hno"][0], out["learnable_hno"][0]
    print(f"\n=== target: {kind} (m={m}, s={s}, depth={depth}) ===")
    print(f"  FNO  best relL2 = {fno:.4f}")
    print(f"  HNO  best relL2 = {hno:.4f}")
    print(f"  LHNO best relL2 = {lh:.4f}   (delta={out['learnable_hno'][1]:.3f})")
    print(f"  LHNO gain vs HNO       = {hno - lh:+.4f}")
    print(f"  LHNO edge vs best fixed = {lh - min(fno, hno):+.4f}  "
          f"({'BEATS both fixed' if lh < min(fno,hno) - 1e-3 else 'no win'})")
    return dict(kind=kind, fno=fno, hno=hno, lhno=lh,
                edge=lh - min(fno, hno), delta=out["learnable_hno"][1])


if __name__ == "__main__":
    print(f"device={DEVICE}")
    r_diag = run("diagonal")
    r_nond = run("nondiagonal")
    print("\n" + "=" * 60)
    print("VERDICT")
    print(f"  diagonal    LHNO edge = {r_diag['edge']:+.4f}  (expect ~0)")
    print(f"  nondiagonal LHNO edge = {r_nond['edge']:+.4f}  (expect << 0 if B is real)")
    if r_nond["edge"] < -0.02 and abs(r_diag["edge"]) < 0.02:
        print("  => learnable basis wins iff operator is NON-diagonal.")
        print("     B is 'dead' on our PDEs because they are translation-invariant (diagonal).")
    else:
        print("  => inconclusive / parameterization may be too weak; inspect above.")