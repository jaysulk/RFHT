"""
darcy.py -- variable-coefficient (Darcy) flow generator.
=====================================================================
Solves the canonical non-translation-invariant elliptic benchmark

      -div( a(x) grad u(x) ) = f ,   u = 0 on the boundary,

and returns the operator-learning pairs  (a, u).  Unlike Poisson/heat/
wave, the solution operator a -> u is NOT diagonal in any fixed Fourier-
like basis (a(x) varies in space), so a learnable / Monarch transform
has a genuine representational floor to close. This is exactly the
regime where hypothesis B can be alive (see b_sanity.py).

Coefficient field a(x):
  * "threshold" (Li et al. default): a = a_hi where mu>=0 else a_lo,
    mu ~ GRF with covariance (-Lap + tau^2)^(-alpha).
  * "lognormal": a = exp(mu)  (smooth, still non-diagonal).

Returns float tensors (N, s, s): inputs a (normalized to ~[0,1]) and
solutions u (global max-abs normalized), matching the make_elliptic API
so it drops into the same harness.
=====================================================================
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

try:
    from spectral_operators import set_seed, DEVICE
except Exception:                       # standalone fallback
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def set_seed(s):
        np.random.seed(s); torch.manual_seed(s)


# --------------------------------------------------------------------------- #
#  Random coefficient field
# --------------------------------------------------------------------------- #
def _grf(s, alpha=2.0, tau=3.0):
    """Gaussian random field, covariance (-Lap + tau^2)^(-alpha) on the torus."""
    k = np.fft.fftfreq(s, d=1.0 / s)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    coef = (4 * np.pi ** 2 * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2.0)
    xi = np.random.randn(s, s) + 1j * np.random.randn(s, s)
    field = np.fft.ifft2(coef * xi).real
    return (field - field.mean()) / (field.std() + 1e-8)


def coefficient_field(s, kind="threshold", a_lo=3.0, a_hi=12.0):
    mu = _grf(s)
    if kind == "threshold":
        return np.where(mu >= 0.0, a_hi, a_lo)
    elif kind == "lognormal":
        return np.exp(0.5 * mu) * (a_hi - a_lo) / 6.0 + a_lo
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
#  Finite-difference Darcy solver  (5-point, harmonic face conductivities)
# --------------------------------------------------------------------------- #
def _solve_darcy(a, f_val=1.0):
    s = a.shape[0]
    h = 1.0 / (s - 1)
    N = s * s

    def harm(x, y):
        return 2.0 * x * y / (x + y + 1e-12)

    fx = harm(a[:-1, :], a[1:, :])      # x-faces between (i,j)-(i+1,j): shape (s-1, s)
    fy = harm(a[:, :-1], a[:, 1:])      # y-faces between (i,j)-(i,j+1): shape (s, s-1)

    ii, jj = np.meshgrid(np.arange(1, s - 1), np.arange(1, s - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    p = ii * s + jj
    fe, fw = fx[ii, jj], fx[ii - 1, jj]
    fn, fsr = fy[ii, jj], fy[ii, jj - 1]
    diag = (fe + fw + fn + fsr) / h ** 2

    rows = np.concatenate([p, p, p, p, p])
    cols = np.concatenate([p, p + s, p - s, p + 1, p - 1])
    data = np.concatenate([diag, -fe / h ** 2, -fw / h ** 2, -fn / h ** 2, -fsr / h ** 2])

    # boundary nodes -> identity rows (Dirichlet u = 0)
    bmask = np.ones((s, s), bool); bmask[1:-1, 1:-1] = False
    bidx = np.where(bmask.ravel())[0]
    rows = np.concatenate([rows, bidx]); cols = np.concatenate([cols, bidx])
    data = np.concatenate([data, np.ones_like(bidx, float)])

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    rhs = np.full(N, f_val); rhs[bidx] = 0.0
    u = spsolve(A, rhs).reshape(s, s)
    return u


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
def make_darcy(pde_or_n, ic_or_s=None, n=None, s=128, device=DEVICE, seed=0,
               kind="threshold"):
    """Flexible signature so it can be called either as
         make_darcy(n, s, seed=...)                      (standalone), or
         make_darcy('darcy', kind, n, s, seed=...)       (harness-style).
    Returns (a, u) float tensors of shape (N, s, s)."""
    if isinstance(pde_or_n, str):                 # harness-style: ('darcy', kind, n, s)
        kind = ic_or_s or kind
        N = n
    else:                                         # standalone: (n, s)
        N = pde_or_n
        if ic_or_s is not None:
            s = ic_or_s
    set_seed(seed)
    A_list, U_list = [], []
    for _ in range(N):
        a = coefficient_field(s, kind=kind)
        u = _solve_darcy(a)
        A_list.append(a); U_list.append(u)
    a = torch.from_numpy(np.stack(A_list)).float()
    u = torch.from_numpy(np.stack(U_list)).float()
    a = a / (a.abs().max() + 1e-8)
    u = u / (u.abs().max() + 1e-8)
    return a, u


if __name__ == "__main__":
    a, u = make_darcy(8, 48, seed=0)
    print("make_darcy ok:", a.shape, u.shape,
          "| a range", (float(a.min()), float(a.max())),
          "| u range", (float(u.min()), float(u.max())))