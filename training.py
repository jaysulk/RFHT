"""
Training utilities for RHNO experiments.
Matches paper setup exactly for fair comparison with HNO results.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import math


# ---------------------------------------------------------------------------
# Loss functions (paper Eq. 10-11)
# ---------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Relative L2 error (paper Eq. 10)."""
    return (torch.norm(pred - true, dim=(-1, -2)) /
            torch.norm(true, dim=(-1, -2))).mean()


def gradient_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Gradient error via central finite differences (paper Eq. 11)."""
    def grad(u):
        # Central differences, periodic
        gx = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / 2
        gy = (torch.roll(u, -1, -2) - torch.roll(u, 1, -2)) / 2
        return torch.stack([gx, gy], dim=-1)
    
    gp = grad(pred)
    gt = grad(true)
    return (torch.norm(gp - gt, dim=(-1, -2, -3)) /
            torch.norm(gt, dim=(-1, -2, -3))).mean()


# ---------------------------------------------------------------------------
# Initial condition generators (paper Section 3.1)
# ---------------------------------------------------------------------------

def make_grf(n_samples: int, resolution: int, nu: float = 2.5,
             length_scale: float = 0.15, device='cpu') -> torch.Tensor:
    """
    Gaussian Random Field with Matérn covariance (paper Eq. 4).
    Spectral method: sample in Fourier space, apply sqrt(PSD), IFFT.
    """
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Matérn spectral density: C_nu(k) ~ (2nu/l^2 + 4pi^2|k|^2)^{-(nu + d/2)}
    d = 2  # spatial dimension
    matern_psd = (2*nu / length_scale**2 + 4*math.pi**2 * K2) ** (-(nu + d/2))
    matern_psd[0, 0] = 0.0  # zero mean
    
    # Sample complex Gaussian noise
    noise_real = torch.randn(n_samples, resolution, resolution, device=device)
    noise_imag = torch.randn(n_samples, resolution, resolution, device=device)
    noise = torch.complex(noise_real, noise_imag)
    
    # Scale by sqrt(PSD) and IFFT
    scaled = noise * torch.sqrt(matern_psd).unsqueeze(0)
    grf = torch.fft.ifft2(scaled).real
    
    # Normalize
    grf = grf / (grf.abs().amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return grf


def make_eigenfunction_ic(n_samples: int, resolution: int, s: float = 2.0,
                           K_max: int = 8, device='cpu') -> torch.Tensor:
    """Eigenfunction ICs (paper Eq. 5): superposition of Fourier modes."""
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    u0 = torch.zeros(n_samples, resolution, resolution, device=device)
    
    k_vals = torch.arange(1, K_max+1, device=device, dtype=torch.float32)
    for i in range(n_samples):
        for k in range(1, K_max+1):
            for l in range(1, K_max+1):
                coeff = torch.randn(1, device=device).item()
                weight = coeff / (1 + k**2 + l**2)**s
                u0[i] += weight * torch.sin(math.pi*k*X) * torch.sin(math.pi*l*Y)
    
    u0 = u0 / (u0.abs().amax(dim=(-1,-2), keepdim=True) + 1e-8)
    return u0


def make_gaussian_bump_ic(n_samples: int, resolution: int,
                           n_bumps_range=(2, 5), device='cpu') -> torch.Tensor:
    """Gaussian bump ICs (paper Eq. 6)."""
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    u0 = torch.zeros(n_samples, resolution, resolution, device=device)
    
    for i in range(n_samples):
        n_bumps = np.random.randint(n_bumps_range[0], n_bumps_range[1]+1)
        for _ in range(n_bumps):
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.06, 0.15)
            a = np.random.uniform(-1, 1)
            u0[i] += a * torch.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))
    
    u0 = u0 / (u0.abs().amax(dim=(-1,-2), keepdim=True) + 1e-8)
    return u0


# ---------------------------------------------------------------------------
# PDE solvers (matching paper Section 3.2)
# ---------------------------------------------------------------------------

def solve_heat_equation(u0: torch.Tensor, nu: float = 0.01,
                         T: float = 0.5, Nt: int = 51) -> torch.Tensor:
    """
    Exact Fourier-space solution: u_hat(k,t) = u_hat0(k) * exp(-nu|k|^2*t)
    Paper: evaluated at Nt=51 uniformly spaced times in [0, T].
    Returns: (batch, Nt, H, W)
    """
    B, H, W = u0.shape
    device = u0.device
    
    kx = torch.fft.fftfreq(H, d=1.0/H).to(device)
    ky = torch.fft.fftfreq(W, d=1.0/W).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    u0_hat = torch.fft.fft2(u0)  # (B, H, W)
    
    t_vals = torch.linspace(0, T, Nt, device=device)
    solutions = []
    
    for t in t_vals:
        decay = torch.exp(-nu * K2 * t)
        u_hat_t = u0_hat * decay.unsqueeze(0)
        u_t = torch.fft.ifft2(u_hat_t).real
        solutions.append(u_t)
    
    return torch.stack(solutions, dim=1)  # (B, Nt, H, W)


def solve_poisson(f: torch.Tensor) -> torch.Tensor:
    """
    Exact spectral Poisson solve: u_hat(k) = f_hat(k) / (4pi^2|k|^2)
    Paper: zero mean enforced.
    """
    B, H, W = f.shape
    device = f.device
    
    kx = torch.fft.fftfreq(H, d=1.0/H).to(device)
    ky = torch.fft.fftfreq(W, d=1.0/W).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = (4 * math.pi**2 * (KX**2 + KY**2))
    K2[0, 0] = 1.0  # avoid division by zero; zero mode set to 0 below
    
    f_hat = torch.fft.fft2(f)
    u_hat = f_hat / K2.unsqueeze(0)
    u_hat[:, 0, 0] = 0.0  # zero mean
    
    return torch.fft.ifft2(u_hat).real


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class HeatDataset(Dataset):
    """Heat equation dataset. Input: [u0, x, y, t]. Output: u(x,y,t)."""
    
    def __init__(self, n_samples: int = 200, resolution: int = 128,
                 nu: float = 0.01, T: float = 0.5, Nt: int = 51,
                 ic_type: str = 'grf', device: str = 'cpu'):
        super().__init__()
        self.resolution = resolution
        self.Nt = Nt
        
        print(f"Generating heat dataset: {n_samples} samples, nu={nu}, IC={ic_type}")
        
        # Generate ICs
        if ic_type == 'grf':
            u0 = make_grf(n_samples, resolution, device=device)
        elif ic_type == 'eigenfunction':
            u0 = make_eigenfunction_ic(n_samples, resolution, device=device)
        else:
            u0 = make_gaussian_bump_ic(n_samples, resolution, device=device)
        
        # Solve
        solutions = solve_heat_equation(u0, nu=nu, T=T, Nt=Nt)
        # solutions: (B, Nt, H, W)
        
        # Build coordinate grids
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        t_vals = torch.linspace(0, T, Nt, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Store: inputs (u0 repeated + coords), outputs
        self.inputs = []
        self.targets = []
        
        for i in range(n_samples):
            for t_idx in range(Nt):
                t_val = t_vals[t_idx]
                T_grid = torch.full_like(X, t_val.item())
                
                # Stack: (H, W, 4)
                inp = torch.stack([u0[i], X, Y, T_grid], dim=-1)
                tgt = solutions[i, t_idx]
                
                self.inputs.append(inp)
                self.targets.append(tgt)
        
        self.inputs = torch.stack(self.inputs)   # (N*Nt, H, W, 4)
        self.targets = torch.stack(self.targets)  # (N*Nt, H, W)
        
        print(f"  Dataset size: {len(self.inputs)}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx].unsqueeze(-1)


class PoissonDataset(Dataset):
    """Poisson equation dataset. Input: [f, x, y]. Output: u."""
    
    def __init__(self, n_samples: int = 200, resolution: int = 128,
                 ic_type: str = 'grf', device: str = 'cpu'):
        super().__init__()
        print(f"Generating Poisson dataset: {n_samples} samples, IC={ic_type}")
        
        if ic_type == 'grf':
            f = make_grf(n_samples, resolution, device=device)
        elif ic_type == 'eigenfunction':
            f = make_eigenfunction_ic(n_samples, resolution, device=device)
        else:
            f = make_gaussian_bump_ic(n_samples, resolution, device=device)
        
        u = solve_poisson(f)
        # Normalize
        u = u / (u.abs().amax(dim=(-1,-2), keepdim=True) + 1e-8)
        
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # inputs: (N, H, W, 3), targets: (N, H, W, 1)
        X_exp = X.unsqueeze(0).expand(n_samples, -1, -1)
        Y_exp = Y.unsqueeze(0).expand(n_samples, -1, -1)
        
        self.inputs = torch.stack([f, X_exp, Y_exp], dim=-1)
        self.targets = u.unsqueeze(-1)
        print(f"  Dataset size: {len(self.inputs)}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                clip_grad: float = 5.0,
                device: str = 'cpu') -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        pred = model(x)
        # pred: (B, H, W, 1), y: (B, H, W, 1)
        loss = relative_l2(
            pred.squeeze(-1), y.squeeze(-1)
        )
        loss.backward()
        
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               device: str = 'cpu') -> Dict[str, float]:
    model.eval()
    rel_l2_total = 0.0
    grad_err_total = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        
        p = pred.squeeze(-1)
        t = y.squeeze(-1)
        
        rel_l2_total += relative_l2(p, t).item()
        grad_err_total += gradient_error(p, t).item()
    
    n = len(loader)
    return {
        'rel_l2': rel_l2_total / n,
        'grad_err': grad_err_total / n,
    }


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader,
                n_epochs: int = 200,
                lr: float = 3.8e-3,
                weight_decay: float = 1e-6,
                clip_grad: float = 5.0,
                scheduler_type: str = 'step',
                device: str = 'cpu',
                verbose: bool = True) -> Dict:
    """
    Full training run matching paper protocol.
    Default hyperparams are HNO-optimized values from Table 3.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
    
    history = {'train_loss': [], 'test_rel_l2': [], 'test_grad_err': []}
    best_test = float('inf')
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer,
                                  clip_grad=clip_grad, device=device)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = eval_epoch(model, test_loader, device=device)
            history['train_loss'].append(train_loss)
            history['test_rel_l2'].append(metrics['rel_l2'])
            history['test_grad_err'].append(metrics['grad_err'])
            
            if metrics['rel_l2'] < best_test:
                best_test = metrics['rel_l2']
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"Test L2: {metrics['rel_l2']:.4f} | "
                      f"Grad: {metrics['grad_err']:.4f}")
    
    history['best_test_rel_l2'] = best_test
    return history
