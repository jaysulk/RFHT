# Regularized Hartley Neural Operator (RHNO)

Extension of the Hartley Neural Operator (HNO) from:

> **"When Real Beats Complex: Hartley Neural Operators and the Role of
> Spectral Basis Alignment in PDE Learning"**  
> Jason Sulskis, Sathya Ravi

---

## Core Idea

The HNO paper establishes that real-valued Hartley spectral convolution
outperforms complex Fourier on elliptic PDEs and broadband initial conditions.
The implementation uses dense weight matrices `Weven`, `Wodd` per frequency
quadrant (8 matrices total).

**This work:** replaces those dense weights with a butterfly-factorized
structure inspired by Jones (2022):

> Keith John Jones, *The Regularized Fast Hartley Transform:
> Low-Complexity Parallel Computation of the FHT in One and Multiple
> Dimensions*, Springer, 2nd ed. 2022.

Jones' RFHT is designed for hardware efficiency on FPGA/ASIC via:
- Partitioned memory with dibit-reversal reordering between stages
- 8-fold parallelism from the "double butterfly" processing element
- A simple, regular architecture that is resource-efficient and scalable

We translate this algorithmic regularity into a **structural prior on
learned spectral weights**: instead of each frequency bin coupling
arbitrarily to all others (dense), bins are coupled in a butterfly
pattern matching the RFHT compute graph.

---

## Parameter Reduction

| Config (modes=12, width=32) | Total Params | Spectral Params | vs HNO |
|---|---|---|---|
| HNO (dense, paper) | 3,547,873 | 3,538,944 | 1.00x |
| RHNO (3 stages, log₂M) | 918,241 | 909,312 | 0.26x |
| RHNO (2 stages) | 623,329 | 614,400 | 0.17x |
| RHNO (1 stage, max reg) | 328,417 | 319,488 | 0.09x |

At large mode counts, HNO spectral params scale O(M²) while RHNO scales
O(M log M) — mirroring the FFT vs DFT complexity improvement.

---

## Architecture

```
Input [u0, x, y, t]  (or [f, x, y] for elliptic)
       |
  Input Projection (Linear → GELU → Linear)
       |
  ┌─── Spectral Block ───────────────────────────────────┐
  │  DHT (via Re-Im FFT trick)                          │
  │  Even/Odd decomposition: H±[k] = (H[k] ± H[-k])/2  │
  │  Extract 4 frequency quadrants                       │
  │  Per quadrant: RFHT Butterfly Stack                  │
  │    Stage 0: pair bins [k, k+M/2], learn 2×2 mixing  │
  │    Stage 1: pair bins [k, k+M/4], ...               │
  │    ... (log₂M stages total)                          │
  │    Dibit-reversal permutation between stages         │
  │  Even/odd channel mixing (1×1, lightweight)          │
  │  Scatter back to full frequency grid                 │
  │  Inverse DHT                                         │
  │  + Residual 1×1 bypass                               │
  │  GELU + InstanceNorm                                 │
  └─────────────────────────────────────────────────────┘
       | (× 3 blocks for time-dep, × 4 for elliptic)
  Output Projection (Linear → GELU → Linear)
       |
  Output u(x,y,t)
```

### Key Difference from HNO

| | HNO (paper) | RHNO (this work) |
|---|---|---|
| Spectral weights | 8 dense matrices | Butterfly-factorized per quadrant |
| Params per mode | O(M²) | O(M log M) |
| Freq coupling | Arbitrary | Structured (RFHT graph) |
| Regularization | Implicit (weight decay) | Structural (factorization) |
| Jones connection | Citation only | Architecture inspired by RFHT |

---

## Files

```
rhno/
├── transforms/
│   └── rfht.py          # DHT, dibit-reversal, ButterflyStage, RFHTSpectralConv2d
├── models/
│   └── rhno.py          # HNOSpectralConv2d (paper), SpectralBlock, HartleyNeuralOperator
│                        # make_hno() reproduces paper; make_rhno() is this work
├── utils/
│   └── training.py      # Data generators, PDE solvers, training loop
└── experiments/
    ├── parameter_analysis.py   # Parameter counts, scaling, ablation
    └── run_comparison.py       # HNO vs RHNO on Poisson/Heat
```

---

## Running Experiments

```bash
# Parameter analysis (no GPU needed)
python experiments/parameter_analysis.py

# Quick HNO vs RHNO on Poisson (best-case PDE from paper)
python experiments/run_comparison.py --pde poisson --resolution 64 --epochs 100

# Full resolution (paper setting)
python experiments/run_comparison.py --pde poisson --resolution 128 --epochs 200

# Ablate butterfly depth
python experiments/run_comparison.py --pde poisson --butterfly_stages 2
python experiments/run_comparison.py --pde poisson --butterfly_stages 3

# Heat equation (GRF ICs)
python experiments/run_comparison.py --pde heat --ic grf --epochs 200
```

---

## Research Questions for the Paper

1. **Does structural regularization hurt elliptic accuracy?**  
   The paper shows HNO achieves 0.06x vs FNO on Poisson/GRF. Does RHNO
   maintain this while using 4-6x fewer spectral parameters?

2. **Does butterfly depth matter?**  
   Jones uses log₄(M) stages (radix-4) for hardware efficiency. Does
   shallower factorization (fewer stages = more regularization) help or
   hurt, and for which PDEs?

3. **Does the dibit-reversal ordering add structure?**  
   Ablation: butterfly with vs without the dibit-reversal permutation
   between stages.

4. **Parameter efficiency vs accuracy Pareto frontier:**  
   Plot RHNO (various depths) vs HNO on the error/params plane.
   The hypothesis: RHNO dominates on elliptic PDEs where the Green's
   function alignment (Appendix E) is the dominant factor, not parameter count.

---

## Connection to Jones (2022)

Jones' "regularization" is computational — it refers to the regular,
predictable memory access pattern of the RFHT, enabling efficient FPGA
implementation. We borrow this structural regularity as a *statistical*
regularizer: the butterfly factorization constrains the learned spectral
weights to a submanifold of the full dense weight space, analogous to how
regularization in standard ML constrains the hypothesis class.

The dibit-reversal permutation between stages (Jones Ch. 4) determines
which frequency bins are coupled at each stage, creating a specific
sparsity pattern in the effective weight matrix. This pattern has a
natural interpretation: nearby frequencies (in the dibit-reversed ordering)
are more strongly coupled, which may be appropriate for smooth PDE solutions
where spectral energy decays monotonically.

A future direction: use the actual RFHT twiddle factor structure (fixed,
not learned) as initialization for the butterfly weights, then fine-tune.
This would give a "warm start" from the known optimal real-arithmetic
transform.
