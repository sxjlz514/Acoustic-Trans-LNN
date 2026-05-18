# Readme.md
# Trans-LNN: Parameter-Efficient Continuous-Time Liquid Neural Operators with STLT for Acoustic Transient Manifold Tracking

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![IEEE SPL](https://img.shields.io/badge/IEEE-Signal_Processing_Letters-orange.svg?style=flat-square)](https://ieeexplore.ieee.org/)
[![Academic-Project](https://img.shields.io/badge/DUT-Sensing_and_Vibration_Lab-blue.svg?style=flat-square)](https://me.dlut.edu.cn/)

This repository provides the official implementation, physics-informed data generation engines, pre-processed acoustic datasets, and pre-trained continuous-time weights for the manuscript: **"Parameter-Efficient Continuous-Time Liquid Neural Operators With Short-Time Laplace Transform for Acoustic Transient Manifold Tracking and Analysis"** (IEEE Signal Processing Letters).

---

## 🚀 Core Highlights

* **Ultra-Compact Footprint:** Achieves state-of-the-art long-horizon extrapolation (4000 future frames from a 100-frame window) using only **67,930 parameters (<300 KB memory footprint)**, purpose-built for direct on-chip SRAM edge deployment.
* **Continuous-Time Operators:** Built upon advanced Closed-Form Continuous-Time cells (CfC), reformulating discrete deep learning curve-fitting into a rigorous continuous Neural ODE vector field.
* **Intrinsic Manifold Separation:** Embeds mathematical integral invariants to naturally damp out intense stochastic noise ($I_{\text{noise}}^{\text{crit}} = 0.0730$) while recursively amplifying micro-level structural defects ($I_{\text{leak}}^{\text{crit}} = 0.0135$) under the sub-noise limit.

---

## 📂 Repository Structure

 ```text
 ├── code/
 │   ├── adsr.py                               # Step 1: Acoustic transient segmentation operator
 │   ├── STLT.py                               # Step 2: Short-Time Laplace Transform manifold extraction
 │   ├── SingleStep_Trans_LNN.py               # Step 3: Proposed Phase I Training (Teacher Forcing)
 │   ├── FullStep_Trans_LNN.py                 # Step 4: Proposed Phase II Fine-Tuning (Autoregressive)
 │   ├── check_ratio_Trans_lNN.py              # Step 5: Curriculum ratio evaluation for Proposed model
 │   ├── Trans_LNN_Check.py                    # 🎯 Quick Start: 1x2 Publication-grade Fig.2  reproduction script
 │   │
 │   ├── SingleStep_Trans_LNN_WithoutSigma.py  # Ablation Baseline 1: Phase I without transient rates
 │   ├── FullStep_Trans_LNN_WithoutSigma.py     # Ablation Baseline 1: Phase II without transient rates
 │   ├── Trans_LNN_WithoutSigma_Check.py       # Evaluation & Boundary Check for Baseline 1
 │   │
 │   ├── Single_pure_LNN.py                    # Ablation Baseline 2: Phase I pure CfC without attention
 │   ├── FullStep_pure_LNN.py                  # Ablation Baseline 2: Phase II pure CfC without attention
 │   ├── pure_LNN_check.py                     # Evaluation & Boundary Check for Baseline 2
 │   └── super_Trans_LNN_Check.py              # Downstream: Zero-shot temporal halving (dt -> 0.5dt)
 │
 ├── csv/
 │   └── 036-C_Attack.csv                      # Curated baseline fluid manifold table for 36-C pipe
 │
 ├── wav/
 │   ├── 036-C.wav                             # Raw acoustic recording of the Flöjtlein 2' stop
 │   └── 036-C_Attack.wav                      # Isolated physical transient attack phase audio
 │
 └── models/
    ├── Trans-LNN_withSigma_ratio1.pth        # Proposed Evolved Complete Model (Used for Fig. 2)
    ├── Trans-LNN_withSigma_ratio0.8.pth      # Proposed Curriculum Intermediate State (Ratio = 0.8)
    ├── TransLNN_withSigma_ratio0.pth         # Ablated Baseline 1 Evolved Weight (Ratio = 0)
    ├── Trans-LNN_withSigma_ratio0_fullstep_Start.pth # Ablated Baseline 1 Evolved Weight (Ratio = 0)
    ├── Trans_LNN_withoutSigma_ratio1.pth     # Ablated Baseline 1 Evolved Weight (Ratio = 1.0)
    ├── Trans_LNN_withoutSigma_ratio0.8.pth   # Ablated Baseline 1 Intermediate State
    ├── pureLNN_ratio1.pth                    # Ablated Baseline 2 Evolved Weight (Ratio = 1.0)
    ├── pureLNN_ratio0.8.pth                  # Ablated Baseline 2 Intermediate State
    └── pureLNN_ratio0.pth                    # Ablated Baseline 2 Initial State
  ```
## 1. Environment Setup & Requirements

### Hardware Prerequisites
*   **Minimum Memory**: 16 GB RAM (Required for multi-step adaptive step-size continuous integration).
*   **Processor**: $\ge 8$ Logical CPU Cores or Nvidia GPU with CUDA support.
*   **Performance Note**: If your host machine encounters Out-of-Memory (OOM) or CPU bottlenecking during long-horizon ODE integration, please lower the multi-threading worker settings (`num_workers` or thread-counts) inside the dataloader configuration sections.

### Dependency Installation
Ensure all core scientific computing and Neural ODE dependencies are correctly installed before execution:
When using .py files, please modify the file path (to your own path).
```bash
pip install torch>=2.0 numpy pandas scipy matplotlib librosa ncps
```
##  2. Quick Start: Reproduce Publication Figures (Fig. 2)

To verify the core assertion of Intrinsic Separation and Residual Morphology without running the multi-stage training pipeline from scratch, we provide an all-in-one verification script.

Executing `code/Trans_LNN_Check.py` will automatically load the evolved pre-trained checkpoint (`models/Trans-LNN_withSigma_ratio1.pth`), inject the specified boundary perturbations, perform 100% offline autonomous tracking, and output the publication-grade 1×2 aligned residual manifold plot into the `./inference_results/` directory:

```bash
python code/Trans_LNN_Check.py
```
Stochastic Noise Regime: Set DEFECT_MODE = 'noisy_baseline_coupled' to observe how the continuous-time solver symmetrically dampens and cancels severe high-frequency stochastic noise.
Structural Defect Regime: Set DEFECT_MODE = 'air_leak' to observe the low-frequency directional drift triggered by permanent vector field alteration.
## 3. Complete Training & Fine-Tuning Pipeline
To completely rebuild the continuous-time operator trajectory from scratch, execute the following execution sequences under strict order:

**Step 1: Acoustic Signal Phase Slicing**

Isolate the pure physical transient attack segment from raw audio streams (wav/036-C.wav):
```bash
python code/adsr.py
```

**Step 2: Complex Trajectory Coordinate Extraction**
    Map the time-series segments into the continuous $s$-plane coordinates ($\sigma$, $M$ manifold tables) via Short-Time Laplace Transform to construct the baseline CSV tables:
```bash
    python code/STLT.py
```
    
**Step 3: Phase I - Teacher Forcing EmbeddingInitialize attention matrices and hidden** embeddings under external ground-truth stabilization guidance:
```bash
   python code/SingleStep_Trans_LNN.py
```

**Step 4: Phase II - Full Autoregressive Integration Fine-Tuning**  
    Close the recursive continuous feedback loop to minimize accumulated spectral curvature over the ultra-long 4000-frame integration horizon:
```bash
    python code/FullStep_Trans_LNN.py
```    
Step 5: Model Evaluation & RMSE VerificationValidate the model's convergence and tracking accuracy under different curriculum transition ratios (0.0, 0.8, 1.0):Bashpython code/check_ratio_Trans_lNN.py


> 🚨 **Critical Note on Model Selection & Ablation Collapse**
> *   **Curriculum Transition Evaluation**: When verifying intermediate scaling states via evaluation scripts, you MUST load the strictly corresponding frozen epoch weights mapped in the internal configuration dictionary. For instance, evaluating ratio = 0.8 requires binding `models/Trans-LNN_withSigma_ratio0.8.pth`.
> *   **Ablated Baselines Instability**: As analyzed in the manuscript's ablation section, evaluating non-liquid or non-attention ablated baselines under a pure autonomous ratio (1.0) using `code/Trans_LNN_WithoutSigma_Check.py` or `code/pure_LNN_check.py` will instantly trigger numerical gradient divergence and trajectory integration collapse. For these baselines, it is highly recommended to evaluate using the optimal pre-trained baseline assets frozen in our directory.
---

## 📊 4. Downstream Core Experiments

### Experiment 1: Defect Perception Boundary Analysis
To evaluate the ultimate sensitivity limits under the sub-noise regime, execute the checking scripts for the proposed model and its ablated variants. It validates the tracking capacity under the critical structural defect boundary ($I_{\text{leak}} = 0.0135$) and coupled stochastic noise threshold ($I_{\text{noise}} = 0.0730$):

```bash
python code/Trans_LNN_Check.py
python code/Trans_LNN_WithoutSigma_Check.py
python code/pure_LNN_check.py
```
Experiment 2: Super-Sampling Transcendence (Zero-Shot Scale Invariance)To prove that the critical bifurcation boundary represents an inherent geometric property of the continuous topological manifold rather than a discrete numerical artifact, run the temporal halving simulation ($dt \to 0.5dt$). It demonstrates that the intrinsic separation ratio remains rigidly invariant ($\approx 5.4\times$):
```bash
python code/super_Trans_LNN_Check.py
```

## 5. Numerical Variance & Chaotic Dynamical DisclaimerDeterministic Synchronization:
### All stochastic fields, random channel noise, and data batch samplers are strictly bound to seed = 514 inside the validation blocks to rigidly anchor the exact baseline statistics reported in the text (e.g., the 0.075 noise boundary limit).
### Hardware Compiler Drift: Because the Trans-LNN operates as a continuous-time neural ordinary differential equation (Neural ODE), its forward trajectory vector fields are highly sensitive to the microscopic accumulation of floating-point arithmetic. Switching between different execution backends (e.g., from an RTX 4090 to an NVIDIA A100 GPU) or changing compiler optimizations (CUDA/cuDNN micro-versions) may introduce minute deviations, slightly shifting the empirical critical boundary points within a narrow margin between 0.071 and 0.075.
### 📌 Note: Crucially, the order of magnitude and the core intrinsic separation ratio (rigidly invariant at $\approx 5.4\times$) remain perfectly stable across all heterogeneous execution platforms. We highly appreciate your scholarly and engineering understanding of these inherent continuous-time non-linear dynamical traits.
