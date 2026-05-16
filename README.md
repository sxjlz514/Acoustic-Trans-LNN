# README.md

## Repository Overview
This repository contains the complete implementation, pre-processed datasets, and pre-trained model weights for the manuscript. 

To facilitate seamless verification, we have already included **pre-trained model checkpoints (`.pt`/`.pth`), processed audio segments, and computed STLT feature tables** within the folder. Reviewers can either execute the full pipeline from scratch or directly run the evaluation/checking scripts using our provided assets.

---

## 1. System Requirements & Environment Setup

### Hardware Requirements
* **Minimum Memory:** 16 GB RAM (required for long-horizon continuous-time ODE integration).
* **Minimum Processor:** 8 Logical CPU Cores.
* > ⚠️ **Performance Note:** If your host machine does not meet these specifications or encounters Out-of-Memory (OOM) / CPU bottlenecking, please lower the multi-threading worker settings (e.g., decrease `num_workers` or thread-counts) inside the dataloader configuration sections of the scripts.

### Pre-requisites
1. **Library Installation:** Install the required Python dependencies listed in the code headers (e.g., `torch`, `numpy`, `scipy`, `librosa`).
2. **Path Modification:** Before executing any script, please update the input/output directory strings in the **running execution block** (the `if __name__ == '__main__':` section) to match your local absolute paths.

---

## 2. Complete Execution Pipeline (Step-by-Step)

To reproduce the experimental results from scratch, please follow this strict sequence:

### Step 1: Audio Segmentation
* **Script:** `adsr.py`
* **Operation:** Run this file first to slice the raw acoustic transients and isolate the continuous **Attack (A) phase** audio segments.

### Step 2: Manifold Feature Computation
* **Script:** Run the Short-Time Laplace Transform (STLT) processing script.
* **Operation:** Computes the transient rates ($\sigma$) and magnitudes ($M$) coordinates from the sliced Attack audio to construct the baseline interaction manifold tables.

### Step 3: Phase I Training (Teacher Forcing)
* **Script:** Execute scripts prefixed with **`SingleStep`** (e.g., `SingleStep_Trans_LNN.py`).
* **Operation:** Trains the foundational spatial-attention and embedding projections using external ground-truth guidance.

### Step 4: Phase II Fine-Tuning (Autoregressive Integration)
* **Script:** Execute the corresponding scripts prefixed with **`FullStep`** (e.g., `FullStep_Trans_LNN.py`).
* **Operation:** Closes the recursive feedback path to minimize accumulated spectral curvature across the 4,000-frame extrapolation horizon.

### Step 5: Model Evaluation & RMSE Verification
* **Script:** Run the scripts containing the **`check`** suffix (e.g., `check_ratio_rmse.py`) to validate model performance under different curriculum transition ratios ($0.0$, $0.8$, $1.0$).

> 🚨 **CRITICAL NOTE ON MODEL SELECTION & ABLATION COLLAPSE:**
> * **For the Proposed Trans-LNN:** When verifying a curriculum ratio less than 1 (i.e., `ratio < 1`), you **MUST** load the strictly mapped epoch checkpoint specified in the script's internal dictionary. For instance, evaluating `ratio = 0.8` requires loading the model frozen at `epoch = 1800`.
> * **For the Ablated Benchmarks:** As rigorously demonstrated in the manuscript's ablation section, running the other two ablated variants under `ratio = 1.0` will *inherently trigger an integration trajectory collapse*, making them impossible to fine-tune stably. For these baselines, it is highly recommended to evaluate using either the maximum epoch checkpoint or the optimal pre-trained model pre-loaded in our directory.

---

## 3. Downstream Core Experiments

* **Defect Perception Analysis:** Dedicated evaluation code for the 0.0142 structural defect boundary can be found in the `Defect_Detection/` subfolder.
* **Super-Sampling Transcendence:** The zero-shot scale invariance evaluation ($dt \rightarrow dt/2$) is implemented within the `Transcendence_Experiment/` subfolder.

---

## 4. Reproducibility & Numerical Variance Disclaimer

* **Deterministic Seed:** All stochastic fields and noise generations are natively locked to **`seed = 514`** to anchor the exact paper benchmarks (such as the 0.075 noise boundary).
* **Hardware Shift Notice:** The Trans-LNN acts as a continuous-time neural ODE network. Due to the high-order sensitivity of non-linear dynamical systems, running the scripts under different hardware backends (e.g., switching from RTX 4090 to A100, or using different CUDA/cuDNN compiler versions) may introduce minor micro-level floating-point drifts (e.g., critical boundaries shifting slightly between 0.071 and 0.075). 
* However, the **order of magnitude** and the **intrinsic separation ratio** ($\approx 5.4\times$) remain rigidly invariant across all environments. We highly appreciate your scholarly understanding of these inherent physical/numerical behaviors.