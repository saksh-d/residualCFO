# Stage 2 Implementation Guide: Pre-V Neural Residual CFO Correction

## Document Purpose

This document is a complete implementation specification for Stage 2 of the residual-CFO-tolerant learned waveform basis system. Stage 1 is already implemented and produces trained, frozen matrices W_tx and V. This guide covers everything needed to build, train, evaluate, and compare Stage 2 on top of Stage 1 without modifying any Stage 1 code or weights.

---

## 1. System Overview and Relationship to Stage 1

### What Stage 1 Produced

Stage 1 trained two complex-valued matrices jointly:

- **W_tx** of shape (M, N) = (64, 45): the transmit synthesis matrix. Maps N=45 data symbols to an M=64 time-domain block.
- **V** of shape (N, M) = (45, 64): the receive analysis matrix. Maps the received M=64 time-domain block back to N=45 symbol estimates.

These were trained so that the effective symbol-domain operator A(δ) = V · Φ_δ · W_tx stays close to a scaled identity matrix over a bounded residual-CFO interval δ ∈ [−δ_max, δ_max].

The full Stage 1 simulation parameters are:
- Block length M = 64
- Payload region K = 50
- Information symbols N = 45
- In-band redundancy R = K − N = 5
- Reserved pilot dimensions P = 4
- Guard/null dimensions G = 10
- Modulation: 16-QAM
- Training Eb/N0: 15 dB
- Evaluation Eb/N0: 10/12 dB
- Residual CFO training interval: δ ∈ [−0.20, 0.20]

**Stage 1 weights must be loaded from checkpoint and frozen for all Stage 2 training. They are never updated.**

### What Stage 2 Adds

Stage 2 inserts one processing step between the received block y and the frozen linear receiver V. A small neural network g_θ estimates the residual CFO δ̂ from the raw received block y. A differentiable analytical phase correction is then applied to y before V processes it.

The full Stage 2 receive chain is:

```
y  →  g_θ(y)  →  δ̂  →  Φ_{−δ̂} applied to y  →  ỹ  →  V (frozen)  →  z₀  →  decisions
```

Mathematically:
- δ̂ = g_θ(y)
- ỹ[n] = y[n] · exp(−j2π · δ̂ · n / M)  for n = 0, 1, ..., M−1
- z₀ = V · ỹ  (V is frozen from Stage 1)

The effective operator after correction becomes:

A(δ_err) = V · Φ_{δ−δ̂} · W_tx

where δ_err = δ − δ̂ is the estimation error. When g_θ estimates well, |δ_err| << |δ|, and the learned basis (which was trained to tolerate bounded residual CFO) handles the remaining error even more comfortably than the raw δ.

**Only g_θ is trained in Stage 2. W_tx and V are never updated.**

---

## 3. Configuration


### System parameters (must match Stage 1 exactly)
- M = 64
- K = 50
- N = 45
- P = 4
- G = 10
- modulation = '16QAM'

### CFO parameters
- delta_max_train = 0.10  — train g_θ on this interval (conservative, matches evaluation target)
- delta_max_eval = 0.20   — evaluate on this wider interval to show generalization
- delta_eval_grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]

### Training parameters
- phase1_epochs = 150      — pretrain g_θ on CFO estimation loss only
- phase2_epochs = 300      — fine-tune with combined loss
- batch_size = 512
- phase1_lr = 1e-3
- phase2_lr = 1e-3
- lr_decay_factor = 0.1
- lr_decay_patience = 50   — reduce lr when val loss plateaus
- lambda_delta = 0.5       — weight of CFO estimation loss in combined objective
- alpha_reg = 0.01         — regularization weight on MMSE correction (if used)

### SNR parameters
- train_EbN0_dB = 12.0
- eval_EbN0_dB = 10.0
- eval_EbN0_sweep = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  — for BER vs SNR curves

### Paths
- stage1_checkpoint_path: path to saved Stage 1 W_tx and V weights
- results_dir

---

## 4. CFO Estimator Architecture (g_θ)

### Input construction

The input to g_θ is the raw received complex block y ∈ C^M. Convert to a real multi-channel feature tensor of shape (batch, 5, M) as follows:

- Channel 0: Re(y)       — real part
- Channel 1: Im(y)       — imaginary part
- Channel 2: |y|         — magnitude
- Channel 3: cos(∠y)     — phase cosine
- Channel 4: sin(∠y)     — phase sine

Rationale: CFO imprints a linear phase ramp across time samples. The cos and sin channels give the network direct access to the phase structure. The magnitude channel is relatively CFO-invariant and provides amplitude context. This 5-channel representation has been shown to be effective for phase-ramp detection tasks.

### Network architecture

Use a 1D convolutional neural network. The input is treated as a sequence of M=64 time steps with 5 features per step.

Architecture (process in order):

1. **Conv block 1**: 1D convolution, 5 input channels → 64 output channels, kernel size 7, padding 3. Followed by GELU activation and LayerNorm over the channel dimension.

2. **Conv block 2**: 1D convolution, 64 → 64 channels, kernel size 5, padding 2. Followed by GELU activation and LayerNorm.

3. **Conv block 3**: 1D convolution, 64 → 32 channels, kernel size 3, padding 1. Followed by GELU activation.

4. **Global average pooling**: AdaptiveAvgPool1d(1) to collapse the time dimension. Output shape: (batch, 32).

5. **Flatten**: shape (batch, 32).

6. **FC layer 1**: Linear(32, 64), GELU activation.

7. **FC layer 2**: Linear(64, 1).

8. **Output activation**: Tanh, scaled by delta_max_train.

Final output: scalar δ̂ per sample in batch, constrained to [−delta_max_train, delta_max_train].

### Initialization

Use default PyTorch initialization (Kaiming uniform for conv and linear layers). Do not use zero initialization for g_θ — unlike the residual corrector pattern, g_θ is an estimator and benefits from standard initialization.

---

## 5. Differentiable Phase Correction Module

This module takes y ∈ C^M and δ̂ (scalar per batch element) and applies the analytical phase correction. It must be fully differentiable so that gradients from the communication loss flow back through the correction into g_θ.

### Correction formula

For each sample in the batch, construct the correction vector:

φ[n] = exp(−j · 2π · δ̂ · n / M)  for n = 0, 1, ..., M−1

Then compute:

ỹ[n] = y[n] · φ[n]

In batch form, φ has shape (batch, M) and is constructed from the scalar δ̂ per sample. The index vector n = [0, 1, ..., M−1] is precomputed and stored as a buffer (not a parameter).

### Complex number handling

PyTorch supports complex tensors natively. Use torch.complex to construct φ from its real and imaginary parts:

- real part: cos(−2π · δ̂ · n / M)
- imaginary part: sin(−2π · δ̂ · n / M)

The multiplication ỹ = y · φ is then elementwise complex multiplication.

### Important: gradient flow

Verify that gradients flow correctly through this module during training. The construction of φ from δ̂ involves trigonometric functions which are differentiable in PyTorch. Use torch.autograd.gradcheck during development to confirm.

---

## 6. OFDM Baseline Construction

The OFDM baseline uses fixed analytical matrices, not learned ones. These are constructed from the DFT matrix and the tone selection matrix, matching the same M, K, N, P, G parameters.

### Construction

- F_M: the M×M unitary DFT matrix, where F_M[k,n] = (1/√M) · exp(−j2πkn/M)
- P_D: the M×N tone selection matrix that selects the N active data tones from the M-point frequency grid. The N active tones are placed in the center of the K-dimensional payload region, excluding the P pilot dimensions and G guard tones. The exact tone placement must match whatever OFDM baseline was used in Stage 1 results to ensure a fair comparison.
- W_OFDM = F_M^H · P_D  — shape (M, N), the OFDM synthesis matrix (IDFT of selected tones)
- V_OFDM = P_D^H · F_M  — shape (N, M), the OFDM analysis matrix (DFT followed by tone selection)

These matrices are fixed. They are not trained. They are constructed once and used as frozen tensors.

### OFDM Stage 2

For the OFDM + Stage 2 system, a separate instance of g_θ (same architecture, different weights) is trained using the OFDM received signals. The correction pipeline is identical:

y → g_θ_OFDM(y) → δ̂_OFDM → Φ_{−δ̂_OFDM} applied to y → ỹ → V_OFDM (frozen) → z₀ → decisions

The OFDM g_θ is trained with the same two-phase protocol, same loss functions, same hyperparameters, and same number of training epochs as the learned system's g_θ. The only difference is the W and V matrices used internally.

This is the fair comparison: both systems receive exactly the same Stage 2 receiver capacity.

---

## 7. Loss Functions

Two losses are used in Stage 2 training.

### Loss 1: CFO Estimation Loss (L_delta)

Normalized mean squared error between estimated and true residual CFO:

L_delta = mean( ( (δ̂ − δ_true) / delta_max_train )² )

True δ is available during training because it is sampled from the training distribution. It is used only as a supervised label for g_θ. It is never available at inference.

This loss encourages g_θ to produce accurate δ̂ before the communication loss is introduced.

### Loss 2: Communication Loss (L_comm)

Binary cross-entropy (BCE) on the decoded bits, computed after the full Stage 2 correction chain.

Process:
1. Apply g_θ to get δ̂
2. Apply phase correction to get ỹ
3. Apply frozen V to get z₀ = V · ỹ
4. Map z₀ to bit log-likelihood ratios (LLRs) using the 16-QAM constellation
5. Apply BCE between predicted LLRs and true transmitted bits

Rationale for BCE over symbol-level MSE: BCE operates directly on the bit level, which aligns with the BER metric being optimized. For 16-QAM, BCE has been found empirically to give better gradient signal than MSE because it correctly penalizes incorrect bit decisions near decision boundaries.

### 16-QAM LLR Computation

For each of the N symbol outputs in z₀, compute soft LLRs for each of the 4 bits using the minimum-distance approximation or the exact log-sum-exp formula against the 16-QAM constellation. Use the exact formula if computational cost permits; the min-distance approximation is acceptable for training.

### Combined Loss

During Phase 1 (pretraining): use L_delta only.

During Phase 2 (fine-tuning): use L_total = L_comm + lambda_delta · L_delta

The lambda_delta weight keeps the CFO estimation signal active during fine-tuning so g_θ does not drift toward a solution that minimizes communication loss at the cost of CFO estimation accuracy.

---

## 8. Data Generation

Stage 2 does not use a fixed dataset. All training and evaluation data is generated on the fly.

### Per-batch generation procedure

For each batch during training:

1. Sample N · batch_size 16-QAM symbols uniformly from the constellation. Organize as (batch_size, N) complex tensor s.

2. Sample batch_size residual CFO values δ ~ Uniform(−delta_max_train, delta_max_train). Shape: (batch_size,).

3. Compute the transmitted block: x = W_tx · s for each sample. Shape: (batch_size, M).

4. Construct the CFO phase ramp matrix Φ_δ for each δ in the batch. Shape: (batch_size, M), applied elementwise to x.

5. Sample AWGN noise z ~ CN(0, σ²·I_M). The noise variance σ² is set according to the training Eb/N0 and the signal energy normalization used in Stage 1. Use the same energy normalization convention as Stage 1.

6. Compute received block: y = Φ_δ · x + z. Shape: (batch_size, M).

7. The labels are: s (for communication loss) and δ (for CFO estimation loss).

### For OFDM baseline training

Identical procedure, substituting W_OFDM for W_tx. The noise variance and energy normalization must be computed consistently with the OFDM transmit signal energy.

---

## 9. Training Protocol
### Phase 1: CFO Estimator Pretraining

Goal: teach g_θ to produce useful δ̂ before introducing the communication loss, which is noisier and harder.

- Freeze W_tx, V (load from Stage 1 checkpoint, set requires_grad=False)
- Initialize g_θ with default initialization
- Optimizer: Adam, lr = phase1_lr
- Loss: L_delta only
- Epochs: phase1_epochs
- Validation: every 10 epochs, compute validation L_delta and MAE of δ̂ on a held-out set of 2000 samples
- Save best g_θ checkpoint based on validation L_delta
- Log: training L_delta, validation L_delta, mean δ̂ MAE per epoch

At the end of Phase 1, verify that g_θ is producing reasonable estimates by checking that the MAE of δ̂ is substantially below delta_max_train. If MAE is still close to delta_max_train / sqrt(3) (which would be the MAE of a constant-zero predictor), something is wrong with the architecture or input features.

### Phase 2: Joint Fine-Tuning

Goal: fine-tune g_θ so the full correction chain minimizes BER.

- Load best Phase 1 checkpoint for g_θ
- W_tx and V remain frozen
- Optimizer: Adam, lr = phase2_lr
- Loss: L_comm + lambda_delta · L_delta
- Epochs: phase2_epochs
- Learning rate schedule: ReduceLROnPlateau on validation L_comm, patience = lr_decay_patience, factor = lr_decay_factor
- Validation: every 10 epochs, compute validation L_comm and BER on a held-out set of 10000 samples at eval_EbN0_dB
- Save best g_θ checkpoint based on validation BER
- Log: training loss, validation loss, validation BER, δ̂ MAE per epoch

### Training for OFDM baseline

Run the identical two-phase training protocol for g_θ_OFDM, substituting W_OFDM and V_OFDM. Use the same random seed for reproducibility. Save OFDM checkpoints separately.

---

## 10. Evaluation Protocol

Evaluate all four systems under identical conditions. The four systems are:

1. **OFDM linear**: y → V_OFDM → z₀ → decisions. No Stage 2. This is the existing Stage 1 OFDM baseline.

2. **OFDM + Stage 2**: y → g_θ_OFDM → δ̂ → Φ_{−δ̂} · y → V_OFDM → z₀ → decisions.

3. **Learned linear**: y → V → z₀ → decisions. No Stage 2. This is the existing Stage 1 learned basis result.

4. **Learned + Stage 2**: y → g_θ → δ̂ → Φ_{−δ̂} · y → V → z₀ → decisions.

For all evaluations, use at least 100,000 transmitted symbols per operating point to ensure reliable BER estimates, especially at low BER values.

### Metric 1: BER vs Residual CFO

Sweep δ over delta_eval_grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20].

For each δ value, hold it fixed (not random) and evaluate BER at eval_EbN0_dB = 10 dB.

Report all four systems on the same plot. X-axis: normalized residual CFO δ. Y-axis: BER on log scale.

### Metric 2: BER vs Eb/N0

Fix δ at two values: 0.05 and 0.10. Sweep Eb/N0 over eval_EbN0_sweep.

Produce two separate plots (one per δ value), each showing all four systems. X-axis: Eb/N0 in dB. Y-axis: BER on log scale.

### Metric 3: CFO Estimation Quality (MAE of δ̂)

For the two Stage 2 systems (OFDM + S2 and Learned + S2), compute:

MAE_delta(δ) = mean( |δ̂ − δ| ) over evaluation samples at each fixed δ value.

Plot MAE_delta vs δ for both systems on the same axes. This demonstrates that the estimators are working and shows whether the learned basis provides any structural advantage for CFO estimation.

### Metric 4: Off-Diagonal Leakage η_off(δ)

For each δ value in the evaluation grid, compute the normalized off-diagonal leakage of the effective operator for all four systems:

η_off(δ) = ||A(δ) − diag(A(δ))||²_F / ||A(δ)||²_F

For Stage 2 systems, compute this using the corrected effective operator:

A_corrected(δ) = V · Φ_{δ−δ̂} · W_tx

where δ̂ is the mean estimate over evaluation samples at that CFO value.

This connects the Stage 2 results back to the operator-level analysis in Stage 1 and confirms that Stage 2 is reducing leakage rather than correcting for something else.

### Metric 5: EVM at δ = 0.10

Compute error vector magnitude for all four systems at δ = 0.10 and eval_EbN0_dB = 10 dB:

EVM = sqrt( mean( ||ẑ − s||² ) / mean( ||s||² ) )

where ẑ are the soft symbol estimates before hard decisions.

### Metric 6: Robustness Window Δ_robust

For each system, compute the maximum |δ| such that BER(δ) ≤ threshold, interpolating from the BER vs δ curve.

Compute for two thresholds: 10^{−2} and 10^{−1}.

Report as a table with all four systems as rows and the two thresholds as columns.

---

## 11. Results to Generate

### Plots (save to results/plots/)

All plots should use consistent styling: the same line colors and markers for each system across all figures. Suggested scheme:
- OFDM linear: blue solid line, circle markers
- OFDM + Stage 2: blue dashed line, square markers
- Learned linear: red solid line, circle markers
- Learned + Stage 2: red dashed line, square markers

**Figure 1**: BER vs residual CFO δ at Eb/N0 = 10 dB, all four systems.

**Figure 2a**: BER vs Eb/N0 at δ = 0.05, all four systems.

**Figure 2b**: BER vs Eb/N0 at δ = 0.10, all four systems.

**Figure 3**: MAE of δ̂ vs δ for OFDM + S2 and Learned + S2.

**Figure 4**: η_off(δ) vs δ for all four systems (extending the existing Fig. 4 from Stage 1).

### Tables (save to results/tables/ as CSV)

**Table 1: BER Summary**

Rows: four systems. Columns: BER at δ=0, δ=0.05, δ=0.10.

**Table 2: Robustness Window**

Rows: four systems. Columns: Δ_robust at BER ≤ 10^{−2}, Δ_robust at BER ≤ 10^{−1}.

**Table 3: EVM and CFO Estimation Quality**

Rows: four systems. Columns: EVM at δ=0.10, MAE_delta at δ=0.10 (for Stage 2 systems only; N/A for linear systems).

---

## 12. Sanity Checks

Run these checks before trusting any results.

**Check 1 — Phase 1 converged**: After Phase 1, MAE of δ̂ should be less than 0.3 × delta_max_train. If it is not, the estimator is not learning from y. Debug input feature construction and verify that the training CFO distribution matches the evaluation distribution.

**Check 2 — Stage 2 does not hurt at δ=0**: At zero residual CFO, Stage 2 BER should be approximately equal to or slightly worse than the linear Stage 1 BER for the same system. Stage 2 should never give large gain at δ=0. If it does, the correction is fitting something other than CFO.

**Check 3 — Gain grows with |δ|**: The BER improvement from Stage 2 over Stage 1 should increase as |δ| increases. At small δ, Stage 1 is nearly perfect and Stage 2 has little to correct. At large δ, Stage 2 should give progressively more gain.

**Check 4 — Learned + S2 beats OFDM + S2**: This is the key result. If it does not hold, the basis contribution claim is weakened. Investigate whether the OFDM g_θ is being trained fairly with equal capacity and budget.

**Check 5 — η_off decreases after Stage 2**: The operator-level leakage metric should decrease for Stage 2 systems. If BER improves but η_off does not decrease, Stage 2 may be exploiting constellation geometry rather than correcting CFO-induced leakage. This needs to be understood before claiming a mechanism.

**Check 6 — Gradient flow through correction**: Run a small test to confirm that gradients from L_comm flow back through the phase correction module into g_θ. Use torch.autograd.gradcheck or simply verify that g_θ parameters receive nonzero gradients during Phase 2 training.


## 14. Key Design Rationale Summary

| Decision | Rationale |
|---|---|
| g_θ operates on y, not z₀ | CFO phase ramp lives in time domain; estimating from y gives richer information than from compressed symbol-domain z₀ |
| 1D-CNN architecture | CFO produces local phase ramp structure across time steps; convolutions exploit this local structure better than global MLP |
| Two-phase training | Phase 1 establishes a useful δ̂ signal before the noisier L_comm gradient is introduced; prevents g_θ from converging to a trivial solution |
| BCE on bits for L_comm | Directly optimizes the BER metric; more informative gradient for 16-QAM decisions near decision boundaries than symbol-level MSE |
| Separate g_θ per system | OFDM and learned y signals have different structure due to different W_tx; a shared g_θ would disadvantage one system unfairly |
| Frozen W_tx and V | Preserves the Stage 1 contribution cleanly; Stage 2 gain is attributable to g_θ alone, not to waveform retraining |
| delta_max_train < delta_max_eval | Training on a conservative interval and evaluating on a wider one tests generalization; if Stage 2 helps beyond the training interval, it indicates the estimator has learned meaningful structure |
