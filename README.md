# Residual CFO — Learned OFDM Transceiver

This repository contains code, experiments, and analysis for learning transmitter/receiver components that compensate residual carrier frequency offset (CFO) in OFDM-style communications. The experiments explore learned signaling, joint transmitter/receiver optimization, and comparisons to nominal/analytic baselines.


**Repository layout (top-level highlights)**
`idea3_stage2_awgn/` — stage-based experiment scripts and outputs.
- Common scripts: `transmitter.py`, `receiver.py`, `channel.py`, `comm_core.py`, `reporting.py` in each experiment folder.

**Quick start**
1. Install Python (3.8+) and required packages (check each folder for requirements). Typical packages: `numpy`, `torch`/`tensorflow`, `scipy`, `pandas`, `matplotlib`.

2. Run a script or notebook. Example, from an experiment folder:

```bash
python run_stage1_linear.py
# or open the Jupyter notebook: NOFS_residual_cfo_joint_tx_rx.ipynb
```

3. Inspect outputs in the corresponding `*_outputs/` folder (CSV and trained model checkpoints).

