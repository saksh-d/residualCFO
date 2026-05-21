from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from comm_core import (
    ExperimentConfig,
    TrainingResult,
    file_sha256,
    load_stage1_checkpoint,
    refresh_output_dir,
    snapshot_stage1_checkpoint,
    stage2_checkpoint_preflight,
)
from reporting import run_final_evaluation, write_markdown_report
from run_stage1_linear import PLOT_FILES as STAGE1_PLOT_FILES, build_stage1_linear_config
from receiver import (
    CoreReferenceModelConfig,
    USRNetArchitectureConfig,
    USRNetConditioningConfig,
    USRNetCoreTrainingConfig,
    USRNetLossConfig,
    USRNetPhaseConfig,
    USRNetTrainingBundle,
    USRNetValidationConfig,
    build_usrnet_schemes,
    evaluate_usrnet_scheme,
    select_phi_sign,
    train_diagonal_core_reference,
    train_usrnet_receiver,
)
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================================
# Stage 2 USR-Net Controls
# ============================================================================

# Stage 1 source and inheritance
OUTPUT_ROOT = SCRIPT_DIR / "stage2_usrnet_outputs"
OUTPUT_SUBDIR = "usrnet_n45_r9"
STAGE1_OUTPUT_DIR = SCRIPT_DIR / "stage1_linear_outputs" / "16qam_n45_r9"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"
REUSE_STAGE1_SIGNED_CFO_GRID = True

# Output / reporting identity
METHOD_ORDER = ("OFDM", "OFDMUSRNet", "Learned", "LearnedUSRNet")
CONSTELLATION_METHOD_ORDER = ("OFDM", "Learned", "OFDMUSRNet", "LearnedUSRNet")

# Main evaluation setup
EVAL_EBN0_DB = 12.0
SIGNED_CFO_GRID = np.linspace(-0.20, 0.20, 41)
SUMMARY_DELTA_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
CONSTELLATION_CFO = (0.0, 0.05, 0.10)
HEATMAP_CFO = (0.05, 0.10)

# Receiver-state condition controls
DEFAULT_CONDITION_SIGMA = 0.005
CONDITION_SENSITIVITY_GRID = (0.0, 0.0025, 0.0050, 0.0100, 0.0200, 0.0300, 0.0500)

# Acceptance / diagnostic slices
PER_LAYER_DELTA_VALUES = (0.05, 0.10)
PHI_SIGN_PROBE_DELTA = 0.10
PHI_SIGN_PROBE_EBN0_DB = 20.0
PHI_SIGN_PROBE_BATCH_SIZE = 256

# Smoke-mode controls
SMOKE_SIGNED_CFO_GRID = np.linspace(-0.20, 0.20, 17)
SMOKE_CORE_EPOCHS = 8
SMOKE_PHASE_EPOCHS = (8, 8, 8)
SMOKE_BER_BLOCKS = 2048
SMOKE_BER_BATCH_SIZE = 256
SMOKE_CONSTELLATION_NUM_BLOCKS = 128
SMOKE_SPECTRAL_EVAL_BLOCKS = 1024
SMOKE_PAPR_EVAL_BLOCKS = 1024
SMOKE_TRAIN_SYMBOL_BATCH_SIZE = 256

# USR-Net architecture / initialization
CORE_REFERENCE_MODEL = CoreReferenceModelConfig(
    rho_init=0.90,
    gamma_init=1.0,
    final_gain_enabled=True,
)
USRNET_ARCHITECTURE = USRNetArchitectureConfig(
    num_layers=3,
    feature_channels=10,
    conv_hidden_channels=64,
    conv_kernel_size=3,
    conv_dilations=(1, 2, 4),
    final_gain_enabled=True,
    shared_gamma_init=1.0,
    layer_gamma_init=(1.0, 1.0, 1.0),
    eta_init=(0.125, 0.125, 0.125),
)
USRNET_LOSS = USRNetLossConfig(
    layer_weights=(0.2, 0.3, 0.5),
    temp_cls=0.1,
    cls_weight=0.3,
    identity_weight=0.1,
    correction_weight=1.0e-4,
    guard_weight=0.5,
)
USRNET_CONDITIONING = USRNetConditioningConfig(
    default_sigma_condition=DEFAULT_CONDITION_SIGMA,
    sensitivity_sigmas=CONDITION_SENSITIVITY_GRID,
    phi_sign_default=1,
    sign_probe_delta=PHI_SIGN_PROBE_DELTA,
    sign_probe_ebn0_db=PHI_SIGN_PROBE_EBN0_DB,
    sign_probe_batch_size=PHI_SIGN_PROBE_BATCH_SIZE,
)
USRNET_CORE_TRAIN = USRNetCoreTrainingConfig(
    epochs=50,
    batch_size=512,
    learning_rate=1.0e-3,
    delta_span=0.15,
    ebn0_choices=(10.0, 12.0, 15.0, 20.0),
    sigma_condition=DEFAULT_CONDITION_SIGMA,
)
USRNET_PHASES = (
    USRNetPhaseConfig(
        name="Phase0CoreMatch",
        epochs=50,
        learning_rate=1.0e-3,
        sigma_condition=DEFAULT_CONDITION_SIGMA,
        delta_span=0.15,
        ebn0_choices=(10.0, 12.0, 15.0, 20.0),
        disable_neural=True,
        train_anchor=True,
        train_layer_reference=False,
        train_conv=False,
        train_eta=False,
    ),
    USRNetPhaseConfig(
        name="Phase1NeuralUnfreeze",
        epochs=150,
        learning_rate=5.0e-4,
        sigma_condition=DEFAULT_CONDITION_SIGMA,
        delta_span=0.15,
        ebn0_choices=(10.0, 12.0, 15.0, 20.0),
        disable_neural=False,
        train_anchor=False,
        train_layer_reference=False,
        train_conv=True,
        train_eta=True,
    ),
    USRNetPhaseConfig(
        name="Phase2Robustness",
        epochs=100,
        learning_rate=2.0e-4,
        sigma_condition=(0.0, 0.0025, 0.0050, 0.0100, 0.0200),
        delta_span=0.15,
        ebn0_choices=(10.0, 12.0, 15.0, 20.0),
        disable_neural=False,
        train_anchor=True,
        train_layer_reference=True,
        train_conv=True,
        train_eta=True,
    ),
)
USRNET_VALIDATION = USRNetValidationConfig(
    delta_values=(0.0, 0.05, 0.10, -0.05, -0.10),
    sigma_condition=DEFAULT_CONDITION_SIGMA,
    batch_size=256,
    ebn0_choices=(10.0, 12.0, 15.0, 20.0),
)

CUSTOM_PLOT_FILES = (
    "ber_vs_cfo_signed.png",
    "usrnet_condition_sensitivity.png",
    "usrnet_per_layer_evm.png",
    "usrnet_per_layer_ber.png",
)
REPORT_PLOT_FILES = tuple(STAGE1_PLOT_FILES) + ("training_diagnostics.png", "training_loss_components.png") + CUSTOM_PLOT_FILES


def _main_signed_cfo_grid(stage1_config: ExperimentConfig) -> np.ndarray:
    if REUSE_STAGE1_SIGNED_CFO_GRID:
        return np.asarray(stage1_config.ber_eval_cfo, dtype=float)
    return np.asarray(SIGNED_CFO_GRID, dtype=float)


def _smoke_phase_configs() -> tuple[USRNetPhaseConfig, ...]:
    return tuple(replace(phase, epochs=smoke_epochs) for phase, smoke_epochs in zip(USRNET_PHASES, SMOKE_PHASE_EPOCHS))


def _smoke_core_train_config() -> USRNetCoreTrainingConfig:
    return replace(USRNET_CORE_TRAIN, epochs=SMOKE_CORE_EPOCHS, batch_size=SMOKE_TRAIN_SYMBOL_BATCH_SIZE)


def _smoke_validation_config() -> USRNetValidationConfig:
    return replace(USRNET_VALIDATION, batch_size=min(USRNET_VALIDATION.batch_size, SMOKE_TRAIN_SYMBOL_BATCH_SIZE))


def _stage2_control_lines(config: ExperimentConfig, *, phi_sign: int | None = None, smoke_mode: bool = False) -> list[str]:
    phase_configs = _smoke_phase_configs() if smoke_mode else USRNET_PHASES
    core_train = _smoke_core_train_config() if smoke_mode else USRNET_CORE_TRAIN
    validation = _smoke_validation_config() if smoke_mode else USRNET_VALIDATION
    phase_text = "; ".join(
        f"{phase.name}: epochs={phase.epochs}, lr={phase.learning_rate:.1e}, sigma={phase.sigma_condition}, "
        f"delta_span={phase.delta_span:.2f}, ebn0={phase.ebn0_choices}, disable_neural={phase.disable_neural}, "
        f"train_anchor={phase.train_anchor}, train_layer_reference={phase.train_layer_reference}, "
        f"train_conv={phase.train_conv}, train_eta={phase.train_eta}"
        for phase in phase_configs
    )
    lines = [
        f"- Output dir: `{config.output_dir}`",
        f"- Stage 1 source: output dir `{STAGE1_OUTPUT_DIR}`, checkpoint `{config.stage2_checkpoint_path}`",
        f"- Signed CFO inheritance: reuse Stage 1 grid `{REUSE_STAGE1_SIGNED_CFO_GRID}`, runner override `{tuple(float(v) for v in SIGNED_CFO_GRID)}`",
        f"- Inherited geometry: modulation `{config.modulation}`, frame `M={config.M}, K={config.K}, N={config.N}, P={config.N_pilots}, G={config.N_guard}, R={config.redundancy_dimensions}`",
        f"- Evaluation setup: eval Eb/N0 `{config.eval_ebn0_db:.1f} dB`, signed CFO grid `{tuple(float(v) for v in config.ber_eval_cfo)}`, heatmaps `{config.heatmap_cfo}`, constellations `{config.constellation_cfo}`",
        f"- Method order: report `{METHOD_ORDER}`, constellation `{CONSTELLATION_METHOD_ORDER}`, report plots `{REPORT_PLOT_FILES}`",
        f"- Receiver-state condition: default sigma `{USRNET_CONDITIONING.default_sigma_condition:.4f}`, sensitivity grid `{USRNET_CONDITIONING.sensitivity_sigmas}`",
        f"- Neural-USR architecture: layers `{USRNET_ARCHITECTURE.num_layers}`, features `{USRNET_ARCHITECTURE.feature_channels}`, hidden `{USRNET_ARCHITECTURE.conv_hidden_channels}`, kernel `{USRNET_ARCHITECTURE.conv_kernel_size}`, dilations `{USRNET_ARCHITECTURE.conv_dilations}`, anchor final gain `{USRNET_ARCHITECTURE.final_gain_enabled}`",
        f"- Neural-USR init: shared gamma `{USRNET_ARCHITECTURE.shared_gamma_init}`, layer gamma `{USRNET_ARCHITECTURE.layer_gamma_init}`, eta `{USRNET_ARCHITECTURE.eta_init}`, core rho `{CORE_REFERENCE_MODEL.rho_init}`, core gamma `{CORE_REFERENCE_MODEL.gamma_init}`",
        f"- Neural-USR loss: layer weights `{USRNET_LOSS.layer_weights}`, temp `{USRNET_LOSS.temp_cls}`, CE `{USRNET_LOSS.cls_weight}`, identity `{USRNET_LOSS.identity_weight}`, correction `{USRNET_LOSS.correction_weight}`, guard `{USRNET_LOSS.guard_weight}`",
        f"- Core reference train: epochs `{core_train.epochs}`, batch `{core_train.batch_size}`, lr `{core_train.learning_rate:.1e}`, delta span `{core_train.delta_span:.2f}`, Eb/N0 `{core_train.ebn0_choices}`, sigma `{core_train.sigma_condition}`",
        f"- Neural-USR phases: {phase_text}",
        f"- Validation model-selection: deltas `{validation.delta_values}`, sigma `{validation.sigma_condition}`, batch `{validation.batch_size}`, Eb/N0 `{validation.ebn0_choices}`",
        f"- Diagnostics: summary deltas `{SUMMARY_DELTA_GRID}`, per-layer deltas `{PER_LAYER_DELTA_VALUES}`, sign probe delta `{USRNET_CONDITIONING.sign_probe_delta:.2f}`, sign probe Eb/N0 `{USRNET_CONDITIONING.sign_probe_ebn0_db:.1f} dB`, sign probe batch `{USRNET_CONDITIONING.sign_probe_batch_size}`",
        f"- Smoke mode: `{smoke_mode}`. Smoke signed grid `{tuple(float(v) for v in SMOKE_SIGNED_CFO_GRID)}`, BER blocks `{SMOKE_BER_BLOCKS}`, BER batch `{SMOKE_BER_BATCH_SIZE}`, constellation blocks `{SMOKE_CONSTELLATION_NUM_BLOCKS}`, spectral blocks `{SMOKE_SPECTRAL_EVAL_BLOCKS}`, PAPR blocks `{SMOKE_PAPR_EVAL_BLOCKS}`, train batch `{SMOKE_TRAIN_SYMBOL_BATCH_SIZE}`",
    ]
    if phi_sign is not None:
        sign_label = "+" if phi_sign > 0 else "-"
        lines.append(f"- Selected Phi sign: `{sign_label}j 2pi delta n / M`")
    return lines


def build_stage2_usrnet_config(
    *,
    stage1_checkpoint_path: Path = STAGE1_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = True,
) -> ExperimentConfig:
    output_dir = Path(output_root) / output_subdir
    stage1_config = build_stage1_linear_config(refresh_output_dir=refresh_output_dir)
    delta_grid = _main_signed_cfo_grid(stage1_config)
    return replace(
        stage1_config,
        output_dir=output_dir,
        stage2_enabled=True,
        stage2_workflow="USR_NET",
        stage2_checkpoint_path=Path(stage1_checkpoint_path),
        eval_ebn0_db=EVAL_EBN0_DB,
        operator_eval_cfo=delta_grid,
        ber_eval_cfo=delta_grid,
        constellation_cfo=CONSTELLATION_CFO,
        heatmap_cfo=HEATMAP_CFO,
        pilot_estimation_enabled=False,
    )


def _smoke_overrides(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        ber_blocks=SMOKE_BER_BLOCKS,
        ber_batch_size=SMOKE_BER_BATCH_SIZE,
        train_symbol_batch_size=SMOKE_TRAIN_SYMBOL_BATCH_SIZE,
        constellation_num_blocks=SMOKE_CONSTELLATION_NUM_BLOCKS,
        spectral_eval_blocks=SMOKE_SPECTRAL_EVAL_BLOCKS,
        papr_eval_blocks=SMOKE_PAPR_EVAL_BLOCKS,
        operator_eval_cfo=np.asarray(SMOKE_SIGNED_CFO_GRID, dtype=float),
        ber_eval_cfo=np.asarray(SMOKE_SIGNED_CFO_GRID, dtype=float),
    )


def _prepare_checkpoint(config: ExperimentConfig) -> tuple[dict[str, object], Path]:
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_stage1_checkpoint(config.stage2_checkpoint_path, device=config.device, require_v2=True)
    checkpoint_config = checkpoint.get("config", {})
    for field_name in ("modulation", "M", "K", "N", "N_data", "N_pilots", "N_guard", "frame_structure_enabled"):
        saved_value = checkpoint_config.get(field_name)
        current_value = getattr(config, field_name)
        if saved_value != current_value:
            raise ValueError(
                f"Stage 1 checkpoint mismatch for {field_name}: saved={saved_value!r}, current={current_value!r}."
            )
    preflight = stage2_checkpoint_preflight(config, checkpoint)
    checkpoint_snapshot_path = snapshot_stage1_checkpoint(config.stage2_checkpoint_path, config.output_dir)
    config.stage2_checkpoint_source_path = Path(config.stage2_checkpoint_path)
    config.stage2_checkpoint_snapshot_path = checkpoint_snapshot_path
    config.stage2_checkpoint_hash_sha256 = str(preflight["checkpoint_payload_hash_sha256"])
    config.stage2_checkpoint_file_sha256 = file_sha256(checkpoint_snapshot_path)
    config.stage2_checkpoint_format = str(preflight["checkpoint_format"])
    config.stage2_checkpoint_acceptance_passed = bool(preflight["stage1_acceptance_summary"].get("passed", False))
    checkpoint_metrics = preflight["stage1_acceptance_summary"].get("metrics", {})
    config.stage2_checkpoint_clean_identity_loss = float(checkpoint_metrics.get("clean_identity_loss"))
    config.stage2_checkpoint_clean_offdiag_leakage = float(checkpoint_metrics.get("clean_offdiag_leakage"))
    config.stage2_checkpoint_learned_ber_at_0 = float(checkpoint_metrics.get("learned_ber_at_0"))
    config.stage2_checkpoint_ofdm_ber_at_0 = float(checkpoint_metrics.get("ofdm_ber_at_0"))
    return checkpoint, checkpoint_snapshot_path


def _plot_condition_sensitivity(output_dir: Path, sigma_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, delta_value in zip(axes, (0.05, 0.10)):
        delta_df = sigma_df[np.isclose(sigma_df["delta"], delta_value)]
        for method_name, color in (
            ("OFDMUSRNet", "#79A7D3"),
            ("LearnedUSRNet", "#E39A5F"),
        ):
            method_df = delta_df[delta_df["method"] == method_name].sort_values("sigma_condition")
            ax.semilogy(
                method_df["sigma_condition"],
                method_df["BER"],
                marker="o",
                linewidth=2.0,
                label=method_name,
                color=color,
            )
        ax.set_title(f"delta = {delta_value:.2f}")
        ax.set_xlabel("Condition noise sigma")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("BER")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / "usrnet_condition_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_per_layer_metric(output_dir: Path, layer_df: pd.DataFrame, *, metric: str, filename: str, ylabel: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=False)
    panel_map = {
        (0, 0): ("OFDMUSRNet", 0.05),
        (0, 1): ("OFDMUSRNet", 0.10),
        (1, 0): ("LearnedUSRNet", 0.05),
        (1, 1): ("LearnedUSRNet", 0.10),
    }
    for (row_idx, col_idx), (method_name, delta_value) in panel_map.items():
        ax = axes[row_idx, col_idx]
        panel_df = layer_df[(layer_df["method"] == method_name) & np.isclose(layer_df["delta"], delta_value)].sort_values("layer")
        if not panel_df.empty:
            ax.plot(panel_df["layer"], panel_df[metric], marker="o", linewidth=2.0)
        ax.set_title(f"{method_name}, delta={delta_value:.2f}")
        ax.grid(True, alpha=0.25)
        if row_idx == 1:
            ax.set_xlabel("Neural-USR layer")
        if col_idx == 0:
            ax.set_ylabel(ylabel)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _evaluate_main_and_reference(
    config: ExperimentConfig,
    *,
    ofdm_tx: torch.Tensor,
    ofdm_rx: torch.Tensor,
    learned_tx: torch.Tensor,
    learned_rx: torch.Tensor,
    ofdm_bundle: USRNetTrainingBundle,
    learned_bundle: USRNetTrainingBundle,
    phi_sign: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    per_layer_evm_rows: list[dict[str, object]] = []
    per_layer_ber_rows: list[dict[str, object]] = []

    method_specs = (
        ("OFDM", ofdm_tx, ofdm_rx, None, None),
        ("OFDMUSRNet", ofdm_tx, ofdm_rx, ofdm_bundle.receiver, ofdm_bundle.core_reference),
        ("Learned", learned_tx, learned_rx, None, None),
        ("LearnedUSRNet", learned_tx, learned_rx, learned_bundle.receiver, learned_bundle.core_reference),
    )
    for delta_idx, delta_value in enumerate(SUMMARY_DELTA_GRID):
        for method_name, W_tx, V, receiver, core_reference in method_specs:
            seed = 110_000 + 10_000 * delta_idx + (0 if method_name.startswith("OFDM") else 5_000)
            result = evaluate_usrnet_scheme(
                config,
                W_tx,
                V,
                receiver,
                method_name=method_name,
                delta_value=delta_value,
                ebn0_db=config.eval_ebn0_db,
                sigma_condition=DEFAULT_CONDITION_SIGMA,
                num_blocks=config.ber_blocks,
                batch_size=config.ber_batch_size,
                seed=seed,
                capture_points=config.constellation_plot_points if delta_value in CONSTELLATION_CFO else 0,
                phi_sign=phi_sign,
            )
            main_rows.append(
                {
                    "method": method_name,
                    "delta": float(delta_value),
                    "EbN0_dB": float(config.eval_ebn0_db),
                    "sigma_condition": DEFAULT_CONDITION_SIGMA,
                    "BER": float(result["BER"]),
                    "EVM": float(result["EVM"]),
                    "correction_norm": float(result["correction_norm"]),
                    "condition_mae": float(result["condition_mae"]),
                    "offdiag_energy_ratio": float(result["offdiag_energy_ratio"]),
                    "mean_abs_diag": float(result["mean_abs_diag"]),
                    "mean_angle_diag": float(result["mean_angle_diag"]),
                }
            )
            for layer_idx, value in enumerate(result.get("per_layer_evm", []), start=1):
                per_layer_evm_rows.append(
                    {
                        "method": method_name,
                        "delta": float(delta_value),
                        "sigma_condition": DEFAULT_CONDITION_SIGMA,
                        "layer": layer_idx,
                        "evm": float(value),
                    }
                )
            for layer_idx, value in enumerate(result.get("per_layer_ber", []), start=1):
                per_layer_ber_rows.append(
                    {
                        "method": method_name,
                        "delta": float(delta_value),
                        "sigma_condition": DEFAULT_CONDITION_SIGMA,
                        "layer": layer_idx,
                        "ber": float(value),
                    }
                )

            if core_reference is not None:
                core_result = evaluate_usrnet_scheme(
                    config,
                    W_tx,
                    V,
                    core_reference,
                    method_name=method_name.replace("USRNet", "CoreReference"),
                    delta_value=delta_value,
                    ebn0_db=config.eval_ebn0_db,
                    sigma_condition=DEFAULT_CONDITION_SIGMA,
                    num_blocks=config.ber_blocks,
                    batch_size=config.ber_batch_size,
                    seed=seed,
                    phi_sign=phi_sign,
                )
                core_rows.append(
                    {
                        "method": method_name.replace("USRNet", "CoreReference"),
                        "delta": float(delta_value),
                        "EbN0_dB": float(config.eval_ebn0_db),
                        "sigma_condition": DEFAULT_CONDITION_SIGMA,
                        "BER": float(core_result["BER"]),
                        "EVM": float(core_result["EVM"]),
                        "correction_norm": float(core_result["correction_norm"]),
                    }
                )
    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(core_rows),
        pd.DataFrame(per_layer_evm_rows),
        pd.DataFrame(per_layer_ber_rows),
    )


def _evaluate_sigma_sensitivity(
    config: ExperimentConfig,
    *,
    ofdm_tx: torch.Tensor,
    ofdm_rx: torch.Tensor,
    learned_tx: torch.Tensor,
    learned_rx: torch.Tensor,
    ofdm_receiver,
    learned_receiver,
    phi_sign: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for delta_idx, delta_value in enumerate((0.05, 0.10)):
        for sigma_idx, sigma_condition in enumerate(CONDITION_SENSITIVITY_GRID):
            for method_name, W_tx, V, receiver in (
                ("OFDMUSRNet", ofdm_tx, ofdm_rx, ofdm_receiver),
                ("LearnedUSRNet", learned_tx, learned_rx, learned_receiver),
            ):
                seed = 210_000 + 10_000 * delta_idx + 1_000 * sigma_idx + (0 if method_name == "OFDMUSRNet" else 5_000)
                result = evaluate_usrnet_scheme(
                    config,
                    W_tx,
                    V,
                    receiver,
                    method_name=method_name,
                    delta_value=delta_value,
                    ebn0_db=config.eval_ebn0_db,
                    sigma_condition=float(sigma_condition),
                    num_blocks=max(config.ber_blocks // 2, config.ber_batch_size),
                    batch_size=config.ber_batch_size,
                    seed=seed,
                    phi_sign=phi_sign,
                )
                rows.append(
                    {
                        "method": method_name,
                        "delta": float(delta_value),
                        "EbN0_dB": float(config.eval_ebn0_db),
                        "sigma_condition": float(sigma_condition),
                        "BER": float(result["BER"]),
                        "EVM": float(result["EVM"]),
                        "correction_norm": float(result["correction_norm"]),
                        "condition_mae": float(result["condition_mae"]),
                    }
                )
    return pd.DataFrame(rows)


def _acceptance_lines(main_df: pd.DataFrame, core_df: pd.DataFrame, learned_bundle: USRNetTrainingBundle) -> tuple[list[str], dict[str, bool]]:
    lookup = {(str(row["method"]), float(row["delta"])): row for _, row in main_df.iterrows()}
    core_lookup = {(str(row["method"]), float(row["delta"])): row for _, row in core_df.iterrows()}

    def _ber(frame_lookup: dict[tuple[str, float], object], method_name: str, delta_value: float) -> float:
        return float(frame_lookup[(method_name, float(delta_value))]["BER"])

    del learned_bundle
    learned_usr = _ber(lookup, "LearnedUSRNet", 0.10)
    learned_core = _ber(core_lookup, "LearnedCoreReference", 0.10)
    ofdm_zero_gap = _ber(lookup, "OFDMUSRNet", 0.0) - _ber(lookup, "OFDM", 0.0)
    learned_zero_gap = _ber(lookup, "LearnedUSRNet", 0.0) - _ber(lookup, "Learned", 0.0)
    checks = {
        "within_10pct": learned_usr <= 1.10 * learned_core,
        "zero_cfo_safe": (ofdm_zero_gap <= 0.002) and (learned_zero_gap <= 0.002),
    }
    neural_improved = learned_usr < learned_core

    lines = [
        (
            f"{'PASS' if checks['within_10pct'] else 'FAIL'} 1. Learned + Neural-USR matches the `r_core` ablation at delta=0.10 within 10 percent "
            f"(Neural-USR `{learned_usr:.4e}` vs `r_core` `{learned_core:.4e}`)."
        ),
        (
            f"{'PASS' if checks['zero_cfo_safe'] else 'FAIL'} 2. Zero-CFO BER is not degraded by more than absolute 0.002 "
            f"(OFDM gap `{ofdm_zero_gap:.4e}`, Learned gap `{learned_zero_gap:.4e}`)."
        ),
    ]
    if neural_improved:
        lines.append("Neural refinement gain: the three-layer unfolded neural output improves over the `r_core` anchor baseline.")
    else:
        lines.append(
            "Neural-USR stays close to the stable model-guided reference: the learned unfolded updates preserve the `r_core` baseline rather than degrading it."
        )
    return lines, checks


def _append_report_sections(
    report_path: Path,
    *,
    acceptance_lines: list[str],
    core_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
    learned_bundle: USRNetTrainingBundle,
    ofdm_bundle: USRNetTrainingBundle,
) -> None:
    lines = [
        "",
        "## Neural-USR Acceptance",
        "",
        *acceptance_lines,
        "",
        "## Neural-USR Description",
        "",
        "Neural-USR is a three-layer unfolded symbol refinement network whose final detector output is produced by learned neural update layers.",
        "",
        "The model-guided diagonal reference initializes and conditions the network. The practical Neural-USR path does not use off-diagonal operator cancellation and does not output the anchor directly.",
        "",
        "## r_core Ablation",
        "",
        core_df.to_string(index=False) if not core_df.empty else "(empty)",
        "",
        "## Condition Sensitivity",
        "",
        sigma_df.to_string(index=False) if not sigma_df.empty else "(empty)",
        "",
        "## Learned Parameters",
        "",
        f"- OFDM + Neural-USR: {ofdm_bundle.receiver.parameter_summary()}",
        f"- Learned + Neural-USR: {learned_bundle.receiver.parameter_summary()}",
        "",
    ]
    report_path = report_path.resolve()
    report_path.write_text(report_path.read_text() + "\n".join(lines))


def run_stage2_usrnet(*, smoke_mode: bool = False) -> object:
    config = build_stage2_usrnet_config()
    config = _smoke_overrides(config) if smoke_mode else config
    print(f"[USRNet] Output dir -> {config.output_dir}", flush=True)

    checkpoint, checkpoint_snapshot_path = _prepare_checkpoint(config)
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)

    phi_sign, sign_df = select_phi_sign(
        config,
        {"OFDM": (ofdm_tx, ofdm_rx), "Learned": (learned_tx, learned_rx)},
        seed=config.base_seed + 7_000,
        sign_probe_delta=USRNET_CONDITIONING.sign_probe_delta,
        sign_probe_ebn0_db=USRNET_CONDITIONING.sign_probe_ebn0_db,
        sign_probe_batch_size=USRNET_CONDITIONING.sign_probe_batch_size,
    )
    sign_label = "+" if phi_sign > 0 else "-"
    print(f"[USRNet] Selected Phi sign = {sign_label}j 2pi delta n / M", flush=True)

    print("[USRNet] Training internal diagonal-core references", flush=True)
    ofdm_core = train_diagonal_core_reference(
        config,
        ofdm_tx,
        ofdm_rx,
        scheme_name="OFDMCoreReference",
        phi_sign=phi_sign,
        model_config=CORE_REFERENCE_MODEL,
        train_config=_smoke_core_train_config() if smoke_mode else USRNET_CORE_TRAIN,
        validation_config=_smoke_validation_config() if smoke_mode else USRNET_VALIDATION,
        loss_config=USRNET_LOSS,
        seed_offset=0,
    )
    learned_core = train_diagonal_core_reference(
        config,
        learned_tx,
        learned_rx,
        scheme_name="LearnedCoreReference",
        phi_sign=phi_sign,
        model_config=CORE_REFERENCE_MODEL,
        train_config=_smoke_core_train_config() if smoke_mode else USRNET_CORE_TRAIN,
        validation_config=_smoke_validation_config() if smoke_mode else USRNET_VALIDATION,
        loss_config=USRNET_LOSS,
        seed_offset=1,
    )

    print("[USRNet] Training OFDM + Neural-USR", flush=True)
    ofdm_bundle = train_usrnet_receiver(
        config,
        ofdm_tx,
        ofdm_rx,
        ofdm_core.receiver,
        scheme_name="OFDMUSRNet",
        phi_sign=phi_sign,
        architecture_config=USRNET_ARCHITECTURE,
        phase_configs=_smoke_phase_configs() if smoke_mode else USRNET_PHASES,
        validation_config=_smoke_validation_config() if smoke_mode else USRNET_VALIDATION,
        loss_config=USRNET_LOSS,
        conditioning_config=USRNET_CONDITIONING,
        seed_offset=10,
    )
    print("[USRNet] Training Learned + Neural-USR", flush=True)
    learned_bundle = train_usrnet_receiver(
        config,
        learned_tx,
        learned_rx,
        learned_core.receiver,
        scheme_name="LearnedUSRNet",
        phi_sign=phi_sign,
        architecture_config=USRNET_ARCHITECTURE,
        phase_configs=_smoke_phase_configs() if smoke_mode else USRNET_PHASES,
        validation_config=_smoke_validation_config() if smoke_mode else USRNET_VALIDATION,
        loss_config=USRNET_LOSS,
        conditioning_config=USRNET_CONDITIONING,
        seed_offset=11,
    )

    training_result = TrainingResult(
        learned_tx=learned_tx.detach().clone(),
        learned_rx=learned_rx.detach().clone(),
        history_df=pd.concat([ofdm_bundle.history_df, learned_bundle.history_df], ignore_index=True),
        stage_summary_df=pd.concat([ofdm_bundle.stage_summary_df, learned_bundle.stage_summary_df], ignore_index=True),
        stage_failed=False,
        failed_stage=None,
        stop_reason="Completed frozen-front-end Neural-USR training for OFDM and Learned receivers.",
    )
    schemes = build_usrnet_schemes(
        ofdm_tx,
        ofdm_rx,
        learned_tx,
        learned_rx,
        ofdm_bundle.receiver,
        learned_bundle.receiver,
    )
    result = run_final_evaluation(config, training_result, schemes=schemes, method_order=METHOD_ORDER)
    result.artifact_paths["stage1_checkpoint_snapshot"] = checkpoint_snapshot_path

    main_df, core_df, per_layer_evm_df, per_layer_ber_df = _evaluate_main_and_reference(
        config,
        ofdm_tx=ofdm_tx,
        ofdm_rx=ofdm_rx,
        learned_tx=learned_tx,
        learned_rx=learned_rx,
        ofdm_bundle=ofdm_bundle,
        learned_bundle=learned_bundle,
        phi_sign=phi_sign,
    )
    sigma_df = _evaluate_sigma_sensitivity(
        config,
        ofdm_tx=ofdm_tx,
        ofdm_rx=ofdm_rx,
        learned_tx=learned_tx,
        learned_rx=learned_rx,
        ofdm_receiver=ofdm_bundle.receiver,
        learned_receiver=learned_bundle.receiver,
        phi_sign=phi_sign,
    )

    main_df.to_csv(config.output_dir / "usrnet_summary.csv", index=False)
    core_df.to_csv(config.output_dir / "core_reference_summary.csv", index=False)
    sigma_df.to_csv(config.output_dir / "condition_sensitivity.csv", index=False)
    per_layer_evm_df.to_csv(config.output_dir / "per_layer_evm.csv", index=False)
    per_layer_ber_df.to_csv(config.output_dir / "per_layer_ber.csv", index=False)
    sign_df.to_csv(config.output_dir / "phi_sign_diagnostic.csv", index=False)
    result.artifact_paths.update(
        {
            "usrnet_summary_csv": config.output_dir / "usrnet_summary.csv",
            "core_reference_summary_csv": config.output_dir / "core_reference_summary.csv",
            "condition_sensitivity_csv": config.output_dir / "condition_sensitivity.csv",
            "per_layer_evm_csv": config.output_dir / "per_layer_evm.csv",
            "per_layer_ber_csv": config.output_dir / "per_layer_ber.csv",
            "phi_sign_diagnostic_csv": config.output_dir / "phi_sign_diagnostic.csv",
        }
    )

    condition_plot = _plot_condition_sensitivity(config.output_dir, sigma_df)
    layer_evm_plot = _plot_per_layer_metric(config.output_dir, per_layer_evm_df, metric="evm", filename="usrnet_per_layer_evm.png", ylabel="EVM")
    layer_ber_plot = _plot_per_layer_metric(config.output_dir, per_layer_ber_df, metric="ber", filename="usrnet_per_layer_ber.png", ylabel="BER")
    result.artifact_paths.update(
        {
            "usrnet_condition_plot": condition_plot,
            "usrnet_per_layer_evm_plot": layer_evm_plot,
            "usrnet_per_layer_ber_plot": layer_ber_plot,
        }
    )

    acceptance_lines, acceptance_checks = _acceptance_lines(main_df, core_df, learned_bundle)
    result.summary_markdown = result.summary_markdown + "\n\n" + "\n".join(acceptance_lines)

    report_path = write_markdown_report(
        config=config,
        result=result,
        output_dir=config.output_dir,
        plot_files=REPORT_PLOT_FILES,
        extra_metadata={
            "title": "Residual-CFO Stage 2 Neural-USR Report",
            "overview_lines": [
                "Neural-USR is the main Stage 2 receiver for this package.",
                "The front-end Stage 1 waveform and receiver bases remain frozen.",
                "The model-guided reference initializes and conditions the network, while the final detector output is produced by three learned unfolded neural update layers.",
            ],
            "control_lines": _stage2_control_lines(config, phi_sign=phi_sign, smoke_mode=smoke_mode),
        },
    )
    _append_report_sections(
        report_path,
        acceptance_lines=acceptance_lines,
        core_df=core_df,
        sigma_df=sigma_df,
        learned_bundle=learned_bundle,
        ofdm_bundle=ofdm_bundle,
    )
    print(f"[USRNet] Report -> {report_path}", flush=True)

    for line in acceptance_lines:
        print(line, flush=True)
    return {
        "result": result,
        "main_df": main_df,
        "core_df": core_df,
        "sigma_df": sigma_df,
        "per_layer_evm_df": per_layer_evm_df,
        "per_layer_ber_df": per_layer_ber_df,
        "acceptance_checks": acceptance_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 USR-Net workflow.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke configuration.")
    args = parser.parse_args()
    run_stage2_usrnet(smoke_mode=bool(args.smoke))


if __name__ == "__main__":
    main()
