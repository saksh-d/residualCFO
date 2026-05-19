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
from reporting import average_metrics_by_abs_cfo, run_final_evaluation, write_markdown_report
from run_stage1_linear import PLOT_FILES as STAGE1_PLOT_FILES, build_stage1_linear_config
from stage2_grrnet_receiver import (
    DEFAULT_CONDITION_SENSITIVITY,
    DEFAULT_CONDITION_SIGMA,
    DEFAULT_PHASE2_SIGMAS,
    DEFAULT_SNR_GRID,
    DEFAULT_SUMMARY_DELTA_GRID,
    GRRNetArchitectureConfig,
    GRRNetLossConfig,
    GRRNetPhaseConfig,
    GRRNetTrainingBundle,
    GRRNetValidationConfig,
    build_grrnet_schemes,
    evaluate_grrnet_scheme,
    run_grrnet_sanity_check,
    train_grrnet_receiver,
)
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = SCRIPT_DIR / "stage2_grrnet_outputs"
OUTPUT_SUBDIR = "grrnet_n45_r9"
STAGE1_OUTPUT_DIR = SCRIPT_DIR / "stage1_linear_outputs" / "16qam_n45_r9"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"
USRNET_OUTPUT_DIR = SCRIPT_DIR / "stage2_usrnet_outputs" / "usrnet_n45_r9"

SUMMARY_DELTA_GRID = DEFAULT_SUMMARY_DELTA_GRID
SIGNED_CFO_GRID = np.asarray(sorted({-float(value) for value in SUMMARY_DELTA_GRID} | {float(value) for value in SUMMARY_DELTA_GRID}), dtype=float)
CONSTELLATION_CFO = (0.0, 0.05, 0.10)
HEATMAP_CFO = (0.05, 0.10)
EVAL_EBN0_DB = 10.0

METHOD_ORDER = ("OFDM", "OFDMGRRNet", "Learned", "LearnedGRRNet")
CONSTELLATION_METHOD_ORDER = ("OFDM", "Learned", "OFDMGRRNet", "LearnedGRRNet")
COMPARE_METHOD_ORDER = ("OFDM", "OFDMUSRNet", "OFDMGRRNet", "Learned", "LearnedUSRNet", "LearnedGRRNet")

GRRNET_LOSS = GRRNetLossConfig()
GRRNET_PHASES = (
    GRRNetPhaseConfig(
        name="Phase1HighSNRCleanup",
        epochs=100,
        learning_rate=1.0e-3,
        batch_size=512,
        delta_span=0.12,
        ebn0_choices=(20.0,),
        sigma_condition=DEFAULT_CONDITION_SIGMA,
    ),
    GRRNetPhaseConfig(
        name="Phase2RealisticTraining",
        epochs=200,
        learning_rate=5.0e-4,
        batch_size=512,
        delta_span=0.15,
        ebn0_choices=(10.0, 12.0, 15.0, 20.0),
        sigma_condition=DEFAULT_PHASE2_SIGMAS,
    ),
)
GRRNET_VALIDATION = GRRNetValidationConfig(
    delta_values=(0.0, 0.05, 0.10),
    ebn0_db=10.0,
    sigma_condition=DEFAULT_CONDITION_SIGMA,
    num_blocks=2048,
    batch_size=256,
)
SANITY_EPOCHS = 200

SMOKE_SANITY_EPOCHS = 5
SMOKE_PHASE_EPOCHS = 5
SMOKE_BER_BLOCKS = 1024
SMOKE_BER_BATCH_SIZE = 256
SMOKE_CONSTELLATION_NUM_BLOCKS = 96
SMOKE_SPECTRAL_EVAL_BLOCKS = 512
SMOKE_PAPR_EVAL_BLOCKS = 512
SMOKE_TRAIN_SYMBOL_BATCH_SIZE = 256
SMOKE_VALIDATION = GRRNetValidationConfig(
    delta_values=(0.0, 0.05, 0.10),
    ebn0_db=10.0,
    sigma_condition=DEFAULT_CONDITION_SIGMA,
    num_blocks=512,
    batch_size=128,
)

CUSTOM_PLOT_FILES = (
    "grrnet_condition_sensitivity.png",
    "training_diagnostics.png",
    "training_loss_components.png",
    "correction_norm_vs_cfo.png",
    "branch_norms.png",
    "ber_vs_cfo_with_usrnet.png",
)
REPORT_PLOT_FILES = tuple(STAGE1_PLOT_FILES) + CUSTOM_PLOT_FILES


def _parse_bool_flag(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _smoke_phase_configs() -> tuple[GRRNetPhaseConfig, ...]:
    return tuple(replace(phase, epochs=SMOKE_PHASE_EPOCHS, batch_size=SMOKE_TRAIN_SYMBOL_BATCH_SIZE) for phase in GRRNET_PHASES)


def _stage2_control_lines(config: ExperimentConfig, *, smoke_mode: bool = False) -> list[str]:
    phase_configs = _smoke_phase_configs() if smoke_mode else GRRNET_PHASES
    validation = SMOKE_VALIDATION if smoke_mode else GRRNET_VALIDATION
    phase_text = "; ".join(
        f"{phase.name}: epochs={phase.epochs}, lr={phase.learning_rate:.1e}, batch={phase.batch_size}, "
        f"delta_span={phase.delta_span:.2f}, ebn0={phase.ebn0_choices}, sigma={phase.sigma_condition}"
        for phase in phase_configs
    )
    return [
        f"- Output dir: `{config.output_dir}`",
        f"- Stage 1 source: output dir `{STAGE1_OUTPUT_DIR}`, checkpoint `{config.stage2_checkpoint_path}`",
        f"- Evaluation setup: eval Eb/N0 `{config.eval_ebn0_db:.1f} dB`, signed CFO grid `{tuple(float(v) for v in config.ber_eval_cfo)}`, constellations `{config.constellation_cfo}`",
        f"- GRR-Net mode: conditioned `{getattr(config, 'grrnet_conditioned', True)}`, rank `{getattr(config, 'grrnet_rank', 16)}`, default condition sigma `{DEFAULT_CONDITION_SIGMA:.4f}`",
        f"- Architecture: local channels `32`, global rank `{getattr(config, 'grrnet_rank', 16)}`, Conv1d stack `32->64->64->64->2`, bounded alpha `0.2 * sigmoid(raw_alpha)`",
        f"- Loss: symbol-MSE `1.0`, CE `{GRRNET_LOSS.cls_weight}`, identity `{GRRNET_LOSS.identity_weight}`, correction `{GRRNET_LOSS.correction_weight}`, cls temp `{GRRNET_LOSS.temp_cls}`",
        f"- Sanity phase: epochs `{SMOKE_SANITY_EPOCHS if smoke_mode else SANITY_EPOCHS}`, fixed delta `0.10`, Eb/N0 `20 dB`, sigma `{DEFAULT_CONDITION_SIGMA if getattr(config, 'grrnet_conditioned', True) else 0.0}`",
        f"- Train phases: {phase_text}",
        f"- Validation model-selection: deltas `{validation.delta_values}`, Eb/N0 `{validation.ebn0_db:.1f} dB`, sigma `{validation.sigma_condition:.4f}`, blocks `{validation.num_blocks}`, batch `{validation.batch_size}`",
        f"- Condition sensitivity grid: `{DEFAULT_CONDITION_SENSITIVITY}`",
        f"- Smoke mode: `{smoke_mode}`. BER blocks `{config.ber_blocks}`, BER batch `{config.ber_batch_size}`, train batch `{config.train_symbol_batch_size}`, constellation blocks `{config.constellation_num_blocks}`",
    ]


def build_stage2_grrnet_config(
    *,
    seed: int = 9,
    conditioned: bool = True,
    rank: int = 4,
    stage1_checkpoint_path: Path = STAGE1_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir_flag: bool = True,
) -> ExperimentConfig:
    output_dir = Path(output_root) / output_subdir
    stage1_config = build_stage1_linear_config(refresh_output_dir=refresh_output_dir_flag)
    config = replace(
        stage1_config,
        base_seed=int(seed),
        output_dir=output_dir,
        stage2_enabled=True,
        stage2_workflow="GRR_NET",
        stage2_checkpoint_path=Path(stage1_checkpoint_path),
        eval_ebn0_db=EVAL_EBN0_DB,
        operator_eval_cfo=SIGNED_CFO_GRID,
        ber_eval_cfo=SIGNED_CFO_GRID,
        snr_sweep_ebn0_db_grid=np.asarray(DEFAULT_SNR_GRID, dtype=float),
        snr_sweep_cfo_values=(0.0, 0.05, 0.10),
        constellation_cfo=CONSTELLATION_CFO,
        heatmap_cfo=HEATMAP_CFO,
        pilot_estimation_enabled=False,
    )
    config.grrnet_conditioned = bool(conditioned)
    config.grrnet_rank = int(rank)
    return config


def _smoke_overrides(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        ber_blocks=SMOKE_BER_BLOCKS,
        ber_batch_size=SMOKE_BER_BATCH_SIZE,
        train_symbol_batch_size=SMOKE_TRAIN_SYMBOL_BATCH_SIZE,
        constellation_num_blocks=SMOKE_CONSTELLATION_NUM_BLOCKS,
        spectral_eval_blocks=SMOKE_SPECTRAL_EVAL_BLOCKS,
        papr_eval_blocks=SMOKE_PAPR_EVAL_BLOCKS,
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
            raise ValueError(f"Stage 1 checkpoint mismatch for {field_name}: saved={saved_value!r}, current={current_value!r}.")
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


def _evaluate_main(
    config: ExperimentConfig,
    *,
    ofdm_tx: torch.Tensor,
    ofdm_rx: torch.Tensor,
    learned_tx: torch.Tensor,
    learned_rx: torch.Tensor,
    ofdm_bundle: GRRNetTrainingBundle,
    learned_bundle: GRRNetTrainingBundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_specs = (
        ("OFDM", ofdm_tx, ofdm_rx, None),
        ("OFDMGRRNet", ofdm_tx, ofdm_rx, ofdm_bundle.receiver),
        ("Learned", learned_tx, learned_rx, None),
        ("LearnedGRRNet", learned_tx, learned_rx, learned_bundle.receiver),
    )
    for delta_idx, delta_value in enumerate(SUMMARY_DELTA_GRID):
        for method_name, W_tx, V, receiver in method_specs:
            seed = 310_000 + 10_000 * delta_idx + (0 if method_name.startswith("OFDM") else 5_000)
            result = evaluate_grrnet_scheme(
                config,
                W_tx,
                V,
                receiver,
                method_name=method_name,
                delta_value=float(delta_value),
                ebn0_db=float(config.eval_ebn0_db),
                sigma_condition=DEFAULT_CONDITION_SIGMA,
                num_blocks=int(config.ber_blocks),
                batch_size=int(config.ber_batch_size),
                seed=seed,
                capture_points=config.constellation_plot_points if float(delta_value) in CONSTELLATION_CFO else 0,
            )
            rows.append(
                {
                    "method": method_name,
                    "delta": float(delta_value),
                    "EbN0_dB": float(config.eval_ebn0_db),
                    "sigma_condition": DEFAULT_CONDITION_SIGMA,
                    "BER": float(result["BER"]),
                    "EVM": float(result["EVM"]),
                    "correction_norm": float(result["correction_norm"]),
                    "condition_mae": float(result["condition_mae"]),
                    "alpha": float(result["alpha"]),
                    "local_norm": float(result["local_norm"]),
                    "global_norm": float(result["global_norm"]),
                    "combined_norm": float(result["combined_norm"]),
                    "conv_norm": float(result["conv_norm"]),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_condition_sensitivity(
    config: ExperimentConfig,
    *,
    ofdm_tx: torch.Tensor,
    ofdm_rx: torch.Tensor,
    learned_tx: torch.Tensor,
    learned_rx: torch.Tensor,
    ofdm_receiver,
    learned_receiver,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for delta_idx, delta_value in enumerate((0.05, 0.10)):
        for sigma_idx, sigma_condition in enumerate(DEFAULT_CONDITION_SENSITIVITY):
            for method_name, W_tx, V, receiver in (
                ("OFDMGRRNet", ofdm_tx, ofdm_rx, ofdm_receiver),
                ("LearnedGRRNet", learned_tx, learned_rx, learned_receiver),
            ):
                seed = 410_000 + 10_000 * delta_idx + 1_000 * sigma_idx + (0 if method_name == "OFDMGRRNet" else 5_000)
                result = evaluate_grrnet_scheme(
                    config,
                    W_tx,
                    V,
                    receiver,
                    method_name=method_name,
                    delta_value=float(delta_value),
                    ebn0_db=float(config.eval_ebn0_db),
                    sigma_condition=float(sigma_condition),
                    num_blocks=max(config.ber_blocks // 2, config.ber_batch_size),
                    batch_size=int(config.ber_batch_size),
                    seed=seed,
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
                        "alpha": float(result["alpha"]),
                    }
                )
    return pd.DataFrame(rows)


def _plot_condition_sensitivity(output_dir: Path, sigma_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, delta_value in zip(axes, (0.05, 0.10)):
        delta_df = sigma_df[np.isclose(sigma_df["delta"], delta_value)]
        for method_name, color in (
            ("OFDMGRRNet", "#4F81BD"),
            ("LearnedGRRNet", "#D07A3A"),
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
    path = output_dir / "grrnet_condition_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_correction_norm(output_dir: Path, main_df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(8.0, 4.8), dpi=150)
    for method_name, color in (("OFDMGRRNet", "#4F81BD"), ("LearnedGRRNet", "#D07A3A")):
        method_df = main_df[main_df["method"] == method_name].sort_values("delta")
        plt.plot(method_df["delta"], method_df["correction_norm"], marker="o", linewidth=2.0, label=method_name, color=color)
    plt.xlabel("Absolute residual CFO")
    plt.ylabel("Correction norm")
    plt.title("GRR-Net correction norm vs residual CFO")
    plt.grid(True, alpha=0.25)
    plt.legend()
    path = output_dir / "correction_norm_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_branch_norms(output_dir: Path, branch_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=150, sharey=False)
    for ax, method_name in zip(axes, ("OFDMGRRNet", "LearnedGRRNet")):
        method_df = branch_df[branch_df["method"] == method_name].sort_values("delta")
        ax.plot(method_df["delta"], method_df["local_norm"], marker="o", linewidth=2.0, label="Local")
        ax.plot(method_df["delta"], method_df["global_norm"], marker="s", linewidth=2.0, label="Global")
        ax.plot(method_df["delta"], method_df["combined_norm"], marker="^", linewidth=2.0, label="Combined")
        ax.plot(method_df["delta"], method_df["conv_norm"], marker="d", linewidth=2.0, label="Conv residual")
        ax.set_title(method_name)
        ax.set_xlabel("Absolute residual CFO")
        ax.set_ylabel("Branch norm")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path = output_dir / "branch_norms.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_usrnet_artifacts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    summary_path = USRNET_OUTPUT_DIR / "summary_metrics.csv"
    ber_path = USRNET_OUTPUT_DIR / "ber_vs_cfo.csv"
    if not summary_path.exists() or not ber_path.exists():
        return None, None
    return pd.read_csv(summary_path), pd.read_csv(ber_path)


def _build_usrnet_comparison(main_df: pd.DataFrame, usr_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for baseline_method, grr_method, usr_method in (
        ("OFDM", "OFDMGRRNet", "OFDMUSRNet"),
        ("Learned", "LearnedGRRNet", "LearnedUSRNet"),
    ):
        for delta_value, column_name in ((0.0, "ber_at_0"), (0.05, "ber_at_0p05"), (0.10, "ber_at_0p10")):
            grr_row = main_df[(main_df["method"] == grr_method) & np.isclose(main_df["delta"], delta_value)].iloc[0]
            baseline_row = main_df[(main_df["method"] == baseline_method) & np.isclose(main_df["delta"], delta_value)].iloc[0]
            usr_row = usr_summary_df[usr_summary_df["method"] == usr_method].iloc[0]
            rows.append(
                {
                    "family": baseline_method,
                    "delta": float(delta_value),
                    "baseline_method": baseline_method,
                    "grr_method": grr_method,
                    "usr_method": usr_method,
                    "baseline_ber": float(baseline_row["BER"]),
                    "grr_ber": float(grr_row["BER"]),
                    "usr_ber": float(usr_row[column_name]),
                    "grr_minus_usr": float(grr_row["BER"] - usr_row[column_name]),
                }
            )
    return pd.DataFrame(rows)


def _plot_ber_vs_cfo_with_usrnet(output_dir: Path, grr_ber_df: pd.DataFrame, usr_ber_df: pd.DataFrame) -> Path:
    plot_df = pd.concat(
        [
            average_metrics_by_abs_cfo(grr_ber_df[grr_ber_df["method"].isin({"OFDM", "OFDMGRRNet", "Learned", "LearnedGRRNet"})]),
            average_metrics_by_abs_cfo(usr_ber_df[usr_ber_df["method"].isin({"OFDMUSRNet", "LearnedUSRNet"})]),
        ],
        ignore_index=True,
    )
    fig = plt.figure(figsize=(8.2, 4.8), dpi=150)
    color_map = {
        "OFDM": "#3A5F8A",
        "OFDMUSRNet": "#79A7D3",
        "OFDMGRRNet": "#4F81BD",
        "Learned": "#C05A2B",
        "LearnedUSRNet": "#E39A5F",
        "LearnedGRRNet": "#D07A3A",
    }
    for method_name in ("OFDM", "OFDMUSRNet", "OFDMGRRNet", "Learned", "LearnedUSRNet", "LearnedGRRNet"):
        method_df = plot_df[plot_df["method"] == method_name].sort_values("eps")
        if method_df.empty:
            continue
        plt.semilogy(
            method_df["eps"],
            method_df["ber"],
            marker="o",
            linewidth=2.0,
            label=method_name,
            color=color_map[method_name],
        )
    plt.xlabel("Absolute residual CFO")
    plt.ylabel("BER")
    plt.title("BER vs residual CFO with USR-Net comparison")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(ncol=2)
    path = output_dir / "ber_vs_cfo_with_usrnet.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _acceptance_lines(
    main_df: pd.DataFrame,
    *,
    conditioned: bool,
    usr_summary_df: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, bool]]:
    lookup = {(str(row["method"]), float(row["delta"])): row for _, row in main_df.iterrows()}

    def _ber(method_name: str, delta_value: float) -> float:
        return float(lookup[(method_name, float(delta_value))]["BER"])

    def _correction(method_name: str, delta_value: float) -> float:
        return float(lookup[(method_name, float(delta_value))]["correction_norm"])

    def _alpha(method_name: str, delta_value: float) -> float:
        return float(lookup[(method_name, float(delta_value))]["alpha"])

    checks = {
        "1": _ber("LearnedGRRNet", 0.10) < _ber("Learned", 0.10),
        "2": _ber("LearnedGRRNet", 0.05) < _ber("Learned", 0.05),
        "3": _ber("LearnedGRRNet", 0.10) < _ber("OFDMGRRNet", 0.10),
        "4": _ber("LearnedGRRNet", 0.05) <= 1.05 * _ber("OFDMGRRNet", 0.05),
        "5": (_ber("OFDMGRRNet", 0.0) - _ber("OFDM", 0.0) <= 0.002) and (_ber("LearnedGRRNet", 0.0) - _ber("Learned", 0.0) <= 0.002),
        "6": _correction("LearnedGRRNet", 0.10) > 1.0e-4,
        "7": False,
    }
    checks["7"] = (
        (_alpha("LearnedGRRNet", 0.10) < 0.19)
        or (
            _ber("LearnedGRRNet", 0.10) < _ber("Learned", 0.10)
            and (_ber("LearnedGRRNet", 0.0) - _ber("Learned", 0.0) <= 0.002)
        )
    )

    lines = [
        f"{'PASS' if checks['1'] else 'FAIL'} 1. Learned + GRR-Net improves over Learned at delta=0.10 (BER `{_ber('LearnedGRRNet', 0.10):.4e}` vs `{_ber('Learned', 0.10):.4e}`).",
        f"{'PASS' if checks['2'] else 'FAIL'} 2. Learned + GRR-Net improves over Learned at delta=0.05 (BER `{_ber('LearnedGRRNet', 0.05):.4e}` vs `{_ber('Learned', 0.05):.4e}`).",
        f"{'PASS' if checks['3'] else 'FAIL'} 3. Learned + GRR-Net beats OFDM + GRR-Net at delta=0.10 (BER `{_ber('LearnedGRRNet', 0.10):.4e}` vs `{_ber('OFDMGRRNet', 0.10):.4e}`).",
        f"{'PASS' if checks['4'] else 'FAIL'} 4. Learned + GRR-Net beats OFDM + GRR-Net at delta=0.05 or is close (BER `{_ber('LearnedGRRNet', 0.05):.4e}` vs `{_ber('OFDMGRRNet', 0.05):.4e}`).",
        f"{'PASS' if checks['5'] else 'FAIL'} 5. At delta=0, GRR-Net does not worsen BER by more than absolute 0.002 (OFDM gap `{(_ber('OFDMGRRNet', 0.0) - _ber('OFDM', 0.0)):.4e}`, Learned gap `{(_ber('LearnedGRRNet', 0.0) - _ber('Learned', 0.0)):.4e}`).",
        f"{'PASS' if checks['6'] else 'FAIL'} 6. Correction norm is nonzero at delta=0.10 (Learned + GRR-Net correction norm `{_correction('LearnedGRRNet', 0.10):.4e}`).",
        f"{'PASS' if checks['7'] else 'FAIL'} 7. Alpha is not saturated at maximum unless performance improves and zero-CFO remains stable (Learned + GRR-Net alpha `{_alpha('LearnedGRRNet', 0.10):.4f}`).",
    ]

    if usr_summary_df is not None:
        learned_usr = usr_summary_df[usr_summary_df["method"] == "LearnedUSRNet"].iloc[0]
        ofdm_usr = usr_summary_df[usr_summary_df["method"] == "OFDMUSRNet"].iloc[0]
        learned_grr = _ber("LearnedGRRNet", 0.10)
        ofdm_gain = _ber("OFDM", 0.10) - _ber("OFDMGRRNet", 0.10)
        learned_gain = _ber("Learned", 0.10) - _ber("LearnedGRRNet", 0.10)
        if learned_grr < float(learned_usr["ber_at_0p10"]):
            lines.append("BLACK-BOX GAIN: GRR-Net improves over model-guided USR-Net on the learned basis.")
        elif learned_grr <= 1.05 * float(learned_usr["ber_at_0p10"]):
            lines.append("MATCHED USR-NET: Black-box global mixing matches the model-guided unfolded receiver.")
        if learned_gain > 0.0 and ofdm_gain > 1.5 * learned_gain:
            lines.append("DIAGNOSTIC: Stronger black-box receiver recovers additional OFDM leakage; learned basis had already suppressed much of it.")
        if (_ber("OFDMGRRNet", 0.10) <= 1.05 * _ber("LearnedGRRNet", 0.10)) or (_ber("OFDMGRRNet", 0.05) <= 1.05 * _ber("LearnedGRRNet", 0.05)):
            lines.append("UPPER-BOUND BEHAVIOR: High-capacity black-box Stage 2 may obscure the Stage 1 waveform advantage.")
        if conditioned and np.isfinite(float(ofdm_usr["ber_at_0p10"])):
            lines.append(
                f"USR-Net comparison at delta=0.10: OFDM + USR-Net `{float(ofdm_usr['ber_at_0p10']):.4e}`, "
                f"Learned + USR-Net `{float(learned_usr['ber_at_0p10']):.4e}`."
            )
    return lines, checks


def _append_report_sections(
    output_dir: Path,
    *,
    acceptance_lines: list[str],
    main_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
    architecture_config: GRRNetArchitectureConfig,
    stage_summary_df: pd.DataFrame,
    sanity_summary: dict[str, float | bool],
    usr_comparison_df: pd.DataFrame | None,
) -> None:
    table_df = main_df[main_df["delta"].isin({0.0, 0.05, 0.10})].copy()
    lines = [
        "",
        "## GRR-Net Acceptance",
        "",
        *acceptance_lines,
        "",
        "## GRR-Net Architecture Summary",
        "",
        "GRR-Net is a black-box post-V global residual refiner.",
        "",
        f"- Conditioned: `{architecture_config.conditioned}`",
        f"- Rank: `{architecture_config.rank}`",
        "- Input features: `Re(z0), Im(z0), |z0|, cos(angle(z0)), sin(angle(z0)), Re(q), Im(q), Re(e), Im(e), |e|^2`",
        "- Optional condition features: `c_b, c_b^2, sin(pi c_b), cos(pi c_b)` appended per symbol",
        "- Mixer: local projection plus low-rank all-to-all symbol mixing, followed by bounded residual dilated Conv1D refinement",
        "",
        "## Training Phases",
        "",
        f"- Phase 0 sanity overfit passed: `{sanity_summary['passed']}`. Baseline EVM `{float(sanity_summary['baseline_evm']):.4e}`, final EVM `{float(sanity_summary['final_evm']):.4e}`, best EVM `{float(sanity_summary['best_evm']):.4e}`.",
        "",
    ]
    for _, row in stage_summary_df.iterrows():
        lines.append(
            f"- {row['scheme']} {row['stage']}: best epoch `{int(row.get('best_stage_epoch', row['best_global_epoch']))}`, "
            f"val BER `{row['val_ber']:.4e}`, val EVM `{row['val_evm']:.4e}`, hard-CFO BER `{row['hard_cfo_weighted_ber']:.4e}`, alpha `{row.get('alpha', float('nan')):.4f}`."
        )
    lines.extend(
        [
            "",
            "## BER Summary Table",
            "",
            table_df.to_string(index=False) if not table_df.empty else "(empty)",
            "",
            "## Condition Sensitivity Table",
            "",
            sigma_df.to_string(index=False) if not sigma_df.empty else "(empty)",
            "",
        ]
    )
    if usr_comparison_df is not None:
        lines.extend(
            [
                "## Comparison Against USR-Net",
                "",
                usr_comparison_df.to_string(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "GRR-Net is a diagnostic black-box post-V receiver. If it improves strongly over USR-Net, that points to recoverable nonlocal residual structure beyond the current model-guided path. If it matches USR-Net, the model-guided receiver is already competitive while remaining more interpretable.",
            "",
        ]
    )
    for report_name in ("report.md", "report.MD"):
        report_path = output_dir / report_name
        report_path.write_text(report_path.read_text() + "\n".join(lines))


def run_stage2_grrnet(
    *,
    smoke_mode: bool = False,
    conditioned: bool = True,
    rank: int = 16,
    compare_usrnet: bool = False,
    seed: int = 9,
    output_dir: Path | None = None,
) -> dict[str, object]:
    config = build_stage2_grrnet_config(
        seed=seed,
        conditioned=conditioned,
        rank=rank,
        output_root=(output_dir.parent if output_dir is not None else OUTPUT_ROOT),
        output_subdir=(output_dir.name if output_dir is not None else OUTPUT_SUBDIR),
    )
    config = _smoke_overrides(config) if smoke_mode else config
    config.grrnet_compare_usrnet = bool(compare_usrnet)
    print(f"[GRRNet] Output dir -> {config.output_dir}", flush=True)

    checkpoint, checkpoint_snapshot_path = _prepare_checkpoint(config)
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)

    architecture_config = GRRNetArchitectureConfig(
        conditioned=bool(conditioned),
        rank=int(rank),
        num_symbols=int(config.N),
        eval_condition_sigma=DEFAULT_CONDITION_SIGMA,
    )
    phase_configs = _smoke_phase_configs() if smoke_mode else GRRNET_PHASES
    validation_config = SMOKE_VALIDATION if smoke_mode else GRRNET_VALIDATION

    print("[GRRNet] Running one-batch sanity overfit on Learned", flush=True)
    sanity_history_df, sanity_stage_df, sanity_summary = run_grrnet_sanity_check(
        config,
        learned_tx,
        learned_rx,
        architecture_config=architecture_config,
        loss_config=GRRNET_LOSS,
        epochs=(SMOKE_SANITY_EPOCHS if smoke_mode else SANITY_EPOCHS),
        batch_size=512 if not smoke_mode else 256,
        seed_offset=90,
    )
    print(
        f"[GRRNet] Sanity {'PASS' if sanity_summary['passed'] else 'FAIL'} | "
        f"baseline EVM={float(sanity_summary['baseline_evm']):.4e} final EVM={float(sanity_summary['final_evm']):.4e}",
        flush=True,
    )

    print("[GRRNet] Training OFDM + GRR-Net", flush=True)
    ofdm_bundle = train_grrnet_receiver(
        config,
        ofdm_tx,
        ofdm_rx,
        scheme_name="OFDMGRRNet",
        architecture_config=architecture_config,
        phase_configs=phase_configs,
        validation_config=validation_config,
        loss_config=GRRNET_LOSS,
        seed_offset=100,
    )
    print("[GRRNet] Training Learned + GRR-Net", flush=True)
    learned_bundle = train_grrnet_receiver(
        config,
        learned_tx,
        learned_rx,
        scheme_name="LearnedGRRNet",
        architecture_config=architecture_config,
        phase_configs=phase_configs,
        validation_config=validation_config,
        loss_config=GRRNET_LOSS,
        seed_offset=101,
    )

    ofdm_ckpt_path = config.output_dir / "ofdm_grrnet_checkpoint.pt"
    learned_ckpt_path = config.output_dir / "learned_grrnet_checkpoint.pt"
    torch.save({"state_dict": ofdm_bundle.receiver.state_dict(), "parameter_summary": ofdm_bundle.receiver.parameter_summary()}, ofdm_ckpt_path)
    torch.save({"state_dict": learned_bundle.receiver.state_dict(), "parameter_summary": learned_bundle.receiver.parameter_summary()}, learned_ckpt_path)

    history_df = pd.concat([sanity_history_df, ofdm_bundle.history_df, learned_bundle.history_df], ignore_index=True)
    stage_summary_df = pd.concat([sanity_stage_df, ofdm_bundle.stage_summary_df, learned_bundle.stage_summary_df], ignore_index=True)
    training_result = TrainingResult(
        learned_tx=learned_tx.detach().clone(),
        learned_rx=learned_rx.detach().clone(),
        history_df=history_df,
        stage_summary_df=stage_summary_df,
        stage_failed=False,
        failed_stage=None,
        stop_reason="Completed frozen-front-end GRR-Net training for OFDM and Learned receivers.",
    )
    schemes = build_grrnet_schemes(
        ofdm_tx,
        ofdm_rx,
        learned_tx,
        learned_rx,
        ofdm_bundle.receiver,
        learned_bundle.receiver,
    )
    result = run_final_evaluation(config, training_result, schemes=schemes, method_order=METHOD_ORDER)
    result.artifact_paths["stage1_checkpoint_snapshot"] = checkpoint_snapshot_path
    result.artifact_paths["ofdm_grrnet_checkpoint"] = ofdm_ckpt_path
    result.artifact_paths["learned_grrnet_checkpoint"] = learned_ckpt_path

    main_df = _evaluate_main(
        config,
        ofdm_tx=ofdm_tx,
        ofdm_rx=ofdm_rx,
        learned_tx=learned_tx,
        learned_rx=learned_rx,
        ofdm_bundle=ofdm_bundle,
        learned_bundle=learned_bundle,
    )
    sigma_df = _evaluate_condition_sensitivity(
        config,
        ofdm_tx=ofdm_tx,
        ofdm_rx=ofdm_rx,
        learned_tx=learned_tx,
        learned_rx=learned_rx,
        ofdm_receiver=ofdm_bundle.receiver,
        learned_receiver=learned_bundle.receiver,
    )

    grr_summary_path = config.output_dir / "grrnet_summary.csv"
    condition_path = config.output_dir / "condition_sensitivity.csv"
    correction_path = config.output_dir / "correction_norm.csv"
    alpha_path = config.output_dir / "alpha_summary.csv"
    branch_path = config.output_dir / "branch_norms.csv"
    sanity_path = config.output_dir / "sanity_overfit_summary.csv"
    main_df.to_csv(grr_summary_path, index=False)
    sigma_df.to_csv(condition_path, index=False)
    main_df[["method", "delta", "correction_norm"]].to_csv(correction_path, index=False)
    main_df[main_df["method"].isin({"OFDMGRRNet", "LearnedGRRNet"})][["method", "delta", "alpha"]].to_csv(alpha_path, index=False)
    main_df[["method", "delta", "local_norm", "global_norm", "combined_norm", "conv_norm"]].to_csv(branch_path, index=False)
    pd.DataFrame([sanity_summary]).to_csv(sanity_path, index=False)
    result.artifact_paths.update(
        {
            "grrnet_summary_csv": grr_summary_path,
            "condition_sensitivity_csv": condition_path,
            "correction_norm_csv": correction_path,
            "alpha_summary_csv": alpha_path,
            "branch_norms_csv": branch_path,
            "sanity_summary_csv": sanity_path,
        }
    )

    result.artifact_paths["grrnet_condition_plot"] = _plot_condition_sensitivity(config.output_dir, sigma_df)
    result.artifact_paths["correction_norm_plot"] = _plot_correction_norm(config.output_dir, main_df)
    result.artifact_paths["branch_norms_plot"] = _plot_branch_norms(config.output_dir, main_df)

    usr_summary_df = None
    usr_ber_df = None
    usr_comparison_df = None
    if compare_usrnet:
        usr_summary_df, usr_ber_df = _load_usrnet_artifacts()
        if usr_summary_df is None or usr_ber_df is None:
            print(f"[GRRNet] Warning: USR-Net artifacts not found under {USRNET_OUTPUT_DIR}; skipping comparison.", flush=True)
        else:
            usr_comparison_df = _build_usrnet_comparison(main_df, usr_summary_df)
            comparison_path = config.output_dir / "comparison_to_usrnet.csv"
            usr_comparison_df.to_csv(comparison_path, index=False)
            result.artifact_paths["usrnet_comparison_csv"] = comparison_path
            result.artifact_paths["usrnet_overlay_plot"] = _plot_ber_vs_cfo_with_usrnet(config.output_dir, result.ber_df, usr_ber_df)

    acceptance_lines, acceptance_checks = _acceptance_lines(
        main_df,
        conditioned=conditioned,
        usr_summary_df=usr_summary_df,
    )
    result.summary_markdown = result.summary_markdown + "\n\n" + "\n".join(acceptance_lines)

    report_path = write_markdown_report(
        config=config,
        result=result,
        output_dir=config.output_dir,
        plot_files=REPORT_PLOT_FILES,
        extra_metadata={
            "title": "Residual-CFO Stage 2 GRR-Net Report",
            "overview_lines": [
                "GRR-Net is a diagnostic black-box post-V global residual refiner.",
                "The Stage 1 learned waveform receiver remains frozen.",
                "The practical GRR-Net path operates only on z0 = V y plus optional raw receiver-state condition features.",
            ],
            "control_lines": _stage2_control_lines(config, smoke_mode=smoke_mode),
        },
    )
    _append_report_sections(
        config.output_dir,
        acceptance_lines=acceptance_lines,
        main_df=main_df,
        sigma_df=sigma_df,
        architecture_config=architecture_config,
        stage_summary_df=stage_summary_df,
        sanity_summary=sanity_summary,
        usr_comparison_df=usr_comparison_df,
    )
    print(f"[GRRNet] Report -> {report_path}", flush=True)

    print(
        f"[GRRNet] Smoke method check -> methods present {sorted(set(result.summary_df['method'].astype(str).tolist()))}",
        flush=True,
    )
    for line in acceptance_lines:
        print(line, flush=True)
    return {
        "result": result,
        "main_df": main_df,
        "sigma_df": sigma_df,
        "sanity_summary": sanity_summary,
        "acceptance_checks": acceptance_checks,
        "usr_comparison_df": usr_comparison_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 GRR-Net workflow.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke configuration.")
    parser.add_argument("--conditioned", default="true", help="Use condition features (true/false).")
    parser.add_argument("--rank", type=int, default=16, choices=(4, 8, 16, 32), help="Low-rank global mixer rank.")
    parser.add_argument("--compare_usrnet", action="store_true", help="Compare against saved USR-Net artifacts if available.")
    parser.add_argument("--seed", type=int, default=9, help="Base random seed.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT / OUTPUT_SUBDIR), help="Output directory.")
    args = parser.parse_args()
    run_stage2_grrnet(
        smoke_mode=bool(args.smoke),
        conditioned=_parse_bool_flag(args.conditioned),
        rank=int(args.rank),
        compare_usrnet=bool(args.compare_usrnet),
        seed=int(args.seed),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
