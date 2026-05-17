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
    DEFAULT_DELTA_GRID,
    DEFAULT_SIGMA_CONDITION,
    SIGMA_CONDITION_SENSITIVITY,
    USRNetTrainingBundle,
    build_usrnet_schemes,
    evaluate_usrnet_scheme,
    select_phi_sign,
    train_diagonal_core_reference,
    train_usrnet_receiver,
)
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "stage2_usrnet_outputs"
OUTPUT_SUBDIR = "usrnet_n45_r9"
STAGE1_OUTPUT_DIR = SCRIPT_DIR / "stage1_linear_outputs" / "16qam_n45_r9"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"

METHOD_ORDER = ("OFDM", "OFDMUSRNet", "Learned", "LearnedUSRNet")
CUSTOM_PLOT_FILES = (
    "usrnet_condition_sensitivity.png",
    "usrnet_per_layer_evm.png",
    "usrnet_per_layer_ber.png",
)
REPORT_PLOT_FILES = tuple(STAGE1_PLOT_FILES) + ("training_diagnostics.png", "training_loss_components.png") + CUSTOM_PLOT_FILES


def build_stage2_usrnet_config(
    *,
    stage1_checkpoint_path: Path = STAGE1_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = True,
) -> ExperimentConfig:
    output_dir = Path(output_root) / output_subdir
    delta_grid = np.asarray(DEFAULT_DELTA_GRID, dtype=float)
    return replace(
        build_stage1_linear_config(refresh_output_dir=refresh_output_dir),
        output_dir=output_dir,
        stage2_enabled=True,
        stage2_workflow="USR_NET",
        stage2_checkpoint_path=Path(stage1_checkpoint_path),
        eval_ebn0_db=12.0,
        operator_eval_cfo=delta_grid,
        ber_eval_cfo=delta_grid,
        constellation_cfo=(0.0, 0.05, 0.10),
        heatmap_cfo=(0.05, 0.10),
        pilot_estimation_enabled=False,
    )


def _smoke_overrides(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        ber_blocks=2048,
        ber_batch_size=256,
        constellation_num_blocks=128,
        spectral_eval_blocks=1024,
        papr_eval_blocks=1024,
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
            ax.set_xlabel("USR-Net layer")
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
    for delta_idx, delta_value in enumerate(DEFAULT_DELTA_GRID):
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
                sigma_condition=DEFAULT_SIGMA_CONDITION,
                num_blocks=config.ber_blocks,
                batch_size=config.ber_batch_size,
                seed=seed,
                capture_points=config.constellation_plot_points if delta_value in (0.0, 0.05, 0.10) else 0,
                phi_sign=phi_sign,
            )
            main_rows.append(
                {
                    "method": method_name,
                    "delta": float(delta_value),
                    "EbN0_dB": float(config.eval_ebn0_db),
                    "sigma_condition": DEFAULT_SIGMA_CONDITION,
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
                        "sigma_condition": DEFAULT_SIGMA_CONDITION,
                        "layer": layer_idx,
                        "evm": float(value),
                    }
                )
            for layer_idx, value in enumerate(result.get("per_layer_ber", []), start=1):
                per_layer_ber_rows.append(
                    {
                        "method": method_name,
                        "delta": float(delta_value),
                        "sigma_condition": DEFAULT_SIGMA_CONDITION,
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
                    sigma_condition=DEFAULT_SIGMA_CONDITION,
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
                        "sigma_condition": DEFAULT_SIGMA_CONDITION,
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
        for sigma_idx, sigma_condition in enumerate(SIGMA_CONDITION_SENSITIVITY):
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

    learned_usr = _ber(lookup, "LearnedUSRNet", 0.10)
    learned_core = _ber(core_lookup, "LearnedCoreReference", 0.10)
    checks = {
        "1": learned_usr <= 1.10 * learned_core,
        "2": _ber(lookup, "LearnedUSRNet", 0.10) < _ber(lookup, "OFDMUSRNet", 0.10),
        "3": _ber(lookup, "LearnedUSRNet", 0.05) <= 1.02 * _ber(lookup, "OFDMUSRNet", 0.05),
        "4": (_ber(lookup, "OFDMUSRNet", 0.0) - _ber(lookup, "OFDM", 0.0) <= 0.002)
        and (_ber(lookup, "LearnedUSRNet", 0.0) - _ber(lookup, "Learned", 0.0) <= 0.002),
    }
    alpha_values = learned_bundle.receiver.parameter_summary()["alpha_t"]
    neural_improved = learned_usr < learned_core
    checks["5"] = neural_improved or max(alpha_values) <= 0.08

    lines = [
        (
            f"{'PASS' if checks['1'] else 'FAIL'} 1. Learned + USR-Net matches or improves the internal diagonal-core reference at delta=0.10 within 10 percent relative "
            f"(USR `{learned_usr:.4e}` vs core `{learned_core:.4e}`)."
        ),
        (
            f"{'PASS' if checks['2'] else 'FAIL'} 2. Learned + USR-Net beats OFDM + USR-Net at delta=0.10 "
            f"(BER `{_ber(lookup, 'LearnedUSRNet', 0.10):.4e}` vs `{_ber(lookup, 'OFDMUSRNet', 0.10):.4e}`)."
        ),
        (
            f"{'PASS' if checks['3'] else 'FAIL'} 3. Learned + USR-Net beats OFDM + USR-Net at delta=0.05 or is very close "
            f"(BER `{_ber(lookup, 'LearnedUSRNet', 0.05):.4e}` vs `{_ber(lookup, 'OFDMUSRNet', 0.05):.4e}`)."
        ),
        (
            f"{'PASS' if checks['4'] else 'FAIL'} 4. At delta=0, USR-Net does not worsen BER by more than absolute 0.002 "
            f"(OFDM gap `{(_ber(lookup, 'OFDMUSRNet', 0.0) - _ber(lookup, 'OFDM', 0.0)):.4e}`, "
            f"Learned gap `{(_ber(lookup, 'LearnedUSRNet', 0.0) - _ber(lookup, 'Learned', 0.0)):.4e}`)."
        ),
        (
            f"{'PASS' if checks['5'] else 'FAIL'} 5. If the neural residual branch does not improve over the diagonal core, alpha stays small and the stable correction is retained "
            f"(alpha_t={alpha_values})."
        ),
    ]
    if not neural_improved:
        lines.append(
            "USR-Net selected the stable model-guided correction: the learned residual branch stayed modest relative to the diagonal-core anchor."
        )
    return lines, checks


def _append_report_sections(
    output_dir: Path,
    *,
    acceptance_lines: list[str],
    core_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
    learned_bundle: USRNetTrainingBundle,
    ofdm_bundle: USRNetTrainingBundle,
) -> None:
    lines = [
        "",
        "## USR-Net Acceptance",
        "",
        *acceptance_lines,
        "",
        "## USR-Net Description",
        "",
        "USR-Net is a three-layer unfolded symbol refinement network with model-guided correction anchors and learned residual dilated convolutional refinement blocks.",
        "",
        "The receiver-state condition is used only to build the diagonal anchor from the fixed effective operator. The practical USR-Net path does not use off-diagonal operator cancellation.",
        "",
        "## Internal Diagonal-Core Reference",
        "",
        core_df.to_string(index=False) if not core_df.empty else "(empty)",
        "",
        "## Condition Sensitivity",
        "",
        sigma_df.to_string(index=False) if not sigma_df.empty else "(empty)",
        "",
        "## Learned Parameters",
        "",
        f"- OFDM + USR-Net: {ofdm_bundle.receiver.parameter_summary()}",
        f"- Learned + USR-Net: {learned_bundle.receiver.parameter_summary()}",
        "",
    ]
    for report_name in ("report.md", "report.MD"):
        report_path = output_dir / report_name
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
        smoke_mode=smoke_mode,
        seed_offset=0,
    )
    learned_core = train_diagonal_core_reference(
        config,
        learned_tx,
        learned_rx,
        scheme_name="LearnedCoreReference",
        phi_sign=phi_sign,
        smoke_mode=smoke_mode,
        seed_offset=1,
    )

    print("[USRNet] Training OFDM + USR-Net", flush=True)
    ofdm_bundle = train_usrnet_receiver(
        config,
        ofdm_tx,
        ofdm_rx,
        ofdm_core.receiver,
        scheme_name="OFDMUSRNet",
        phi_sign=phi_sign,
        smoke_mode=smoke_mode,
        seed_offset=10,
    )
    print("[USRNet] Training Learned + USR-Net", flush=True)
    learned_bundle = train_usrnet_receiver(
        config,
        learned_tx,
        learned_rx,
        learned_core.receiver,
        scheme_name="LearnedUSRNet",
        phi_sign=phi_sign,
        smoke_mode=smoke_mode,
        seed_offset=11,
    )

    training_result = TrainingResult(
        learned_tx=learned_tx.detach().clone(),
        learned_rx=learned_rx.detach().clone(),
        history_df=pd.concat([ofdm_bundle.history_df, learned_bundle.history_df], ignore_index=True),
        stage_summary_df=pd.concat([ofdm_bundle.stage_summary_df, learned_bundle.stage_summary_df], ignore_index=True),
        stage_failed=False,
        failed_stage=None,
        stop_reason="Completed frozen-front-end USR-Net training for OFDM and Learned receivers.",
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
            "title": "Residual-CFO Stage 2 USR-Net Report",
            "overview_lines": [
                "USR-Net is the main Stage 2 receiver for this package.",
                "The front-end Stage 1 waveform and receiver bases remain frozen.",
                "The practical receiver is a three-layer unfolded symbol refinement network anchored by diagonal model-guided corrections and small residual dilated-convolution updates.",
            ],
            "control_lines": [
                f"- Output dir: `{config.output_dir}`",
                f"- Stage 1 checkpoint: `{config.stage2_checkpoint_path}`",
                f"- Evaluation grid: `{tuple(float(v) for v in config.ber_eval_cfo)}` at `{config.eval_ebn0_db:.1f} dB`",
                f"- Default receiver-state condition noise: `{DEFAULT_SIGMA_CONDITION:.4f}`",
                f"- Condition sensitivity grid: `{SIGMA_CONDITION_SENSITIVITY}`",
                f"- Report methods: `{', '.join(METHOD_ORDER)}`",
                f"- Selected Phi sign: `{sign_label}`",
            ],
        },
    )
    _append_report_sections(
        config.output_dir,
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
