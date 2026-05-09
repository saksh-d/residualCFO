from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comm_core import (
    TrainingResult,
    default_config,
    ebn0_to_noise_variance,
    file_sha256,
    load_stage1_checkpoint,
    refresh_output_dir,
    run_stage2_experiment,
    set_seed,
    snapshot_stage1_checkpoint,
    stage2_checkpoint_preflight,
)
from receiver import EvaluationScheme, evaluate_scheme_set, train_stage2_nonlinear_receiver
from reporting import method_color, method_display_name, plot_ber_curve_multi, run_final_evaluation
from run_stage1_linear import build_stage1_linear_config
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "stage2_oracle_diagnostics"
OUTPUT_SUBDIR = "oracle_n44_r6"
REFRESH_OUTPUT_DIR = True

STAGE1_OUTPUT_DIR = ROOT_DIR / "stage1_linear_outputs" / "16qam_n44_r6"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"

NONLINEAR_ONLY_SUMMARY_PATH = ROOT_DIR / "stage2_nonlinear_outputs" / "nonlinear_only_n44_r6" / "summary_metrics.csv"

ORACLE_METHOD_ORDER = (
    "OFDM",
    "Learned",
    "LearnedOraclePreV",
    "LearnedOracleMMSE",
    "LearnedNonlinearOracleEps",
)
ALPHA_SCALES = (0.25, 1.0, 4.0)
KEY_EPS = (0.0, 0.05, 0.10)


def build_stage2_oracle_diagnostic_config(
    *,
    stage1_checkpoint_path: Path = STAGE1_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = REFRESH_OUTPUT_DIR,
) -> object:
    output_dir = Path(output_root) / output_subdir
    return replace(
        build_stage1_linear_config(refresh_output_dir=refresh_output_dir),
        output_dir=output_dir,
        stage2_enabled=True,
        stage2_workflow="NONLINEAR_ONLY",
        stage2_detector_arch="LOCAL",
        stage2_decision_loss="BIT_BCE",
        stage2_checkpoint_path=Path(stage1_checkpoint_path),
    )


def _prepare_stage2_checkpoint(config: object) -> tuple[dict[str, object], Path]:
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_source_path = Path(config.stage2_checkpoint_path)
    print(f"[Oracle] Loading Stage 1 checkpoint -> {checkpoint_source_path}", flush=True)
    checkpoint = load_stage1_checkpoint(checkpoint_source_path, device=config.device, require_v2=True)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config:
        for field_name in ("modulation", "M", "K", "N", "N_data", "N_pilots", "N_guard", "frame_structure_enabled"):
            saved_value = checkpoint_config.get(field_name)
            current_value = getattr(config, field_name)
            if saved_value != current_value:
                raise ValueError(
                    f"Stage 1 checkpoint mismatch for {field_name}: saved={saved_value!r}, current={current_value!r}."
                )

    print("[Oracle] Running checkpoint preflight", flush=True)
    preflight = stage2_checkpoint_preflight(config, checkpoint)
    checkpoint_snapshot_path = snapshot_stage1_checkpoint(checkpoint_source_path, config.output_dir)
    config.stage2_checkpoint_source_path = checkpoint_source_path
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


def _build_oracle_schemes(config: object, learned_tx, learned_rx, nonlinear_receiver) -> dict[str, EvaluationScheme]:
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)
    mmse_alpha = ebn0_to_noise_variance(config.eval_ebn0_db, config.bits_per_symbol)
    return {
        "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
        "Learned": EvaluationScheme(tx_basis=learned_tx, rx_basis=learned_rx, nonlinear_receiver=None),
        "LearnedOraclePreV": EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=None,
            oracle_pre_v_cfo=True,
        ),
        "LearnedOracleMMSE": EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=None,
            oracle_post_v_mmse=True,
            oracle_mmse_alpha=mmse_alpha,
        ),
        "LearnedNonlinearOracleEps": EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=nonlinear_receiver,
            oracle_eps_conditioning=True,
        ),
    }


def _build_oracle_alpha_sensitivity(config: object, learned_tx, learned_rx, baseline_ber_df: pd.DataFrame) -> pd.DataFrame:
    base_alpha = ebn0_to_noise_variance(config.eval_ebn0_db, config.bits_per_symbol)
    frames: list[pd.DataFrame] = []
    for scale_idx, scale in enumerate(ALPHA_SCALES):
        alpha = scale * base_alpha
        scheme = EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=None,
            oracle_post_v_mmse=True,
            oracle_mmse_alpha=alpha,
        )
        frame = evaluate_scheme_set(
            config=config,
            schemes={"LearnedOracleMMSE": scheme},
            cfo_points=np.array(KEY_EPS, dtype=float),
            ebn0_db=config.eval_ebn0_db,
            num_blocks=config.ber_blocks,
            batch_size=config.ber_batch_size,
            seed=88_000 + scale_idx,
            progress_label=f"Oracle MMSE alpha x{scale:g}",
        ).copy()
        frame["method_label"] = frame["method"].map(method_display_name)
        frame["alpha_scale"] = float(scale)
        frame["alpha"] = float(alpha)
        frames.append(frame)
    sensitivity_df = pd.concat(frames, ignore_index=True)
    learned_baseline = baseline_ber_df[baseline_ber_df["method"] == "Learned"][["eps", "ber"]].rename(
        columns={"ber": "learned_ber"}
    )
    sensitivity_df = sensitivity_df.merge(learned_baseline, on="eps", how="left")
    sensitivity_df["delta_ber_vs_learned"] = sensitivity_df["ber"] - sensitivity_df["learned_ber"]
    return sensitivity_df.sort_values(["alpha_scale", "eps"], kind="stable").reset_index(drop=True)


def _plot_oracle_gain(result, output_dir: Path) -> Path:
    ber_df = result.ber_df.copy()
    baseline = ber_df[ber_df["method"] == "Learned"][["eps", "ber"]].rename(columns={"ber": "learned_ber"})
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in ("LearnedOraclePreV", "LearnedOracleMMSE", "LearnedNonlinearOracleEps"):
        frame = ber_df[ber_df["method"] == method][["eps", "ber"]].merge(baseline, on="eps", how="left")
        if frame.empty:
            continue
        gain = frame["learned_ber"] - frame["ber"]
        plt.plot(
            frame["eps"],
            gain,
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    plt.title("Oracle BER gain vs frozen Learned baseline")
    plt.xlabel("Normalized residual CFO")
    plt.ylabel("BER improvement (positive is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = output_dir / "oracle_gain_vs_baseline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_oracle_constellations(result, output_dir: Path) -> Path:
    plot_df = result.constellation_df.copy()
    plot_df = plot_df[np.isclose(plot_df["eps"], 0.10)]
    methods = ["Learned", "LearnedOraclePreV", "LearnedOracleMMSE", "LearnedNonlinearOracleEps"]
    fig, axes = plt.subplots(1, len(methods), figsize=(15.2, 3.8), dpi=130, constrained_layout=True)
    for ax, method in zip(axes, methods):
        frame = plot_df[plot_df["method"] == method]
        if frame.empty:
            ax.set_visible(False)
            continue
        ax.scatter(frame["est_real"], frame["est_imag"], s=8, alpha=0.22, color=method_color(method), edgecolors="none")
        ax.scatter(frame["ref_real"], frame["ref_imag"], s=10, alpha=0.65, color="#111111", marker="x")
        ax.set_title(method_display_name(method))
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    path = output_dir / "oracle_constellation_snapshots.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_saved_stage2_row(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    summary_df = pd.read_csv(path)
    frame = summary_df[summary_df["method"] == "LearnedNonlinear"]
    if frame.empty:
        return None
    return frame.iloc[0]


def _nearest_row(summary_df: pd.DataFrame, method: str) -> pd.Series:
    frame = summary_df[summary_df["method"] == method]
    if frame.empty:
        raise ValueError(f"Missing summary row for method {method}.")
    return frame.iloc[0]


def _delta_line(method_row: pd.Series, learned_row: pd.Series, eps_label: str, column: str) -> str:
    delta = float(method_row[column]) - float(learned_row[column])
    return (
        f"- {method_display_name(str(method_row['method']))} BER({eps_label}): "
        f"`{float(method_row[column]):.6f}` vs learned `{float(learned_row[column]):.6f}` "
        f"(delta `{delta:+.6f}`)."
    )


def _diagnostic_takeaway(method_row: pd.Series, learned_row: pd.Series) -> str:
    improvements = {
        0.05: float(learned_row["ber_at_0p05"]) - float(method_row["ber_at_0p05"]),
        0.10: float(learned_row["ber_at_0p10"]) - float(method_row["ber_at_0p10"]),
    }
    zero_gain = float(learned_row["ber_at_0"]) - float(method_row["ber_at_0"])
    best_cfo = max(improvements, key=improvements.get)
    best_gain = improvements[best_cfo]
    if max(best_gain, zero_gain) <= 5e-4:
        return f"- {method_display_name(str(method_row['method']))}: no material BER gain over frozen learned."
    if best_gain > zero_gain + 5e-4:
        return (
            f"- {method_display_name(str(method_row['method']))}: gain is concentrated at moderate/high CFO, "
            f"largest around `delta={best_cfo:.2f}` with BER improvement `{best_gain:+.6f}`."
        )
    return (
        f"- {method_display_name(str(method_row['method']))}: gain is present but not especially concentrated away from zero CFO."
    )


def _write_oracle_report(
    config: object,
    result,
    sensitivity_df: pd.DataFrame,
    output_dir: Path,
    comparison_plot: Path,
    gain_plot: Path,
    constellation_plot: Path,
) -> Path:
    summary_df = result.summary_df.copy()
    learned_row = _nearest_row(summary_df, "Learned")
    pre_v_row = _nearest_row(summary_df, "LearnedOraclePreV")
    mmse_row = _nearest_row(summary_df, "LearnedOracleMMSE")
    oracle_eps_row = _nearest_row(summary_df, "LearnedNonlinearOracleEps")
    current_nonlinear_only = _load_saved_stage2_row(NONLINEAR_ONLY_SUMMARY_PATH)

    best_mmse = sensitivity_df.sort_values(["eps", "ber"], kind="stable").groupby("eps", as_index=False).first()
    best_mmse_lines = [
        f"- Best MMSE alpha at delta={row['eps']:.2f}: scale `{row['alpha_scale']:.2f}` with BER `{row['ber']:.6f}`."
        for _, row in best_mmse.iterrows()
    ]

    lines = [
        "# Stage 2 Oracle Diagnostics",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Stage 1 checkpoint: `{config.stage2_checkpoint_source_path}`",
        f"- Configuration: `{config.modulation}`, `M={config.M}`, `K={config.K}`, `N={config.N}`, `P={config.N_pilots}`, `G={config.N_guard}`.",
        f"- Train/Eval SNR: train `{config.train_ebn0_db:.1f} dB`, eval `{config.eval_ebn0_db:.1f} dB`.",
        "",
        "## Test A: Oracle CFO correction before V",
        _delta_line(pre_v_row, learned_row, "0.00", "ber_at_0"),
        _delta_line(pre_v_row, learned_row, "0.05", "ber_at_0p05"),
        _delta_line(pre_v_row, learned_row, "0.10", "ber_at_0p10"),
        _diagnostic_takeaway(pre_v_row, learned_row),
        "",
        "## Test B: Oracle delta-conditioned MMSE after V",
        _delta_line(mmse_row, learned_row, "0.00", "ber_at_0"),
        _delta_line(mmse_row, learned_row, "0.05", "ber_at_0p05"),
        _delta_line(mmse_row, learned_row, "0.10", "ber_at_0p10"),
        _diagnostic_takeaway(mmse_row, learned_row),
        *best_mmse_lines,
        "",
        "## Test C: Current Stage 2 detector with true delta conditioning",
        _delta_line(oracle_eps_row, learned_row, "0.00", "ber_at_0"),
        _delta_line(oracle_eps_row, learned_row, "0.05", "ber_at_0p05"),
        _delta_line(oracle_eps_row, learned_row, "0.10", "ber_at_0p10"),
        _diagnostic_takeaway(oracle_eps_row, learned_row),
    ]
    if current_nonlinear_only is not None:
        lines.append(
            f"- Versus saved `nonlinear_only`: BER(0.05) `{float(oracle_eps_row['ber_at_0p05']):.6f}` vs `{float(current_nonlinear_only['ber_at_0p05']):.6f}`, "
            f"BER(0.10) `{float(oracle_eps_row['ber_at_0p10']):.6f}` vs `{float(current_nonlinear_only['ber_at_0p10']):.6f}`."
        )

    lines.extend(
        [
            "",
            "## Files",
            f"- [summary_metrics.csv]({(output_dir / 'summary_metrics.csv').name})",
            f"- [ber_vs_cfo.csv]({(output_dir / 'ber_vs_cfo.csv').name})",
            f"- [ber_vs_snr_by_cfo.csv]({(output_dir / 'ber_vs_snr_by_cfo.csv').name})",
            f"- [oracle_alpha_sensitivity.csv]({(output_dir / 'oracle_alpha_sensitivity.csv').name})",
            f"- [stage_summary.csv]({(output_dir / 'stage_summary.csv').name})",
            f"- [stage_training_history.csv]({(output_dir / 'stage_training_history.csv').name})",
            "",
            f"![Oracle BER Comparison]({comparison_plot.name})",
            "",
            f"![Oracle Gain vs Baseline]({gain_plot.name})",
            "",
            f"![Oracle Constellation Snapshots]({constellation_plot.name})",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path


def run_stage2_oracle_diagnostics(config=None):
    config = build_stage2_oracle_diagnostic_config() if config is None else config
    if config.stage2_checkpoint_path is None or not Path(config.stage2_checkpoint_path).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {config.stage2_checkpoint_path}")

    print(f"Running Stage 2 oracle diagnostics -> {config.output_dir}", flush=True)
    set_seed(config.base_seed)
    checkpoint, checkpoint_snapshot_path = _prepare_stage2_checkpoint(config)
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)

    learned_stage2 = train_stage2_nonlinear_receiver(
        config=config,
        tx_basis=learned_tx,
        rx_basis=learned_rx,
        scheme_name="LearnedNonlinearOracleEps",
        seed_offset=0,
        oracle_eps_conditioning=True,
    )
    history_df = learned_stage2.history_df.sort_values(["scheme", "global_epoch"], kind="stable")
    stage_summary_df = learned_stage2.stage_summary_df.sort_values("scheme", kind="stable")
    training_result = TrainingResult(
        learned_tx=learned_stage2.learned_tx.detach().clone(),
        learned_rx=learned_stage2.learned_rx.detach().clone(),
        history_df=history_df.copy(),
        stage_summary_df=stage_summary_df.copy(),
        stage_failed=False,
        failed_stage=None,
        stop_reason="Completed Stage 2 oracle nonlinear receiver diagnostics.",
    )

    schemes = _build_oracle_schemes(config, learned_tx, learned_rx, learned_stage2.receiver.eval())
    result = run_final_evaluation(config, training_result, schemes=schemes)
    result.artifact_paths["stage1_checkpoint_snapshot"] = checkpoint_snapshot_path

    sensitivity_df = _build_oracle_alpha_sensitivity(config, learned_tx, learned_rx, result.ber_df)
    sensitivity_path = config.output_dir / "oracle_alpha_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    result.artifact_paths["oracle_alpha_sensitivity_csv"] = sensitivity_path

    comparison_plot = plot_ber_curve_multi(
        ber_df=result.ber_df,
        method_order=ORACLE_METHOD_ORDER,
        title=f"Oracle diagnosis BER vs residual CFO | {config.modulation}, N={config.N}, M={config.M}",
        path=config.output_dir / "oracle_ber_comparison.png",
    )
    if comparison_plot is not None:
        result.artifact_paths["oracle_ber_comparison_plot"] = comparison_plot
    gain_plot = _plot_oracle_gain(result, config.output_dir)
    result.artifact_paths["oracle_gain_plot"] = gain_plot
    constellation_plot = _plot_oracle_constellations(result, config.output_dir)
    result.artifact_paths["oracle_constellation_plot"] = constellation_plot
    report_path = _write_oracle_report(
        config=config,
        result=result,
        sensitivity_df=sensitivity_df,
        output_dir=config.output_dir,
        comparison_plot=comparison_plot if comparison_plot is not None else config.output_dir / "oracle_ber_comparison.png",
        gain_plot=gain_plot,
        constellation_plot=constellation_plot,
    )
    result.artifact_paths["report_md"] = report_path
    print(f"Wrote oracle report -> {report_path}", flush=True)
    return result


if __name__ == "__main__":
    run_stage2_oracle_diagnostics()
