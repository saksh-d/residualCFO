from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from comm_core import (
    TrainingResult,
    ebn0_to_noise_variance,
    file_sha256,
    load_stage1_checkpoint,
    refresh_output_dir,
    snapshot_stage1_checkpoint,
    stage2_checkpoint_preflight,
)
from receiver import EvaluationScheme, train_stage2_nonlinear_receiver
from reporting import run_final_evaluation, write_markdown_report
from run_stage1_linear import build_stage1_linear_config
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "stage2_blind_outputs"
OUTPUT_SUBDIR = "blind_mmse_n45_r9"
REFRESH_OUTPUT_DIR = True

STAGE1_OUTPUT_DIR = SCRIPT_DIR / "stage1_linear_outputs" / "16qam_n45_r9"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"

STAGE2_PLOT_FILES = (
    "ber_vs_cfo.png",
    "ber_vs_snr_by_cfo.png",
    "evm_vs_cfo.png",
    "offdiag_leakage_vs_cfo.png",
    "nearest_neighbor_leakage_vs_cfo.png",
    "operator_heatmaps.png",
    "constellation_snapshots.png",
    "spectral_fairness.png",
    "papr_ccdf.png",
    "frequency_domain_bases_all.png",
    "frequency_domain_bases_random.png",
    "time_domain_waveform.png",
    "time_domain_envelope_phase.png",
)

METHOD_ORDER = (
    "OFDM",
    "Learned",
    "LearnedNonlinear",
)


def build_stage2_blind_mmse_config(
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
        stage2_estimator_max_abs_cfo=0.20,
        stage2_cfo_loss_weight=0.10,
        stage2_sideinfo_enabled=True,
        stage2_sideinfo_mode="SCALED_TRUE",
        stage2_sideinfo_scale=0.25,
        stage2_sideinfo_residual_enabled=True,
    )


def _prepare_stage2_checkpoint(config: object) -> tuple[dict[str, object], Path]:
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_source_path = Path(config.stage2_checkpoint_path)
    print(f"[BlindStage2] Loading Stage 1 checkpoint -> {checkpoint_source_path}", flush=True)
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

    print("[BlindStage2] Running checkpoint preflight", flush=True)
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


def _build_schemes(config: object, learned_tx, learned_rx, nonlinear_receiver) -> dict[str, EvaluationScheme]:
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)
    return {
        "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
        "Learned": EvaluationScheme(tx_basis=learned_tx, rx_basis=learned_rx, nonlinear_receiver=None),
        "LearnedNonlinear": EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=nonlinear_receiver.eval(),
        ),
    }


def control_lines(config: object) -> list[str]:
    return [
        f"- Output dir: `{config.output_dir}`",
        f"- Stage 1 checkpoint: `{config.stage2_checkpoint_path}`",
        f"- Frame: `M={config.M}, K={config.K}, N={config.N}, P={config.N_pilots}, G={config.N_guard}, R={config.redundancy_dimensions}`",
        f"- Train SNR: `{config.train_ebn0_db:.1f} dB`, train range `[{config.train_ebn0_db_min:.1f}, {config.train_ebn0_db_max:.1f}] dB`, eval SNR `{config.eval_ebn0_db:.1f} dB`",
        f"- Stage 2 training: epochs `{config.stage2_nonlinear_only_epochs}`, lr `{config.stage2_nonlinear_only_learning_rate}`, frozen `W` and `V`",
        f"- Detector: `{config.stage2_detector_arch.lower()}` on `z0` with `[Re, Im, |z0|, cos(angle), sin(angle)]` features",
        f"- Conservative CFO side info: mode `{config.stage2_sideinfo_mode}`, coarse seed `hat(delta)_coarse = {config.stage2_sideinfo_scale:.2f} delta`, residual learner `{config.stage2_sideinfo_residual_enabled}`",
        f"- CFO span: training support `|delta| <= {config.stage_c_cfo:.2f}`, estimator output `|delta_hat| <= {config.stage2_estimator_max_abs_cfo:.2f}`, evaluation out to `{max(abs(v) for v in config.ber_eval_cfo):.2f}`",
        f"- Losses: decision `{config.stage2_decision_loss}`, symbol MSE weight `{config.stage2_loss_mse_weight}`, CFO aux weight `{config.stage2_cfo_loss_weight}`",
        f"- Report methods: `{', '.join(METHOD_ORDER)}`",
    ]


def run_stage2_blind_mmse(config: object | None = None) -> object:
    config = build_stage2_blind_mmse_config() if config is None else config
    if config.stage2_checkpoint_path is None or not Path(config.stage2_checkpoint_path).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {config.stage2_checkpoint_path}")

    checkpoint, checkpoint_snapshot_path = _prepare_stage2_checkpoint(config)
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)

    print(f"[BlindStage2] Training conservative coarse-CFO-assisted Stage 2 receiver -> {config.output_dir}", flush=True)
    learned_stage2 = train_stage2_nonlinear_receiver(
        config=config,
        tx_basis=learned_tx,
        rx_basis=learned_rx,
        scheme_name="LearnedNonlinear",
        seed_offset=1,
    )
    training_result = TrainingResult(
        learned_tx=learned_stage2.learned_tx.detach().clone(),
        learned_rx=learned_stage2.learned_rx.detach().clone(),
        history_df=learned_stage2.history_df.copy(),
        stage_summary_df=learned_stage2.stage_summary_df.copy(),
        stage_failed=False,
        failed_stage=None,
        stop_reason="Completed frozen post-V conservative coarse-CFO-assisted Stage 2 estimator/MMSE training for the learned front end.",
    )
    schemes = _build_schemes(config, learned_tx, learned_rx, learned_stage2.receiver)
    result = run_final_evaluation(
        config,
        training_result,
        schemes=schemes,
        method_order=METHOD_ORDER,
    )
    result.artifact_paths["stage1_checkpoint_snapshot"] = checkpoint_snapshot_path
    report_path = write_markdown_report(
        config=config,
        result=result,
        output_dir=config.output_dir,
        plot_files=STAGE2_PLOT_FILES,
        extra_metadata={
            "title": "Residual-CFO Stage 2 Conservative Coarse-CFO-Assisted MMSE Report",
            "overview_lines": [
                "Conservative Stage 2 package built on the frozen Stage 1 structured `16QAM` experiment.",
                "The receiver starts from a coarse residual-CFO seed `hat(delta)_coarse = 0.15 delta`, refines it from `z0`, constructs `A(hat(delta)) = V Phi_hat(delta) W`, and applies a regularized post-`V` MMSE solve.",
                "The comparison shown here is limited to Classical OFDM, the Learned Basis Stage 1 baseline, and the Learned Basis Stage 2 coarse-CFO-assisted estimation plus MMSE path.",
            ],
            "control_lines": control_lines(config),
        },
    )
    print(f"[BlindStage2] Wrote report -> {report_path}", flush=True)
    return result


if __name__ == "__main__":
    run_stage2_blind_mmse()
