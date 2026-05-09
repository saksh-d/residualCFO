from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from comm_core import ExperimentConfig, run_stage2_experiment
from reporting import build_frame_resource_df, write_markdown_report
from run_stage1_linear import PLOT_FILES, build_stage1_linear_config


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "stage2_nonlinear_outputs"
OUTPUT_SUBDIR = "joint_n44_r6"
REFRESH_OUTPUT_DIR = True

STAGE1_OUTPUT_DIR = SCRIPT_DIR / "stage1_linear_outputs" / "16qam_n44_r6"
STAGE1_CHECKPOINT_PATH = STAGE1_OUTPUT_DIR / "stage1_checkpoint.pt"


def build_stage2_joint_config(
    *,
    stage1_checkpoint_path: Path = STAGE1_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = REFRESH_OUTPUT_DIR,
) -> ExperimentConfig:
    output_dir = Path(output_root) / output_subdir
    return replace(
        build_stage1_linear_config(refresh_output_dir=refresh_output_dir),
        output_dir=output_dir,
        stage2_enabled=True,
        stage2_workflow="JOINT",
        stage2_detector_arch="LOCAL",
        stage2_decision_loss="BIT_BCE",
        stage2_checkpoint_path=Path(stage1_checkpoint_path),
    )


def control_lines(config: ExperimentConfig) -> list[str]:
    resource_df = build_frame_resource_df(config)
    lines = [
        f"- Output dir: `{config.output_dir}`",
        f"- Stage 1 checkpoint: `{config.stage2_checkpoint_path}`",
        f"- Base seed: `{config.base_seed}`",
        f"- Frame: `M={config.M}, K={config.K}, N={config.N}, P={config.N_pilots}, G={config.N_guard}, R={config.redundancy_dimensions}`",
        f"- Workflow: `joint`",
        f"- Train SNR: `{config.train_ebn0_db:.1f} dB`, train range `[{config.train_ebn0_db_min:.1f}, {config.train_ebn0_db_max:.1f}] dB`, eval SNR `{config.eval_ebn0_db:.1f} dB`",
        f"- Detector-first stage: epochs `{config.stage2_warmstart_epochs}`, lr `{config.stage2_warmstart_learning_rate}`, freeze `W` and `V`",
        f"- Decision loss: `{config.stage2_decision_loss}`",
        f"- Reopen-V stage: enabled `{config.stage2_reopen_v_after_win}`, require detector win `{config.stage2_reopen_v_requires_gain}`, "
        f"gain tolerance `{config.stage2_reopen_v_gain_tol}`, epochs `{config.stage2_reopen_v_epochs}`, lr `{config.stage2_reopen_v_learning_rate}`",
        f"- Detector: `{config.stage2_detector_arch.lower()}` with channels `{config.stage2_local_channels}`, kernel `{config.stage2_local_kernel_size}`, residual scale init `{config.stage2_residual_scale_init}`, cancellation scale init `{config.stage2_cancellation_scale_init}`",
        f"- Features: confidence `{config.stage2_use_confidence_features}`, symbol correction head `{config.stage2_use_symbol_correction_head}`, residual logit head `{config.stage2_use_residual_logit_head}`",
        f"- Losses: CFO aux `{config.stage2_cfo_loss_weight}`, hard-CFO weight `{config.stage2_hard_cfo_loss_weight}`, non-inferiority `{config.stage2_noninferiority_weight}`, eta `{config.stage2_loss_mse_weight}`",
        f"- Frozen-front-end operator terms logged only: `{config.stage2_detector_only_logs_operator_terms}`",
        f"- Hard-CFO checkpoint target: abs CFO `{config.stage2_selection_abs_cfo_points}` with weights `{config.stage2_selection_abs_cfo_weights}`",
    ]
    if not resource_df.empty:
        row = resource_df.iloc[0]
        lines.append(
            f"- Structured occupancy: learned-data `{row['payload_region_fraction']:.3f}`, information `{row['payload_fraction']:.3f}`, guard `{row['guard_fraction']:.3f}`"
        )
    return lines


def run_stage2_joint(config: ExperimentConfig | None = None) -> object:
    config = build_stage2_joint_config() if config is None else config
    if config.stage2_checkpoint_path is None or not Path(config.stage2_checkpoint_path).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {config.stage2_checkpoint_path}")
    print(f"Running Stage 2 joint experiment -> {config.output_dir}")
    result = run_stage2_experiment(config=config)
    report_path = write_markdown_report(
        config=config,
        result=result,
        output_dir=config.output_dir,
        plot_files=PLOT_FILES,
        extra_metadata={
            "title": "Residual-CFO Stage 2 Joint Report",
            "overview_lines": [
                "Script-only Stage 2 joint package for the structured `16QAM` experiment.",
                "This workflow first trains the nonlinear detector on top of the frozen Stage 1 front-end, then optionally reopens `V` while keeping `W` frozen.",
            ],
            "control_lines": control_lines(config),
        },
    )
    print(f"Wrote report -> {report_path}")
    return result


if __name__ == "__main__":
    run_stage2_joint()
