from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from comm_core import ExperimentConfig, default_config, run_full_experiment
from reporting import build_frame_resource_df, write_markdown_report


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_FILES = (
    "ber_vs_cfo.png",
    "ber_vs_snr_by_cfo.png",
    "offdiag_leakage_vs_cfo.png",
    "nearest_neighbor_leakage_vs_cfo.png",
    "evm_vs_cfo.png",
    "constellation_snapshots.png",
    "operator_heatmaps.png",
    "spectral_fairness.png",
    "papr_ccdf.png",
    "frequency_domain_bases_all.png",
    "frequency_domain_bases_random.png",
    "time_domain_waveform.png",
    "time_domain_envelope_phase.png",
)


OUTPUT_ROOT = SCRIPT_DIR / "stage1_linear_outputs"
OUTPUT_SUBDIR = "16qam_n45_r9"
REFRESH_OUTPUT_DIR = True

MODULATION = "16QAM"
BASE_SEED = 211
M = 64
K = 56
N = 45
N_DATA = 45
N_PILOTS = 4
N_GUARD = 4
PILOT_ESTIMATION_ENABLED = False

TRAIN_EBN0_DB = 15.0
TRAIN_EBN0_DB_MIN = 10.0
TRAIN_EBN0_DB_MAX = 18.0
EVAL_EBN0_DB = 12.0

STAGE_A_EPOCHS = 350
STAGE_B_EPOCHS = 650
STAGE_C_EPOCHS = 900
STAGE_B_CFO = 0.05
STAGE_C_CFO = 0.10

LAMBDA_0 = 1.00
LAMBDA_1 = 0.90
LAMBDA_2 = 0.18
LAMBDA_3 = 5e-4
LAMBDA_SYM = 1.25
LAMBDA_NN = 1.10
LOCAL_LEAKAGE_MAX_DISTANCE = 2

TRAIN_CFO_SUPPORT = (
    -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10,
)
TRAIN_CFO_SUPPORT_WEIGHTS = (
    0.20, 0.35, 1.20, 1.45, 1.60, 1.40, 1.10,
    1.10, 1.40, 1.60, 1.45, 1.20, 0.35, 0.20,
)

OPERATOR_EVAL_CFO = np.linspace(-0.20, 0.20, 41)
BER_EVAL_CFO = np.linspace(-0.20, 0.20, 41)
SNR_SWEEP_EBN0_DB_GRID = np.arange(0.0, 22.0, 2.0)
SNR_SWEEP_CFO_VALUES = (0.0, 0.05, 0.10)
HEATMAP_CFO = (0.05, 0.10)
CONSTELLATION_CFO = (0.0, 0.05, 0.10)

BER_BLOCKS = 32768
BER_BATCH_SIZE = 1024
CONSTELLATION_NUM_BLOCKS = 512
SPECTRAL_EVAL_BLOCKS = 4096
PAPR_EVAL_BLOCKS = 4096


def build_stage1_linear_config(
    *,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = REFRESH_OUTPUT_DIR,
) -> ExperimentConfig:
    output_dir = Path(output_root) / output_subdir
    return replace(
        default_config(),
        base_seed=BASE_SEED,
        frame_structure_enabled=True,
        pilot_estimation_enabled=PILOT_ESTIMATION_ENABLED,
        refresh_output_dir=refresh_output_dir,
        modulation=MODULATION,
        M=M,
        K=K,
        N=N,
        N_data=N_DATA,
        N_pilots=N_PILOTS,
        N_guard=N_GUARD,
        train_ebn0_db=TRAIN_EBN0_DB,
        train_ebn0_db_min=TRAIN_EBN0_DB_MIN,
        train_ebn0_db_max=TRAIN_EBN0_DB_MAX,
        eval_ebn0_db=EVAL_EBN0_DB,
        stage_a_epochs=STAGE_A_EPOCHS,
        stage_b_epochs=STAGE_B_EPOCHS,
        stage_c_epochs=STAGE_C_EPOCHS,
        stage_b_cfo=STAGE_B_CFO,
        stage_c_cfo=STAGE_C_CFO,
        lambda_0=LAMBDA_0,
        lambda_1=LAMBDA_1,
        lambda_2=LAMBDA_2,
        lambda_3=LAMBDA_3,
        lambda_sym=LAMBDA_SYM,
        lambda_nn=LAMBDA_NN,
        local_leakage_max_distance=LOCAL_LEAKAGE_MAX_DISTANCE,
        train_cfo_support=TRAIN_CFO_SUPPORT,
        train_cfo_support_weights=TRAIN_CFO_SUPPORT_WEIGHTS,
        operator_eval_cfo=OPERATOR_EVAL_CFO,
        ber_eval_cfo=BER_EVAL_CFO,
        snr_sweep_ebn0_db_grid=SNR_SWEEP_EBN0_DB_GRID,
        snr_sweep_cfo_values=SNR_SWEEP_CFO_VALUES,
        heatmap_cfo=HEATMAP_CFO,
        constellation_cfo=CONSTELLATION_CFO,
        ber_blocks=BER_BLOCKS,
        ber_batch_size=BER_BATCH_SIZE,
        constellation_num_blocks=CONSTELLATION_NUM_BLOCKS,
        spectral_eval_blocks=SPECTRAL_EVAL_BLOCKS,
        papr_eval_blocks=PAPR_EVAL_BLOCKS,
        output_dir=output_dir,
    )


def control_lines(config: ExperimentConfig) -> list[str]:
    resource_df = build_frame_resource_df(config)
    support_pairs = ", ".join(
        f"{cfo:+.2f}:{weight:.2f}" for cfo, weight in zip(TRAIN_CFO_SUPPORT, TRAIN_CFO_SUPPORT_WEIGHTS)
    )
    lines = [
        f"- Output dir: `{config.output_dir}`",
        f"- Base seed: `{config.base_seed}`",
        f"- Frame: `M={config.M}, K={config.K}, N={config.N}, P={config.N_pilots}, G={config.N_guard}, R={config.redundancy_dimensions}`",
        f"- Train SNR: `{config.train_ebn0_db:.1f} dB`, train range `[{config.train_ebn0_db_min:.1f}, {config.train_ebn0_db_max:.1f}] dB`, eval SNR `{config.eval_ebn0_db:.1f} dB`",
        f"- Stage epochs: `A={config.stage_a_epochs}, B={config.stage_b_epochs}, C={config.stage_c_epochs}`",
        f"- Stage CFO spans: `B={config.stage_b_cfo:.2f}, C={config.stage_c_cfo:.2f}`",
        f"- Acceptance gates: clean identity `<= {config.stage1_clean_identity_tol:.1e}`, clean leakage `<= {config.stage1_clean_leakage_tol:.1e}`",
        f"- Losses: `lambda_0={config.lambda_0}`, `lambda_1={config.lambda_1}`, `lambda_2={config.lambda_2}`, `lambda_3={config.lambda_3}`, `lambda_sym={config.lambda_sym}`, `lambda_nn={config.lambda_nn}`",
        f"- CFO support weights: `{support_pairs}`",
        f"- Eval grids: operator `{len(config.operator_eval_cfo)}`, BER `{len(config.ber_eval_cfo)}`, SNR sweep `{len(config.snr_sweep_ebn0_db_grid)}`",
        f"- Eval batches: BER blocks `{config.ber_blocks}`, constellation blocks `{config.constellation_num_blocks}`, spectral blocks `{config.spectral_eval_blocks}`, PAPR blocks `{config.papr_eval_blocks}`",
    ]
    if not resource_df.empty:
        row = resource_df.iloc[0]
        lines.append(
            f"- Structured occupancy: learned-data `{row['payload_region_fraction']:.3f}`, information `{row['payload_fraction']:.3f}`, guard `{row['guard_fraction']:.3f}`"
        )
    return lines


def run_stage1_linear(config: ExperimentConfig | None = None) -> object:
    config = build_stage1_linear_config() if config is None else config
    print(f"Running Stage 1 linear experiment -> {config.output_dir}")
    result = run_full_experiment(config=config)
    report_path = write_markdown_report(
        config=config,
        result=result,
        output_dir=config.output_dir,
        plot_files=PLOT_FILES,
        extra_metadata={
            "title": "Residual-CFO Stage 1 Linear Report",
            "overview_lines": [
                "Script-only Stage 1 linear learned-transceiver package for the structured `16QAM` experiment.",
                "This output folder includes the saved Stage 1 checkpoint consumed by the Stage 2 nonlinear workflows.",
            ],
            "control_lines": control_lines(config),
        },
    )
    print(f"Wrote report -> {report_path}")
    return result


if __name__ == "__main__":
    run_stage1_linear()
