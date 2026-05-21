from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from comm_core import ExperimentConfig, run_papr_constraint_sidecar
from run_stage1_linear import build_stage1_linear_config


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "stage1_papr_sidecar_outputs"
OUTPUT_SUBDIR = "16qam_n45_r9"

SMOKE_LAMBDA_PAPR_VALUES = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
SMOKE_STAGE_A_EPOCHS = 60
SMOKE_STAGE_B_EPOCHS = 120
SMOKE_STAGE_C_EPOCHS = 180
SMOKE_BER_BLOCKS = 4096
SMOKE_CONSTELLATION_NUM_BLOCKS = 128
SMOKE_SPECTRAL_EVAL_BLOCKS = 2048
SMOKE_PAPR_EVAL_BLOCKS = 2048


def _parse_lambda_values(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one lambda value.")
    return values


def build_stage1_papr_sidecar_config(
    *,
    smoke: bool,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
) -> ExperimentConfig:
    config = build_stage1_linear_config(
        output_root=output_root,
        output_subdir=output_subdir,
        refresh_output_dir=False,
    )
    if not smoke:
        return config
    return replace(
        config,
        stage_a_epochs=SMOKE_STAGE_A_EPOCHS,
        stage_b_epochs=SMOKE_STAGE_B_EPOCHS,
        stage_c_epochs=SMOKE_STAGE_C_EPOCHS,
        ber_blocks=SMOKE_BER_BLOCKS,
        constellation_num_blocks=SMOKE_CONSTELLATION_NUM_BLOCKS,
        spectral_eval_blocks=SMOKE_SPECTRAL_EVAL_BLOCKS,
        papr_eval_blocks=SMOKE_PAPR_EVAL_BLOCKS,
    )


def _print_summary(result) -> None:
    print("PAPR sidecar complete")
    print(f"Output root: {result.artifact_paths['trial_summary_csv'].parent}")
    print(f"Best lambda_papr: {result.best_lambda_papr}")
    print("Trial summary:")
    print(result.constrained_trial_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 PAPR-regularization sidecar sweep.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full Stage 1 schedule instead of the reduced smoke sweep.",
    )
    parser.add_argument(
        "--lambda-values",
        type=str,
        default=None,
        help="Comma-separated lambda_papr values. Defaults to the smoke sweep values.",
    )
    args = parser.parse_args()

    lambda_values = _parse_lambda_values(args.lambda_values)
    if lambda_values is None:
        lambda_values = SMOKE_LAMBDA_PAPR_VALUES

    config = build_stage1_papr_sidecar_config(smoke=not args.full)
    print(f"Running Stage 1 PAPR sidecar -> {config.output_dir}")
    print(f"Mode: {'full' if args.full else 'smoke'}")
    print(f"lambda_papr sweep: {lambda_values}")
    result = run_papr_constraint_sidecar(
        base_config=config,
        lambda_papr_values=lambda_values,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
