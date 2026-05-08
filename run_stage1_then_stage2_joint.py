from __future__ import annotations

from pathlib import Path

from run_stage1_linear import OUTPUT_ROOT as STAGE1_OUTPUT_ROOT
from run_stage1_linear import OUTPUT_SUBDIR as STAGE1_OUTPUT_SUBDIR
from run_stage1_linear import build_stage1_linear_config, run_stage1_linear
from run_stage2_joint import OUTPUT_ROOT as STAGE2_OUTPUT_ROOT
from run_stage2_joint import OUTPUT_SUBDIR as STAGE2_OUTPUT_SUBDIR
from run_stage2_joint import build_stage2_joint_config, run_stage2_joint


SCRIPT_DIR = Path(__file__).resolve().parent

RUN_STAGE1 = True
RUN_STAGE2 = True

EXISTING_STAGE1_CHECKPOINT = STAGE1_OUTPUT_ROOT / STAGE1_OUTPUT_SUBDIR / "stage1_checkpoint.pt"


def run_workflow() -> tuple[object | None, object | None]:
    stage1_result = None
    stage2_result = None
    stage1_checkpoint_path = EXISTING_STAGE1_CHECKPOINT

    if RUN_STAGE1:
        stage1_config = build_stage1_linear_config(
            output_root=STAGE1_OUTPUT_ROOT,
            output_subdir=STAGE1_OUTPUT_SUBDIR,
        )
        stage1_result = run_stage1_linear(config=stage1_config)
        stage1_checkpoint_path = Path(stage1_result.artifact_paths["stage1_checkpoint"])

    if RUN_STAGE2:
        stage2_config = build_stage2_joint_config(
            stage1_checkpoint_path=stage1_checkpoint_path,
            output_root=STAGE2_OUTPUT_ROOT,
            output_subdir=STAGE2_OUTPUT_SUBDIR,
        )
        stage2_result = run_stage2_joint(config=stage2_config)

    return stage1_result, stage2_result


if __name__ == "__main__":
    run_workflow()
