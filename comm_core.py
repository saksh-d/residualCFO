from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SUPPORTED_MODULATIONS = ("QPSK", "16QAM")
SUPPORTED_TONE_PATTERNS = ("CONTIGUOUS", "SPREAD")
SUPPORTED_STAGE2_WORKFLOWS = ("NONLINEAR_ONLY", "JOINT")
SUPPORTED_STAGE2_DETECTORS = ("DENSE", "LOCAL")
SUPPORTED_STAGE2_DECISION_LOSSES = ("SYMBOL_CE", "BIT_BCE")


def modulation_bits_per_symbol(modulation: str) -> int:
    modulation = modulation.upper()
    if modulation == "QPSK":
        return 2
    if modulation == "16QAM":
        return 4
    raise ValueError(f"Unsupported modulation: {modulation}")


def normalize_tone_pattern(pattern: str) -> str:
    pattern = pattern.upper()
    if pattern not in SUPPORTED_TONE_PATTERNS:
        raise ValueError(f"Unsupported tone pattern: {pattern}")
    return pattern


@dataclass
class ExperimentConfig:
    device: str = "cpu"
    base_seed: int = 0
    N: int = 32
    M: int = 48
    K: int = 0
    modulation: str = "16QAM"
    learned_init_pattern: str = "SPREAD"
    refresh_output_dir: bool = True
    learning_rate: float = 2.5e-3
    operator_batch_size: int = 96
    train_symbol_batch_size: int = 384
    train_ebn0_db: float = 15.0
    train_ebn0_db_min: float | None = None
    train_ebn0_db_max: float | None = None
    eval_ebn0_db: float = 15.0
    train_cfo_grid_points: int = 0
    train_cfo_support: tuple[float, ...] = ()
    train_cfo_support_weights: tuple[float, ...] = ()
    stage_a_epochs: int = 250
    stage_b_epochs: int = 450
    stage_c_epochs: int = 650
    stage_b_cfo: float = 0.05
    stage_c_cfo: float = 0.10
    lambda_0: float = 1.0
    lambda_1: float = 1.0
    lambda_2: float = 0.1
    lambda_3: float = 1e-3
    lambda_sym: float = 1.0
    lambda_nn: float = 0.0
    offdiag_distance_weight_power: float = 0.0
    local_leakage_max_distance: int = 2
    spectral_constraint_enabled: bool = False
    lambda_spec: float = 0.0
    spectral_guard_bins: int = 1
    history_log_interval: int = 25
    stage_validation_points: int = 9
    stage_a_identity_tol: float = 5e-3
    stage_a_leakage_tol: float = 1e-3
    stage1_clean_identity_tol: float = 1e-3
    stage1_clean_leakage_tol: float = 1e-3
    stage1_zero_cfo_ber_margin: float = 5e-4
    stage1_zero_cfo_ber_scale: float = 2.0
    terminal_progress_enabled: bool = True
    operator_eval_cfo: np.ndarray = field(default_factory=lambda: np.linspace(-0.20, 0.20, 41))
    ber_eval_cfo: np.ndarray = field(default_factory=lambda: np.linspace(-0.20, 0.20, 41))
    snr_sweep_ebn0_db_grid: np.ndarray = field(default_factory=lambda: np.arange(0.0, 22.0, 2.0))
    snr_sweep_cfo_values: tuple[float, ...] = (0.0, 0.05, 0.10)
    heatmap_cfo: tuple[float, ...] = (0.0, 0.10)
    ber_blocks: int = 32768
    ber_batch_size: int = 1024
    fft_len: int = 2048
    constellation_cfo: tuple[float, ...] = (0.0, 0.10, 0.20)
    constellation_num_blocks: int = 512
    constellation_plot_points: int = 2500
    spectral_eval_blocks: int = 4096
    spectral_fft_len: int = 4096
    occupied_bandwidth_fraction: float = 0.99
    papr_eval_blocks: int = 4096
    frame_structure_enabled: bool = False
    N_data: int = 0
    N_pilots: int = 0
    N_guard: int = 0
    pilot_estimation_enabled: bool = True
    pilot_estimator_cfo_grid: np.ndarray = field(default_factory=lambda: np.linspace(-0.20, 0.20, 81))
    stage2_enabled: bool = False
    stage2_workflow: str = "JOINT"
    stage2_detector_arch: str = "LOCAL"
    stage2_residual_scale_init: float = 0.1
    stage2_hidden_multiplier: int = 4
    stage2_local_channels: int = 32
    stage2_local_kernel_size: int = 5
    stage2_use_confidence_features: bool = True
    stage2_use_symbol_correction_head: bool = True
    stage2_use_residual_logit_head: bool = False
    stage2_decision_loss: str = "SYMBOL_CE"
    stage2_detector_only_logs_operator_terms: bool = True
    stage2_loss_mse_weight: float = 0.25
    stage2_cfo_loss_weight: float = 0.10
    stage2_hard_cfo_loss_weight: float = 1.0
    stage2_noninferiority_weight: float = 0.25
    stage2_easy_margin_threshold: float = 0.35
    stage2_easy_consistency_margin: float = 0.10
    stage2_cancellation_scale_init: float = 0.10
    stage2_checkpoint_path: Path | None = None
    stage2_checkpoint_source_path: Path | None = None
    stage2_checkpoint_snapshot_path: Path | None = None
    stage2_checkpoint_hash_sha256: str = ""
    stage2_checkpoint_file_sha256: str = ""
    stage2_checkpoint_format: str = ""
    stage2_checkpoint_acceptance_passed: bool | None = None
    stage2_checkpoint_clean_identity_loss: float | None = None
    stage2_checkpoint_clean_offdiag_leakage: float | None = None
    stage2_checkpoint_learned_ber_at_0: float | None = None
    stage2_checkpoint_ofdm_ber_at_0: float | None = None
    stage2_nonlinear_only_epochs: int = 200
    stage2_nonlinear_only_learning_rate: float = 1.0e-3
    stage2_warmstart_epochs: int = 200
    stage2_warmstart_learning_rate: float = 1.0e-3
    stage2_reopen_v_after_win: bool = True
    stage2_reopen_v_requires_gain: bool = True
    stage2_reopen_v_gain_tol: float = 0.0
    stage2_reopen_v_epochs: int = 300
    stage2_reopen_v_learning_rate: float = 5.0e-4
    stage2_selection_abs_cfo_points: tuple[float, ...] = (0.05, 0.10)
    stage2_selection_abs_cfo_weights: tuple[float, ...] = (0.35, 0.65)
    # basis_plot_indices: tuple[int, ...] | None = None
    basis_plot_indices: tuple[int, ...] = (5, 21, 45)
    random_basis_plot_count: int = 2
    random_basis_seed: int = 321
    correlation_plot_cfo: float = 0.10
    output_dir: Path = Path("residual_cfo_outputs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.stage2_checkpoint_path is not None:
            self.stage2_checkpoint_path = Path(self.stage2_checkpoint_path)
        if self.stage2_checkpoint_source_path is not None:
            self.stage2_checkpoint_source_path = Path(self.stage2_checkpoint_source_path)
        if self.stage2_checkpoint_snapshot_path is not None:
            self.stage2_checkpoint_snapshot_path = Path(self.stage2_checkpoint_snapshot_path)
        self.modulation = self.modulation.upper()
        if self.modulation not in SUPPORTED_MODULATIONS:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        self.stage2_workflow = self.stage2_workflow.upper()
        if self.stage2_workflow not in SUPPORTED_STAGE2_WORKFLOWS:
            raise ValueError(f"Unsupported stage2_workflow: {self.stage2_workflow}")
        self.stage2_detector_arch = self.stage2_detector_arch.upper()
        if self.stage2_detector_arch not in SUPPORTED_STAGE2_DETECTORS:
            raise ValueError(f"Unsupported stage2_detector_arch: {self.stage2_detector_arch}")
        self.stage2_decision_loss = self.stage2_decision_loss.upper()
        if self.stage2_decision_loss not in SUPPORTED_STAGE2_DECISION_LOSSES:
            raise ValueError(f"Unsupported stage2_decision_loss: {self.stage2_decision_loss}")
        self.learned_init_pattern = normalize_tone_pattern(self.learned_init_pattern)
        if self.train_cfo_grid_points < 0:
            raise ValueError("train_cfo_grid_points cannot be negative.")
        if self.local_leakage_max_distance < 1:
            raise ValueError("local_leakage_max_distance must be at least 1.")
        self.train_cfo_support = tuple(float(value) for value in self.train_cfo_support)
        self.train_cfo_support_weights = tuple(float(value) for value in self.train_cfo_support_weights)
        if bool(self.train_cfo_support) != bool(self.train_cfo_support_weights):
            raise ValueError("train_cfo_support and train_cfo_support_weights must either both be set or both be empty.")
        if len(self.train_cfo_support) != len(self.train_cfo_support_weights):
            raise ValueError("train_cfo_support and train_cfo_support_weights must have the same length.")
        if self.train_cfo_support:
            support = np.asarray(self.train_cfo_support, dtype=float)
            weights = np.asarray(self.train_cfo_support_weights, dtype=float)
            if np.any(np.abs(support) < 1e-12):
                raise ValueError("train_cfo_support should omit zero; the zero-CFO loss is handled separately.")
            if np.any(weights <= 0.0):
                raise ValueError("train_cfo_support_weights must be strictly positive.")
            order = np.argsort(support)
            sorted_support = support[order]
            sorted_weights = weights[order]
            if not np.allclose(sorted_support, -sorted_support[::-1], atol=1e-12):
                raise ValueError("train_cfo_support must be symmetric around zero.")
            if not np.allclose(sorted_weights, sorted_weights[::-1], atol=1e-12):
                raise ValueError("train_cfo_support_weights must be symmetric around zero.")
        if self.frame_structure_enabled:
            if self.payload_region_enabled and self.N_data == 0:
                self.N_data = self.N
            if self.N_data <= 0:
                raise ValueError("N_data must be positive when frame_structure_enabled is True.")
            if self.N_pilots < 0:
                raise ValueError("N_pilots cannot be negative.")
            if self.N_guard < 0:
                raise ValueError("N_guard cannot be negative.")
            if self.payload_region_enabled:
                if self.K <= 0:
                    raise ValueError("K must be positive when payload_region_enabled is True.")
                if self.N != self.N_data:
                    raise ValueError(
                        f"N must equal N_data in payload-region mode, got N={self.N}, N_data={self.N_data}."
                    )
                if self.N_data > self.K:
                    raise ValueError(
                        f"N_data must be <= K in payload-region mode, got N_data={self.N_data}, K={self.K}."
                    )
                if self.M != self.K + self.N_pilots + self.N_guard:
                    raise ValueError(
                        f"M must equal K + N_pilots + N_guard in payload-region mode, got "
                        f"M={self.M}, K={self.K}, N_pilots={self.N_pilots}, N_guard={self.N_guard}."
                    )
            else:
                if self.N_pilots <= 0:
                    raise ValueError("N_pilots must be positive when frame_structure_enabled is True.")
                if self.N != self.N_data + self.N_pilots:
                    raise ValueError(
                        f"N must equal N_data + N_pilots when frame_structure_enabled is True, got "
                        f"N={self.N}, N_data={self.N_data}, N_pilots={self.N_pilots}."
                    )
                if self.M != self.N + self.N_guard:
                    raise ValueError(
                        f"M must equal N + N_guard when frame_structure_enabled is True, got "
                        f"M={self.M}, N={self.N}, N_guard={self.N_guard}."
                    )
        if self.stage2_hidden_multiplier < 1:
            raise ValueError("stage2_hidden_multiplier must be at least 1.")
        if self.stage2_local_channels < 1:
            raise ValueError("stage2_local_channels must be at least 1.")
        if self.stage2_local_kernel_size < 1 or self.stage2_local_kernel_size % 2 == 0:
            raise ValueError("stage2_local_kernel_size must be a positive odd integer.")
        if self.stage2_loss_mse_weight < 0.0:
            raise ValueError("stage2_loss_mse_weight must be non-negative.")
        if self.stage2_cfo_loss_weight < 0.0:
            raise ValueError("stage2_cfo_loss_weight must be non-negative.")
        if self.stage2_hard_cfo_loss_weight < 0.0:
            raise ValueError("stage2_hard_cfo_loss_weight must be non-negative.")
        if self.stage2_noninferiority_weight < 0.0:
            raise ValueError("stage2_noninferiority_weight must be non-negative.")
        if self.stage2_easy_margin_threshold < 0.0:
            raise ValueError("stage2_easy_margin_threshold must be non-negative.")
        if self.stage2_easy_consistency_margin < 0.0:
            raise ValueError("stage2_easy_consistency_margin must be non-negative.")
        if self.stage2_cancellation_scale_init <= 0.0:
            raise ValueError("stage2_cancellation_scale_init must be positive.")
        if self.stage2_nonlinear_only_epochs <= 0:
            raise ValueError("stage2_nonlinear_only_epochs must be positive.")
        if self.stage2_nonlinear_only_learning_rate <= 0.0:
            raise ValueError("stage2_nonlinear_only_learning_rate must be positive.")
        if self.stage2_warmstart_epochs <= 0:
            raise ValueError("stage2_warmstart_epochs must be positive.")
        if self.stage2_warmstart_learning_rate <= 0.0:
            raise ValueError("stage2_warmstart_learning_rate must be positive.")
        if self.stage2_reopen_v_epochs <= 0:
            raise ValueError("stage2_reopen_v_epochs must be positive.")
        if self.stage2_reopen_v_learning_rate <= 0.0:
            raise ValueError("stage2_reopen_v_learning_rate must be positive.")
        self.stage2_selection_abs_cfo_points = tuple(float(value) for value in self.stage2_selection_abs_cfo_points)
        self.stage2_selection_abs_cfo_weights = tuple(float(value) for value in self.stage2_selection_abs_cfo_weights)
        if not self.stage2_selection_abs_cfo_points:
            raise ValueError("stage2_selection_abs_cfo_points must not be empty.")
        if len(self.stage2_selection_abs_cfo_points) != len(self.stage2_selection_abs_cfo_weights):
            raise ValueError("stage2_selection_abs_cfo_points and stage2_selection_abs_cfo_weights must match in length.")
        if any(value <= 0.0 for value in self.stage2_selection_abs_cfo_points):
            raise ValueError("stage2_selection_abs_cfo_points must be strictly positive.")
        if any(weight <= 0.0 for weight in self.stage2_selection_abs_cfo_weights):
            raise ValueError("stage2_selection_abs_cfo_weights must be strictly positive.")

    @property
    def bits_per_symbol(self) -> int:
        return modulation_bits_per_symbol(self.modulation)

    @property
    def payload_region_enabled(self) -> bool:
        return self.frame_structure_enabled and self.K > 0

    @property
    def data_symbol_count(self) -> int:
        if self.frame_structure_enabled:
            return self.N_data
        return self.N

    @property
    def payload_bin_count(self) -> int:
        if self.payload_region_enabled:
            return self.K
        if self.frame_structure_enabled:
            return self.N_data + self.N_pilots
        return self.N

    @property
    def active_resource_bins(self) -> int:
        if self.payload_region_enabled:
            return self.K + self.N_pilots
        if self.frame_structure_enabled:
            return self.N_data + self.N_pilots
        return self.N

    @property
    def redundancy_dimensions(self) -> int:
        if self.payload_region_enabled:
            return self.K - self.N_data
        return 0

    @property
    def N_active(self) -> int:
        if self.frame_structure_enabled:
            return self.active_resource_bins
        return self.N

    @property
    def payload_fraction(self) -> float:
        return float(self.data_symbol_count) / float(self.M) if self.M > 0 else 0.0

    @property
    def payload_region_fraction(self) -> float:
        return float(self.payload_bin_count) / float(self.M) if self.M > 0 else 0.0


@dataclass(frozen=True)
class StageSpec:
    name: str
    epochs: int
    cfo_span: float
    use_cfo_losses: bool


@dataclass
class TrainingResult:
    learned_tx: torch.Tensor
    learned_rx: torch.Tensor
    history_df: pd.DataFrame
    stage_summary_df: pd.DataFrame
    stage_failed: bool
    failed_stage: str | None
    stop_reason: str


@dataclass
class ExperimentResult:
    learned_tx: torch.Tensor
    learned_rx: torch.Tensor
    history_df: pd.DataFrame
    stage_summary_df: pd.DataFrame
    operator_df: pd.DataFrame
    diagonal_df: pd.DataFrame
    ber_df: pd.DataFrame
    ber_snr_df: pd.DataFrame
    constellation_df: pd.DataFrame
    spectral_df: pd.DataFrame
    spectral_summary_df: pd.DataFrame
    papr_df: pd.DataFrame
    papr_summary_df: pd.DataFrame
    summary_df: pd.DataFrame
    snr_summary_df: pd.DataFrame
    artifact_paths: dict[str, Path]
    mapping_markdown: str
    summary_markdown: str
    stage_failed: bool
    failed_stage: str | None
    stop_reason: str


@dataclass
class AblationResult:
    ablation_df: pd.DataFrame
    summary_df: pd.DataFrame
    artifact_paths: dict[str, Path]
    fixed_eps: float
    fixed_eval_ebn0_db: float
    m_values: tuple[int, ...]


@dataclass
class SpectralConstraintResult:
    baseline_result: ExperimentResult
    best_constrained_result: ExperimentResult | None
    constrained_trial_df: pd.DataFrame
    comparison_summary_df: pd.DataFrame
    artifact_paths: dict[str, Path]
    lambda_spec_values: tuple[float, ...]
    best_lambda_spec: float | None


@dataclass
class FrameStructuredStudyResult:
    experiment_result: ExperimentResult
    resource_df: pd.DataFrame
    pilot_summary_df: pd.DataFrame
    artifact_paths: dict[str, Path]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_terminal_progress(config: ExperimentConfig, message: str) -> None:
    if bool(getattr(config, "terminal_progress_enabled", True)):
        print(message, flush=True)


def normalize_columns(basis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(eps)
    return basis / norms


def ebn0_to_noise_variance(ebn0_db: float, bits_per_symbol: int) -> float:
    ebn0 = 10 ** (float(ebn0_db) / 10.0)
    return 1.0 / (bits_per_symbol * ebn0)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def stage_specs(config: ExperimentConfig) -> list[StageSpec]:
    return [
        StageSpec("Stage A", config.stage_a_epochs, 0.0, False),
        StageSpec("Stage B", config.stage_b_epochs, config.stage_b_cfo, True),
        StageSpec("Stage C", config.stage_c_epochs, config.stage_c_cfo, True),
    ]


def default_stage1_checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / "stage1_checkpoint.pt"


def _json_safe_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tensor_hash_update(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def compute_stage1_checkpoint_payload_hash(payload: dict[str, object]) -> str:
    digest = hashlib.sha256()
    hashable_payload = {
        "format": payload.get("format"),
        "base_seed": payload.get("base_seed"),
        "config": _json_safe_value(payload.get("config", {})),
        "provenance": _json_safe_value(payload.get("provenance", {})),
        "stage1_acceptance_summary": _json_safe_value(payload.get("stage1_acceptance_summary", {})),
    }
    digest.update(json.dumps(hashable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    learned_tx = payload.get("learned_tx")
    learned_rx = payload.get("learned_rx")
    if not isinstance(learned_tx, torch.Tensor) or not isinstance(learned_rx, torch.Tensor):
        raise ValueError("Stage 1 checkpoint payload hash requires learned_tx and learned_rx tensors.")
    _tensor_hash_update(digest, learned_tx)
    _tensor_hash_update(digest, learned_rx)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage1_zero_cfo_ber_upper_bound(config: ExperimentConfig, ofdm_ber_at_0: float) -> float:
    return min(
        config.stage1_zero_cfo_ber_scale * float(ofdm_ber_at_0),
        float(ofdm_ber_at_0) + config.stage1_zero_cfo_ber_margin,
    ) + config.stage1_zero_cfo_ber_margin


def build_stage1_acceptance_summary(
    config: ExperimentConfig,
    training_result: TrainingResult,
    summary_df: pd.DataFrame,
) -> dict[str, object]:
    learned = summary_df[summary_df["method"] == "Learned"].iloc[0]
    ofdm = summary_df[summary_df["method"] == "OFDM"].iloc[0]

    clean_identity = float(learned["clean_identity_loss"])
    clean_leakage = float(learned["clean_offdiag_leakage"])
    learned_ber_at_0 = float(learned["ber_at_0"])
    ofdm_ber_at_0 = float(ofdm["ber_at_0"])
    zero_cfo_ber_upper_bound = stage1_zero_cfo_ber_upper_bound(config, ofdm_ber_at_0)

    failed_checks: list[str] = []
    if training_result.stage_failed:
        failed_checks.append(training_result.failed_stage or "Stage A")
    if clean_identity > config.stage1_clean_identity_tol:
        failed_checks.append("clean_identity_loss")
    if clean_leakage > config.stage1_clean_leakage_tol:
        failed_checks.append("clean_offdiag_leakage")
    if learned_ber_at_0 > zero_cfo_ber_upper_bound:
        failed_checks.append("zero_cfo_ber_closeness")

    passed = len(failed_checks) == 0
    if passed:
        stop_reason = "Completed all stages. Stage 1 acceptance passed."
    else:
        stop_reason = (
            "Stage 1 acceptance failed: " + ", ".join(failed_checks) + "."
        )

    return {
        "passed": passed,
        "failed_checks": failed_checks,
        "stop_reason": stop_reason,
        "geometry": {
            "modulation": config.modulation,
            "M": int(config.M),
            "K": int(config.K),
            "N": int(config.N),
            "N_data": int(config.N_data),
            "N_pilots": int(config.N_pilots),
            "N_guard": int(config.N_guard),
            "R": int(config.redundancy_dimensions),
        },
        "schedule": {
            "stage_a_epochs": int(config.stage_a_epochs),
            "stage_b_epochs": int(config.stage_b_epochs),
            "stage_c_epochs": int(config.stage_c_epochs),
            "stage_b_cfo": float(config.stage_b_cfo),
            "stage_c_cfo": float(config.stage_c_cfo),
        },
        "criteria": {
            "clean_identity_loss_max": float(config.stage1_clean_identity_tol),
            "clean_offdiag_leakage_max": float(config.stage1_clean_leakage_tol),
            "zero_cfo_ber_margin": float(config.stage1_zero_cfo_ber_margin),
            "zero_cfo_ber_scale": float(config.stage1_zero_cfo_ber_scale),
            "zero_cfo_ber_upper_bound": float(zero_cfo_ber_upper_bound),
        },
        "metrics": {
            "clean_identity_loss": clean_identity,
            "clean_offdiag_leakage": clean_leakage,
            "learned_ber_at_0": learned_ber_at_0,
            "ofdm_ber_at_0": ofdm_ber_at_0,
        },
    }


def save_stage1_checkpoint(
    config: ExperimentConfig,
    training_result: TrainingResult,
    stage1_acceptance_summary: dict[str, object],
    path: Path | None = None,
) -> Path:
    target = default_stage1_checkpoint_path(config.output_dir) if path is None else Path(path)
    payload = {
        "format": "stage1_linear_receiver_v2",
        "base_seed": int(config.base_seed),
        "config": {name: getattr(config, name) for name in ExperimentConfig.__dataclass_fields__},
        "provenance": {
            "output_dir": str(config.output_dir),
        },
        "stage1_acceptance_summary": stage1_acceptance_summary,
        "learned_tx": training_result.learned_tx.detach().cpu(),
        "learned_rx": training_result.learned_rx.detach().cpu(),
    }
    payload["payload_hash_sha256"] = compute_stage1_checkpoint_payload_hash(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    target.write_bytes(buffer.getvalue())
    return target


def load_stage1_checkpoint(
    path: Path,
    device: str = "cpu",
    require_v2: bool = False,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported Stage 1 checkpoint format: {path}")
    checkpoint_format = payload.get("format")
    if checkpoint_format == "stage1_linear_receiver_v2":
        expected_hash = payload.get("payload_hash_sha256")
        computed_hash = compute_stage1_checkpoint_payload_hash(payload)
        if expected_hash != computed_hash:
            raise ValueError(
                f"Stage 1 checkpoint hash mismatch for {path}: stored={expected_hash!r}, computed={computed_hash!r}."
            )
        return payload
    if checkpoint_format == "stage1_linear_receiver_v1":
        if require_v2:
            raise ValueError(f"Stage 2 requires a v2 Stage 1 checkpoint with acceptance metadata: {path}")
        return payload
    if require_v2:
        raise ValueError(f"Stage 2 requires a v2 Stage 1 checkpoint with acceptance metadata: {path}")
    raise ValueError(f"Unsupported Stage 1 checkpoint format: {path}")


def snapshot_stage1_checkpoint(source_path: Path, output_dir: Path) -> Path:
    target = Path(output_dir) / "stage1_checkpoint_snapshot.pt"
    shutil.copy2(source_path, target)
    return target


def stage2_checkpoint_preflight(
    config: ExperimentConfig,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    from channel import effective_operator
    from receiver import EvaluationScheme, evaluate_scheme_set, identity_loss, offdiag_leakage_ratio
    from transmitter import make_ofdm_baseline_transceiver

    acceptance = checkpoint.get("stage1_acceptance_summary")
    if not isinstance(acceptance, dict):
        raise ValueError("Stage 2 requires checkpoint acceptance metadata.")
    if not bool(acceptance.get("passed", False)):
        raise ValueError(f"Stage 1 checkpoint did not pass acceptance: {acceptance.get('stop_reason', 'unknown')}")
    checkpoint_config = checkpoint.get("config", {})

    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)
    operator_clean = effective_operator(learned_tx, learned_rx, torch.tensor([0.0], device=config.device))
    clean_identity = float(identity_loss(operator_clean).item())
    clean_leakage = float(offdiag_leakage_ratio(operator_clean)[0].item())
    metrics = acceptance.get("metrics", {})
    expected_identity = float(metrics.get("clean_identity_loss"))
    expected_leakage = float(metrics.get("clean_offdiag_leakage"))
    metric_tol = 1e-9
    if abs(clean_identity - expected_identity) > metric_tol:
        raise ValueError(
            f"Stage 1 checkpoint clean-identity mismatch: expected={expected_identity:.12e}, got={clean_identity:.12e}."
        )
    if abs(clean_leakage - expected_leakage) > metric_tol:
        raise ValueError(
            f"Stage 1 checkpoint clean-leakage mismatch: expected={expected_leakage:.12e}, got={clean_leakage:.12e}."
        )

    preflight_config = replace(
        config,
        eval_ebn0_db=float(checkpoint_config.get("eval_ebn0_db", config.eval_ebn0_db)),
        ber_blocks=int(checkpoint_config.get("ber_blocks", config.ber_blocks)),
        ber_batch_size=int(checkpoint_config.get("ber_batch_size", config.ber_batch_size)),
    )
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(preflight_config)
    eval_df = evaluate_scheme_set(
        config=preflight_config,
        schemes={
            "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
            "Learned": EvaluationScheme(tx_basis=learned_tx, rx_basis=learned_rx, nonlinear_receiver=None),
        },
        cfo_points=np.array([0.0], dtype=float),
        ebn0_db=preflight_config.eval_ebn0_db,
        num_blocks=preflight_config.ber_blocks,
        batch_size=preflight_config.ber_batch_size,
    )
    learned_ber_at_0 = float(eval_df[eval_df["method"] == "Learned"]["ber"].iloc[0])
    ofdm_ber_at_0 = float(eval_df[eval_df["method"] == "OFDM"]["ber"].iloc[0])
    expected_learned_ber = float(metrics.get("learned_ber_at_0"))
    expected_ofdm_ber = float(metrics.get("ofdm_ber_at_0"))
    ber_tol = 2e-5
    if abs(learned_ber_at_0 - expected_learned_ber) > ber_tol:
        raise ValueError(
            f"Stage 1 checkpoint learned BER(0) mismatch: expected={expected_learned_ber:.12e}, got={learned_ber_at_0:.12e}."
        )
    if abs(ofdm_ber_at_0 - expected_ofdm_ber) > ber_tol:
        raise ValueError(
            f"Stage 1 checkpoint OFDM BER(0) mismatch: expected={expected_ofdm_ber:.12e}, got={ofdm_ber_at_0:.12e}."
        )

    return {
        "checkpoint_format": checkpoint["format"],
        "checkpoint_payload_hash_sha256": str(checkpoint.get("payload_hash_sha256", "")),
        "stage1_acceptance_summary": acceptance,
        "reproduced_metrics": {
            "clean_identity_loss": clean_identity,
            "clean_offdiag_leakage": clean_leakage,
            "learned_ber_at_0": learned_ber_at_0,
            "ofdm_ber_at_0": ofdm_ber_at_0,
        },
    }


def annotate_stage1_summary_df(
    config: ExperimentConfig,
    training_result: TrainingResult,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    annotated = summary_df.copy()
    acceptance = build_stage1_acceptance_summary(config, training_result, annotated)
    annotated["stage1_acceptance_passed"] = acceptance["passed"]
    annotated["stage1_acceptance_stop_reason"] = acceptance["stop_reason"]
    criteria = acceptance["criteria"]
    annotated["stage1_clean_identity_tol"] = criteria["clean_identity_loss_max"]
    annotated["stage1_clean_leakage_tol"] = criteria["clean_offdiag_leakage_max"]
    annotated["stage1_zero_cfo_ber_upper_bound"] = criteria["zero_cfo_ber_upper_bound"]
    if acceptance["failed_checks"]:
        annotated["stage1_failed_checks"] = ",".join(str(item) for item in acceptance["failed_checks"])
    else:
        annotated["stage1_failed_checks"] = ""
    if not acceptance["passed"]:
        learned_mask = annotated["method"].astype(str).str.startswith("Learned")
        annotated.loc[learned_mask, "stage_failed"] = True
        annotated.loc[learned_mask, "failed_stage"] = "Stage 1 Acceptance"
        annotated.loc[learned_mask, "stop_reason"] = acceptance["stop_reason"]
    return annotated


def refresh_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def run_full_experiment(
    config: ExperimentConfig | None = None,
    tx_init: torch.Tensor | None = None,
    rx_init: torch.Tensor | None = None,
) -> ExperimentResult:
    from receiver import train_feasibility_model
    from reporting import run_final_evaluation

    config = default_config() if config is None else config
    torch.set_num_threads(max(1, torch.get_num_threads()))
    set_seed(config.base_seed)
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    training_result = train_feasibility_model(
        config=config,
        tx_init=tx_init,
        rx_init=rx_init,
    )
    log_terminal_progress(config, "[Stage1] Training finished; starting final evaluation")
    result = run_final_evaluation(config, training_result)
    stage1_acceptance_summary = build_stage1_acceptance_summary(config, training_result, result.summary_df)
    if not stage1_acceptance_summary["passed"]:
        training_result.stage_failed = True
        training_result.failed_stage = "Stage 1 Acceptance"
        training_result.stop_reason = str(stage1_acceptance_summary["stop_reason"])
        result.stage_failed = True
        result.failed_stage = "Stage 1 Acceptance"
        result.stop_reason = str(stage1_acceptance_summary["stop_reason"])
    checkpoint_path = save_stage1_checkpoint(config, training_result, stage1_acceptance_summary)
    result.artifact_paths["stage1_checkpoint"] = checkpoint_path
    return result


def run_feasibility_experiment(
    config: ExperimentConfig | None = None,
    tx_init: torch.Tensor | None = None,
    rx_init: torch.Tensor | None = None,
) -> ExperimentResult:
    return run_full_experiment(config=config, tx_init=tx_init, rx_init=rx_init)


def run_stage2_experiment(config: ExperimentConfig) -> ExperimentResult:
    from receiver import EvaluationScheme, train_stage2_nonlinear_receiver
    from reporting import run_final_evaluation
    from transmitter import make_ofdm_baseline_transceiver

    if not config.stage2_enabled:
        raise ValueError("run_stage2_experiment requires stage2_enabled=True.")
    if config.modulation != "16QAM":
        raise ValueError("Stage 2 is currently implemented only for 16QAM.")
    if not config.payload_region_enabled:
        raise ValueError("Stage 2 is currently implemented only for the structured K-bin experiment.")
    if config.stage2_checkpoint_path is None:
        raise ValueError("stage2_checkpoint_path must point to a saved Stage 1 checkpoint for Stage 2.")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    set_seed(config.base_seed)
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_source_path = Path(config.stage2_checkpoint_path)
    log_terminal_progress(config, f"[Stage2] Loading Stage 1 checkpoint -> {checkpoint_source_path}")
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
    log_terminal_progress(config, "[Stage2] Running checkpoint preflight")
    preflight = stage2_checkpoint_preflight(config, checkpoint)
    checkpoint_snapshot_path = snapshot_stage1_checkpoint(checkpoint_source_path, config.output_dir)
    log_terminal_progress(config, f"[Stage2] Snapshot checkpoint -> {checkpoint_snapshot_path}")
    config.stage2_checkpoint_source_path = checkpoint_source_path
    config.stage2_checkpoint_snapshot_path = checkpoint_snapshot_path
    config.stage2_checkpoint_hash_sha256 = str(preflight["checkpoint_payload_hash_sha256"])
    config.stage2_checkpoint_file_sha256 = file_sha256(checkpoint_snapshot_path)
    config.stage2_checkpoint_format = str(preflight["checkpoint_format"])
    config.stage2_checkpoint_acceptance_passed = bool(
        preflight["stage1_acceptance_summary"].get("passed", False)
    )
    checkpoint_metrics = preflight["stage1_acceptance_summary"].get("metrics", {})
    config.stage2_checkpoint_clean_identity_loss = float(checkpoint_metrics.get("clean_identity_loss"))
    config.stage2_checkpoint_clean_offdiag_leakage = float(checkpoint_metrics.get("clean_offdiag_leakage"))
    config.stage2_checkpoint_learned_ber_at_0 = float(checkpoint_metrics.get("learned_ber_at_0"))
    config.stage2_checkpoint_ofdm_ber_at_0 = float(checkpoint_metrics.get("ofdm_ber_at_0"))
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)

    learned_stage2 = train_stage2_nonlinear_receiver(
        config=config,
        tx_basis=learned_tx,
        rx_basis=learned_rx,
        scheme_name="LearnedNonlinear",
        seed_offset=0,
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
        stop_reason=(
            "Completed Stage 2 nonlinear-only receiver training."
            if config.stage2_workflow == "NONLINEAR_ONLY"
            else "Completed Stage 2 joint nonlinear receiver training."
        ),
    )
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)
    schemes = {
        "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
        "Learned": EvaluationScheme(tx_basis=learned_tx, rx_basis=learned_rx, nonlinear_receiver=None),
        "LearnedNonlinear": EvaluationScheme(
            tx_basis=learned_stage2.learned_tx,
            rx_basis=learned_stage2.learned_rx,
            nonlinear_receiver=learned_stage2.receiver.eval(),
        ),
    }
    result = run_final_evaluation(
        config,
        training_result,
        schemes=schemes,
        method_order=("OFDM", "Learned", "LearnedNonlinear"),
    )
    result.artifact_paths["stage1_checkpoint_snapshot"] = checkpoint_snapshot_path
    return result


def run_redundancy_ablation(
    base_config: ExperimentConfig,
    m_values: tuple[int, ...] = (32, 36, 40, 48, 64),
    fixed_n: int = 32,
    fixed_eps: float = 0.10,
    fixed_eval_ebn0_db: float = 15.0,
    train_ebn0_db: float = 15.0,
) -> AblationResult:
    from receiver import evaluate_scheme_set, train_feasibility_model
    from reporting import (
        build_redundancy_ablation_summary,
        method_display_name,
        plot_redundancy_ablation_ber,
        save_redundancy_ablation_artifacts,
    )
    from transmitter import make_ofdm_baseline_transceiver

    m_values = tuple(int(value) for value in m_values)
    if not m_values:
        raise ValueError("m_values must contain at least one redundancy setting.")

    if base_config.refresh_output_dir:
        refresh_output_dir(base_config.output_dir)
    else:
        base_config.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    fixed_cfo = np.array([float(fixed_eps)], dtype=float)
    for run_idx, m_value in enumerate(m_values):
        run_config = replace(
            base_config,
            base_seed=base_config.base_seed + 10_000 * run_idx,
            N=int(fixed_n),
            M=int(m_value),
            train_ebn0_db=float(train_ebn0_db),
            eval_ebn0_db=float(fixed_eval_ebn0_db),
            refresh_output_dir=False,
        )
        training_result = train_feasibility_model(config=run_config)
        ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(run_config)
        schemes = {
            "OFDM": (ofdm_tx, ofdm_rx),
            "Learned": (training_result.learned_tx, training_result.learned_rx),
        }
        eval_df = evaluate_scheme_set(
            config=run_config,
            schemes=schemes,
            cfo_points=fixed_cfo,
            ebn0_db=float(fixed_eval_ebn0_db),
            num_blocks=run_config.ber_blocks,
            batch_size=run_config.ber_batch_size,
            seed=91_000 + run_idx,
        ).copy()
        eval_df["method_label"] = eval_df["method"].map(method_display_name)
        eval_df["M_over_N"] = eval_df["M"] / eval_df["N"]
        eval_df["run_index"] = int(run_idx)
        eval_df["stage_failed"] = eval_df["method"] == "Learned"
        eval_df.loc[eval_df["method"] == "Learned", "stage_failed"] = training_result.stage_failed
        eval_df["failed_stage"] = ""
        eval_df.loc[eval_df["method"] == "Learned", "failed_stage"] = training_result.failed_stage or ""
        eval_df["stop_reason"] = "Fixed baseline."
        eval_df.loc[eval_df["method"] == "Learned", "stop_reason"] = training_result.stop_reason
        frames.append(eval_df)

    ablation_df = pd.concat(frames, ignore_index=True).sort_values(
        ["M_over_N", "method"],
        kind="stable",
    ).reset_index(drop=True)
    summary_df = build_redundancy_ablation_summary(ablation_df)
    artifact_paths = save_redundancy_ablation_artifacts(
        output_dir=base_config.output_dir,
        ablation_df=ablation_df,
        summary_df=summary_df,
    )
    plot_path = plot_redundancy_ablation_ber(
        config=base_config,
        ablation_df=ablation_df,
        fixed_n=int(fixed_n),
        fixed_eps=float(fixed_eps),
        fixed_eval_ebn0_db=float(fixed_eval_ebn0_db),
    )
    if plot_path is not None:
        artifact_paths["ablation_plot"] = plot_path

    return AblationResult(
        ablation_df=ablation_df,
        summary_df=summary_df,
        artifact_paths=artifact_paths,
        fixed_eps=float(fixed_eps),
        fixed_eval_ebn0_db=float(fixed_eval_ebn0_db),
        m_values=m_values,
    )


def run_spectral_constraint_sidecar(
    base_config: ExperimentConfig,
    lambda_spec_values: tuple[float, ...] = (0.05, 0.10, 0.30),
) -> SpectralConstraintResult:
    from reporting import (
        build_spectral_constraint_comparison_summary,
        build_spectral_constraint_trial_summary,
        save_spectral_constraint_artifacts,
    )

    lambda_spec_values = tuple(float(value) for value in lambda_spec_values)
    if not lambda_spec_values:
        raise ValueError("lambda_spec_values must contain at least one penalty value.")

    root_dir = base_config.output_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = replace(
        base_config,
        spectral_constraint_enabled=False,
        lambda_spec=0.0,
        refresh_output_dir=True,
        output_dir=root_dir / "baseline",
    )
    baseline_result = run_full_experiment(config=baseline_config)

    constrained_results: list[tuple[float, ExperimentResult]] = []
    for run_idx, lambda_spec in enumerate(lambda_spec_values):
        lambda_tag = f"{lambda_spec:.2f}".replace(".", "p")
        constrained_config = replace(
            base_config,
            base_seed=base_config.base_seed + 20_000 * (run_idx + 1),
            spectral_constraint_enabled=True,
            lambda_spec=lambda_spec,
            refresh_output_dir=True,
            output_dir=root_dir / f"constrained_lambda_{lambda_tag}",
        )
        constrained_results.append((lambda_spec, run_full_experiment(config=constrained_config)))

    constrained_trial_df = build_spectral_constraint_trial_summary(
        modulation=base_config.modulation,
        constrained_results=constrained_results,
    )

    passing_trials = constrained_trial_df[~constrained_trial_df["stage_failed"]]
    best_lambda_spec: float | None = None
    best_constrained_result: ExperimentResult | None = None
    if not passing_trials.empty:
        best_row = passing_trials.sort_values("integrated_log10_ber", kind="stable").iloc[0]
    elif not constrained_trial_df.empty:
        best_row = constrained_trial_df.sort_values("integrated_log10_ber", kind="stable").iloc[0]
    else:
        best_row = None
    if best_row is not None:
        best_lambda_spec = float(best_row["lambda_spec"])
        for lambda_spec, result in constrained_results:
            if abs(lambda_spec - best_lambda_spec) < 1e-12:
                best_constrained_result = result
                break

    comparison_summary_df = build_spectral_constraint_comparison_summary(
        baseline_result=baseline_result,
        best_constrained_result=best_constrained_result,
        best_lambda_spec=best_lambda_spec,
    )
    artifact_paths = save_spectral_constraint_artifacts(
        output_dir=root_dir,
        baseline_result=baseline_result,
        constrained_results=constrained_results,
        constrained_trial_df=constrained_trial_df,
        comparison_summary_df=comparison_summary_df,
        best_constrained_result=best_constrained_result,
        best_lambda_spec=best_lambda_spec,
    )

    return SpectralConstraintResult(
        baseline_result=baseline_result,
        best_constrained_result=best_constrained_result,
        constrained_trial_df=constrained_trial_df,
        comparison_summary_df=comparison_summary_df,
        artifact_paths=artifact_paths,
        lambda_spec_values=lambda_spec_values,
        best_lambda_spec=best_lambda_spec,
    )


def run_frame_structured_experiment(
    config: ExperimentConfig | None = None,
    tx_init: torch.Tensor | None = None,
    rx_init: torch.Tensor | None = None,
) -> FrameStructuredStudyResult:
    from reporting import (
        build_frame_pilot_summary_df,
        build_frame_resource_df,
        plot_frame_resource_map,
        plot_pilot_estimation_summary,
        save_frame_structured_artifacts,
    )

    config = default_config() if config is None else config
    if not config.frame_structure_enabled:
        raise ValueError("run_frame_structured_experiment requires frame_structure_enabled=True.")

    experiment_result = run_full_experiment(config=config, tx_init=tx_init, rx_init=rx_init)
    resource_df = build_frame_resource_df(config)
    pilot_summary_df = build_frame_pilot_summary_df(experiment_result.ber_df)
    artifact_paths = dict(experiment_result.artifact_paths)
    artifact_paths.update(
        save_frame_structured_artifacts(
            output_dir=config.output_dir,
            resource_df=resource_df,
            pilot_summary_df=pilot_summary_df,
        )
    )
    resource_plot = plot_frame_resource_map(config)
    if resource_plot is not None:
        artifact_paths["resource_plot"] = resource_plot
    pilot_plot = plot_pilot_estimation_summary(config, pilot_summary_df)
    if pilot_plot is not None:
        artifact_paths["pilot_plot"] = pilot_plot
    experiment_result.artifact_paths = artifact_paths
    return FrameStructuredStudyResult(
        experiment_result=experiment_result,
        resource_df=resource_df,
        pilot_summary_df=pilot_summary_df,
        artifact_paths=artifact_paths,
    )
