from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channel import propagate
from comm_core import (
    StageSpec,
    file_sha256,
    load_stage1_checkpoint,
    refresh_output_dir,
    set_seed,
    snapshot_stage1_checkpoint,
    stage2_checkpoint_preflight,
)
from receiver import (
    CfoEstimatorMmseReceiver,
    _complex_phase_features,
    _sample_training_cfo_values,
    _sample_training_ebn0_values,
    _stage2_estimator_max_abs_cfo,
    _stage2_training_abs_cfo_scale,
    decode_symbols,
    slice_symbols,
)
from run_stage2_blind_mmse import build_stage2_blind_mmse_config
from transmitter import sample_training_symbols, transmit_symbols


OUTPUT_ROOT = SCRIPT_DIR / "outputs"
OUTPUT_SUBDIR = "tmp_pilot_sideinfo_thresholds_n45_r9"
REFRESH_OUTPUT_DIR = True

FAMILY_ORDER = ("scaled", "noisy", "pilot_grid_quantized")
SCALED_ALPHA_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)
NOISY_SIGMA_REL_GRID = (1.00, 0.50, 0.25, 0.10, 0.00)
PILOT_GRID_POINTS = (9, 17, 33, 81)

DIAGNOSTIC_BER_BLOCKS = 4096
DIAGNOSTIC_SNR_BLOCKS = 2048
DIAGNOSTIC_ESTIMATION_BLOCKS = 1536
DIAGNOSTIC_BATCH_SIZE = 256
SCATTER_KEEP_PER_METHOD = 1800


@dataclass(frozen=True)
class SideInfoSpec:
    family: str
    family_label: str
    strength_value: float | int
    strength_label: str
    strength_slug: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    family_label: str
    mode: str
    strength_label: str
    strength_slug: str
    strength_value: float | int | None
    residual_model: nn.Module | None = None


def build_stage2_pilot_sideinfo_config(
    *,
    output_root: Path = OUTPUT_ROOT,
    output_subdir: str = OUTPUT_SUBDIR,
    refresh_output_dir: bool = REFRESH_OUTPUT_DIR,
) -> object:
    return replace(
        build_stage2_blind_mmse_config(
            output_root=output_root,
            output_subdir=output_subdir,
            refresh_output_dir=refresh_output_dir,
        ),
        terminal_progress_enabled=True,
        stage2_sideinfo_enabled=False,
        stage2_sideinfo_mode="NONE",
        stage2_sideinfo_scale=0.0,
        stage2_sideinfo_residual_enabled=False,
    )


def _family_specs() -> list[SideInfoSpec]:
    specs: list[SideInfoSpec] = []
    for alpha in SCALED_ALPHA_GRID:
        specs.append(
            SideInfoSpec(
                family="scaled",
                family_label="Scaled Truth",
                strength_value=float(alpha),
                strength_label=f"alpha={alpha:.2f}",
                strength_slug=f"alpha_{alpha:.2f}".replace(".", "p"),
            )
        )
    for sigma_rel in NOISY_SIGMA_REL_GRID:
        specs.append(
            SideInfoSpec(
                family="noisy",
                family_label="Noisy Truth",
                strength_value=float(sigma_rel),
                strength_label=f"sigma_rel={sigma_rel:.2f}",
                strength_slug=f"sigma_{sigma_rel:.2f}".replace(".", "p"),
            )
        )
    for grid_points in PILOT_GRID_POINTS:
        specs.append(
            SideInfoSpec(
                family="pilot_grid_quantized",
                family_label="Pilot-Grid Quantized",
                strength_value=int(grid_points),
                strength_label=f"grid={int(grid_points)}",
                strength_slug=f"grid_{int(grid_points)}",
            )
        )
    return specs


def _load_stage1_bases(config: object) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    checkpoint_path = Path(config.stage2_checkpoint_path)
    print(f"[SideInfo] Loading Stage 1 checkpoint -> {checkpoint_path}", flush=True)
    checkpoint = load_stage1_checkpoint(checkpoint_path, device=config.device, require_v2=True)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config:
        for field_name in ("modulation", "M", "K", "N", "N_data", "N_pilots", "N_guard", "frame_structure_enabled"):
            saved_value = checkpoint_config.get(field_name)
            current_value = getattr(config, field_name)
            if saved_value != current_value:
                raise ValueError(
                    f"Stage 1 checkpoint mismatch for {field_name}: saved={saved_value!r}, current={current_value!r}."
                )
    return checkpoint, checkpoint["learned_tx"].to(config.device), checkpoint["learned_rx"].to(config.device)


def _prepare_checkpoint_snapshot(config: object, checkpoint: dict[str, object]) -> Path:
    preflight = stage2_checkpoint_preflight(config, checkpoint)
    checkpoint_source_path = Path(config.stage2_checkpoint_path)
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
    return checkpoint_snapshot_path


class SideInfoResidualReceiver(CfoEstimatorMmseReceiver):
    def __init__(
        self,
        config: object,
        tx_basis: torch.Tensor,
        rx_basis: torch.Tensor,
        *,
        residual_max_abs_eps: float,
    ) -> None:
        super().__init__(
            tx_basis=tx_basis,
            rx_basis=rx_basis,
            num_symbols=config.N,
            bits_per_symbol=config.bits_per_symbol,
            architecture="LOCAL",
            channels=config.stage2_local_channels,
            hidden_multiplier=config.stage2_hidden_multiplier,
            kernel_size=config.stage2_local_kernel_size,
            max_abs_eps=_stage2_estimator_max_abs_cfo(config),
        )
        self.residual_max_abs_eps = float(residual_max_abs_eps)
        self.sideinfo_head = nn.Sequential(
            nn.Linear(self.channels + 1, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, 1),
        )
        nn.init.zeros_(self.sideinfo_head[-1].weight)
        nn.init.zeros_(self.sideinfo_head[-1].bias)

    def estimate_eps_with_sideinfo(
        self,
        z0: torch.Tensor,
        eps_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _complex_phase_features(z0)
        hidden = F.gelu(self.conv1(features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = F.gelu(self.conv3(hidden))
        pooled = torch.mean(hidden, dim=-1)
        head_input = torch.cat([pooled, eps_coarse.to(torch.float32).unsqueeze(-1)], dim=-1)
        residual = torch.tanh(self.sideinfo_head(head_input).squeeze(-1)) * self.residual_max_abs_eps
        eps_hat = torch.clamp(
            eps_coarse.to(torch.float32) + residual,
            min=-self.max_abs_eps,
            max=self.max_abs_eps,
        )
        return eps_hat, residual

    def forward_with_sideinfo(
        self,
        z0: torch.Tensor,
        eps_coarse: torch.Tensor,
        ebn0_db_values: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eps_hat, residual = self.estimate_eps_with_sideinfo(z0, eps_coarse)
        corrected_symbols = self._mmse_solve(z0, eps_hat, ebn0_db_values)
        aux = {
            "eps_coarse": eps_coarse.to(torch.float32),
            "eps_hat": eps_hat.to(torch.float32),
            "eps_residual": residual.to(torch.float32),
        }
        return corrected_symbols, aux


def _build_mmse_solver(config: object, tx_basis: torch.Tensor, rx_basis: torch.Tensor) -> CfoEstimatorMmseReceiver:
    return CfoEstimatorMmseReceiver(
        tx_basis=tx_basis,
        rx_basis=rx_basis,
        num_symbols=config.N,
        bits_per_symbol=config.bits_per_symbol,
        architecture="LOCAL",
        channels=config.stage2_local_channels,
        hidden_multiplier=config.stage2_hidden_multiplier,
        kernel_size=config.stage2_local_kernel_size,
        max_abs_eps=_stage2_estimator_max_abs_cfo(config),
    ).to(config.device)


def _pilot_grid_quantized_eps(
    eps_values: torch.Tensor,
    grid_points: int,
    config: object,
) -> torch.Tensor:
    max_abs = max(abs(float(v)) for v in config.ber_eval_cfo)
    grid = torch.linspace(-max_abs, max_abs, int(grid_points), device=eps_values.device, dtype=torch.float32)
    distance = torch.abs(eps_values.to(torch.float32).unsqueeze(-1) - grid.unsqueeze(0))
    return grid[torch.argmin(distance, dim=-1)]


def _noisy_eps(
    eps_values: torch.Tensor,
    sigma_rel: float,
    *,
    generator: torch.Generator,
    max_abs_eps: float,
) -> torch.Tensor:
    if sigma_rel <= 0.0:
        return eps_values.to(torch.float32)
    noise_std = float(sigma_rel) * torch.abs(eps_values.to(torch.float32))
    noise = torch.randn(
        eps_values.shape,
        generator=generator,
        device=eps_values.device,
        dtype=torch.float32,
    )
    return torch.clamp(eps_values.to(torch.float32) + noise_std * noise, min=-max_abs_eps, max=max_abs_eps)


def _coarse_eps_from_spec(
    config: object,
    spec: SideInfoSpec,
    eps_values: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    max_abs_eps = _stage2_estimator_max_abs_cfo(config)
    if spec.family == "scaled":
        return torch.clamp(float(spec.strength_value) * eps_values.to(torch.float32), min=-max_abs_eps, max=max_abs_eps)
    if spec.family == "noisy":
        if generator is None:
            raise ValueError("Noisy side information requires a random generator.")
        return _noisy_eps(
            eps_values,
            float(spec.strength_value),
            generator=generator,
            max_abs_eps=max_abs_eps,
        )
    if spec.family == "pilot_grid_quantized":
        return _pilot_grid_quantized_eps(eps_values, int(spec.strength_value), config)
    raise ValueError(f"Unsupported side information family: {spec.family}")


def _sample_stage_batch(
    config: object,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    stage: StageSpec,
    batch_size: int,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bits, symbols = sample_training_symbols(batch_size, config)
    eps_values = _sample_training_cfo_values(config, stage, batch_size, deterministic=deterministic)
    ebn0_values = _sample_training_ebn0_values(config, batch_size, deterministic=deterministic)
    tx_signal = transmit_symbols(symbols, tx_basis)
    rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_values)
    z0 = decode_symbols(rx_signal, rx_basis)
    return bits, symbols, z0, eps_values.to(torch.float32)


def _train_residual_model(
    config: object,
    spec: SideInfoSpec,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    *,
    seed_offset: int,
) -> tuple[SideInfoResidualReceiver, pd.DataFrame, dict[str, float | str]]:
    set_seed(config.base_seed + seed_offset)
    model = SideInfoResidualReceiver(
        config=config,
        tx_basis=tx_basis,
        rx_basis=rx_basis,
        residual_max_abs_eps=float(config.stage_c_cfo),
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.stage2_nonlinear_only_learning_rate)
    stage = StageSpec("ResidualSideInfo", config.stage2_nonlinear_only_epochs, config.stage_c_cfo, True)
    eps_scale = _stage2_training_abs_cfo_scale(config)
    _, _, val_z0, val_eps = _sample_stage_batch(
        config,
        tx_basis,
        rx_basis,
        stage,
        min(config.train_symbol_batch_size, 256),
        deterministic=True,
    )
    history: list[dict[str, float | str | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0

    print(
        f"[SideInfo] residual train | {spec.family_label} {spec.strength_label} | epochs={config.stage2_nonlinear_only_epochs}",
        flush=True,
    )
    for epoch in range(config.stage2_nonlinear_only_epochs):
        _, _, train_z0, train_eps = _sample_stage_batch(
            config,
            tx_basis,
            rx_basis,
            stage,
            config.train_symbol_batch_size,
            deterministic=False,
        )
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(config.base_seed + 100_000 + seed_offset * 10_000 + epoch)
        train_coarse = _coarse_eps_from_spec(config, spec, train_eps, generator=train_generator)
        eps_hat, residual = model.estimate_eps_with_sideinfo(train_z0, train_coarse)
        train_loss = torch.mean(((eps_hat - train_eps) / eps_scale) ** 2)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        should_log = (
            epoch == 0
            or epoch + 1 == config.stage2_nonlinear_only_epochs
            or (epoch + 1) % config.history_log_interval == 0
        )
        if not should_log:
            continue

        with torch.no_grad():
            val_generator = torch.Generator(device="cpu")
            val_generator.manual_seed(config.base_seed + 200_000 + seed_offset)
            val_coarse = _coarse_eps_from_spec(config, spec, val_eps, generator=val_generator)
            val_eps_hat, val_residual = model.estimate_eps_with_sideinfo(val_z0, val_coarse)
            val_loss = torch.mean(((val_eps_hat - val_eps) / eps_scale) ** 2)
            val_mae = torch.mean(torch.abs(val_eps_hat - val_eps)).item()
            row = {
                "family": spec.family,
                "family_label": spec.family_label,
                "strength_label": spec.strength_label,
                "epoch": epoch + 1,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "val_mae": float(val_mae),
                "coarse_mae": float(torch.mean(torch.abs(val_coarse - val_eps)).item()),
                "residual_mean_abs": float(torch.mean(torch.abs(val_residual)).item()),
                "eps_hat_mean": float(torch.mean(val_eps_hat).item()),
            }
            history.append(row)
            print(
                (
                    f"[SideInfo] residual {spec.family_label} {spec.strength_label} "
                    f"{epoch + 1}/{config.stage2_nonlinear_only_epochs} "
                    f"| train={row['train_loss']:.4e} val={row['val_loss']:.4e} "
                    f"| val_mae={row['val_mae']:.4e} coarse_mae={row['coarse_mae']:.4e}"
                ),
                flush=True,
            )
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_epoch = epoch + 1
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No best residual state captured for {spec.family_label} {spec.strength_label}.")
    model.load_state_dict(best_state)
    model.eval()
    summary = {
        "family": spec.family,
        "family_label": spec.family_label,
        "strength_label": spec.strength_label,
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    return model, pd.DataFrame(history), summary


def _build_method_specs(
    specs: list[SideInfoSpec],
    residual_models: dict[tuple[str, str], nn.Module],
) -> list[MethodSpec]:
    methods = [MethodSpec(name="Learned", family="baseline", family_label="Baseline", mode="baseline", strength_label="", strength_slug="", strength_value=None)]
    for spec in specs:
        methods.append(
            MethodSpec(
                name=f"DirectCoarseMMSE[{spec.family_label}; {spec.strength_label}]",
                family=spec.family,
                family_label=spec.family_label,
                mode="direct",
                strength_label=spec.strength_label,
                strength_slug=spec.strength_slug,
                strength_value=spec.strength_value,
                residual_model=None,
            )
        )
        methods.append(
            MethodSpec(
                name=f"ResidualCoarseMMSE[{spec.family_label}; {spec.strength_label}]",
                family=spec.family,
                family_label=spec.family_label,
                mode="residual",
                strength_label=spec.strength_label,
                strength_slug=spec.strength_slug,
                strength_value=spec.strength_value,
                residual_model=residual_models[(spec.family, spec.strength_slug)],
            )
        )
    return methods


def _run_method_forward(
    config: object,
    solver: CfoEstimatorMmseReceiver,
    method: MethodSpec,
    spec_lookup: dict[tuple[str, str], SideInfoSpec],
    z0: torch.Tensor,
    eps_values: torch.Tensor,
    ebn0_db: float,
    *,
    batch_seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if method.mode == "baseline":
        return z0, {}
    spec = spec_lookup[(method.family, method.strength_slug)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(batch_seed)
    eps_coarse = _coarse_eps_from_spec(config, spec, eps_values, generator=generator)
    if method.mode == "direct":
        corrected = solver._mmse_solve(z0, eps_coarse, ebn0_db)
        return corrected, {"eps_coarse": eps_coarse, "eps_hat": eps_coarse}
    assert method.residual_model is not None
    corrected, aux = method.residual_model.forward_with_sideinfo(z0, eps_coarse, ebn0_db)
    return corrected, aux


def _family_plot_color(method: MethodSpec) -> str:
    family_colors = {
        "scaled": "#C8553D",
        "noisy": "#2E6F95",
        "pilot_grid_quantized": "#4C956C",
        "baseline": "#222222",
    }
    return family_colors.get(method.family, "#555555")


def _mode_linestyle(method: MethodSpec) -> str:
    if method.mode == "direct":
        return "--"
    if method.mode == "residual":
        return "-"
    return "-"


def _evaluate_ber_vs_cfo(
    config: object,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    solver: CfoEstimatorMmseReceiver,
    methods: list[MethodSpec],
    spec_lookup: dict[tuple[str, str], SideInfoSpec],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    cfo_points = np.asarray(config.ber_eval_cfo, dtype=float)
    for cfo_idx, eps in enumerate(cfo_points):
        remaining = DIAGNOSTIC_BER_BLOCKS
        total_errors = {method.name: 0.0 for method in methods}
        total_bits = 0
        while remaining > 0:
            batch_size = min(DIAGNOSTIC_BATCH_SIZE, remaining)
            bits, symbols = sample_training_symbols(batch_size, config)
            eps_values = torch.full((batch_size,), float(eps), device=config.device, dtype=torch.float32)
            tx_signal = transmit_symbols(symbols, tx_basis)
            rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=config.eval_ebn0_db)
            z0 = decode_symbols(rx_signal, rx_basis)
            total_bits += int(bits.numel())
            for method_idx, method in enumerate(methods):
                corrected, _ = _run_method_forward(
                    config,
                    solver,
                    method,
                    spec_lookup,
                    z0,
                    eps_values,
                    config.eval_ebn0_db,
                    batch_seed=config.base_seed + 10_000 * cfo_idx + 97 * method_idx + remaining,
                )
                bits_hat, _ = slice_symbols(corrected, config.modulation)
                total_errors[method.name] += float(torch.sum(bits_hat != bits).item())
            remaining -= batch_size
        for method in methods:
            rows.append(
                {
                    "method": method.name,
                    "family": method.family,
                    "family_label": method.family_label,
                    "mode": method.mode,
                    "strength_label": method.strength_label,
                    "eps_true": float(eps),
                    "abs_eps": float(abs(eps)),
                    "ebn0_db": float(config.eval_ebn0_db),
                    "ber": float(total_errors[method.name] / max(1, total_bits)),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_ber_vs_snr(
    config: object,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    solver: CfoEstimatorMmseReceiver,
    methods: list[MethodSpec],
    spec_lookup: dict[tuple[str, str], SideInfoSpec],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for eps in (0.05, 0.10):
        for snr_idx, ebn0_db in enumerate(np.asarray(config.snr_sweep_ebn0_db_grid, dtype=float)):
            remaining = DIAGNOSTIC_SNR_BLOCKS
            total_errors = {method.name: 0.0 for method in methods}
            total_bits = 0
            while remaining > 0:
                batch_size = min(DIAGNOSTIC_BATCH_SIZE, remaining)
                bits, symbols = sample_training_symbols(batch_size, config)
                eps_values = torch.full((batch_size,), float(eps), device=config.device, dtype=torch.float32)
                tx_signal = transmit_symbols(symbols, tx_basis)
                rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=float(ebn0_db))
                z0 = decode_symbols(rx_signal, rx_basis)
                total_bits += int(bits.numel())
                for method_idx, method in enumerate(methods):
                    corrected, _ = _run_method_forward(
                        config,
                        solver,
                        method,
                        spec_lookup,
                        z0,
                        eps_values,
                        float(ebn0_db),
                        batch_seed=config.base_seed + 300_000 + int(round(100 * eps)) * 10_000 + 103 * snr_idx + method_idx,
                    )
                    bits_hat, _ = slice_symbols(corrected, config.modulation)
                    total_errors[method.name] += float(torch.sum(bits_hat != bits).item())
                remaining -= batch_size
            for method in methods:
                rows.append(
                    {
                        "method": method.name,
                        "family": method.family,
                        "family_label": method.family_label,
                        "mode": method.mode,
                        "strength_label": method.strength_label,
                        "eps_true": float(eps),
                        "ebn0_db": float(ebn0_db),
                        "ber": float(total_errors[method.name] / max(1, total_bits)),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_estimation_metrics(
    config: object,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    solver: CfoEstimatorMmseReceiver,
    methods: list[MethodSpec],
    spec_lookup: dict[tuple[str, str], SideInfoSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    scatter_rng = np.random.default_rng(config.base_seed + 404)
    cfo_points = np.asarray(config.ber_eval_cfo, dtype=float)
    for cfo_idx, eps in enumerate(cfo_points):
        remaining = DIAGNOSTIC_ESTIMATION_BLOCKS
        while remaining > 0:
            batch_size = min(DIAGNOSTIC_BATCH_SIZE, remaining)
            _, symbols = sample_training_symbols(batch_size, config)
            eps_values = torch.full((batch_size,), float(eps), device=config.device, dtype=torch.float32)
            tx_signal = transmit_symbols(symbols, tx_basis)
            rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=config.eval_ebn0_db)
            z0 = decode_symbols(rx_signal, rx_basis)
            for method_idx, method in enumerate(methods):
                corrected, aux = _run_method_forward(
                    config,
                    solver,
                    method,
                    spec_lookup,
                    z0,
                    eps_values,
                    config.eval_ebn0_db,
                    batch_seed=config.base_seed + 500_000 + 211 * cfo_idx + method_idx,
                )
                _ = corrected
                if method.mode == "baseline":
                    continue
                eps_coarse = aux["eps_coarse"].detach().cpu().numpy()
                eps_hat = aux["eps_hat"].detach().cpu().numpy()
                keep_mask = scatter_rng.random(batch_size) < min(1.0, SCATTER_KEEP_PER_METHOD / (len(cfo_points) * DIAGNOSTIC_ESTIMATION_BLOCKS))
                for idx in range(batch_size):
                    rows.append(
                        {
                            "method": method.name,
                            "family": method.family,
                            "family_label": method.family_label,
                            "mode": method.mode,
                            "strength_label": method.strength_label,
                            "eps_true": float(eps),
                            "eps_coarse": float(eps_coarse[idx]),
                            "eps_hat": float(eps_hat[idx]),
                            "scatter_keep": int(keep_mask[idx]),
                        }
                    )
            remaining -= batch_size
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, pd.DataFrame(), pd.DataFrame()
    grid_df = (
        raw_df.groupby(["method", "family", "family_label", "mode", "strength_label", "eps_true"], as_index=False)
        .agg(
            eps_coarse_mean=("eps_coarse", "mean"),
            eps_coarse_std=("eps_coarse", "std"),
            eps_hat_mean=("eps_hat", "mean"),
            eps_hat_std=("eps_hat", "std"),
        )
    )
    summary_rows: list[dict[str, float | str]] = []
    for method_name, method_df in raw_df.groupby("method", sort=False):
        eps_true = method_df["eps_true"].to_numpy(dtype=float)
        eps_coarse = method_df["eps_coarse"].to_numpy(dtype=float)
        eps_hat = method_df["eps_hat"].to_numpy(dtype=float)
        in_range_mask = np.abs(eps_true) <= 0.10 + 1e-12
        sign_mask = np.abs(eps_true) > 0.03
        zero_mask = np.isclose(eps_true, 0.0)
        mean_curve = grid_df[grid_df["method"] == method_name]
        slope, intercept = np.polyfit(mean_curve["eps_true"], mean_curve["eps_hat_mean"], deg=1)
        summary_rows.append(
            {
                "method": method_name,
                "family": str(method_df["family"].iloc[0]),
                "family_label": str(method_df["family_label"].iloc[0]),
                "mode": str(method_df["mode"].iloc[0]),
                "strength_label": str(method_df["strength_label"].iloc[0]),
                "mae_coarse_full": float(np.mean(np.abs(eps_coarse - eps_true))),
                "mae_hat_full": float(np.mean(np.abs(eps_hat - eps_true))),
                "mae_hat_in_range_le_0p10": float(np.mean(np.abs(eps_hat[in_range_mask] - eps_true[in_range_mask]))),
                "sign_accuracy_abs_delta_gt_0p03": float(np.mean(np.sign(eps_hat[sign_mask]) == np.sign(eps_true[sign_mask]))),
                "zero_bias_mean": float(np.mean(eps_hat[zero_mask])),
                "zero_bias_std": float(np.std(eps_hat[zero_mask])),
                "mean_curve_slope": float(slope),
                "mean_curve_intercept": float(intercept),
                "eps_hat_min": float(np.min(eps_hat)),
                "eps_hat_max": float(np.max(eps_hat)),
            }
        )
    return raw_df, grid_df, pd.DataFrame(summary_rows)


def _build_ranking_table(
    ber_df: pd.DataFrame,
    estimation_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    ber_focus = (
        ber_df[ber_df["abs_eps"].isin([0.05, 0.10])]
        .groupby(["method", "abs_eps"], as_index=False)["ber"]
        .mean()
        .pivot(index="method", columns="abs_eps", values="ber")
        .reset_index()
        .rename(columns={0.05: "ber_abs_0p05", 0.10: "ber_abs_0p10"})
    )
    ranking_df = estimation_summary_df.merge(ber_focus, on="method", how="left")
    ranking_df = ranking_df.sort_values(
        ["ber_abs_0p10", "mae_hat_full", "sign_accuracy_abs_delta_gt_0p03"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    return ranking_df


def _save_family_ber_plot(output_dir: Path, family: str, family_label: str, ber_df: pd.DataFrame) -> Path | None:
    family_df = ber_df[(ber_df["family"] == family) | (ber_df["family"] == "baseline")]
    if family_df.empty:
        return None
    fig = plt.figure(figsize=(8.2, 5.8), dpi=140)
    for method_name, method_df in family_df.groupby("method", sort=False):
        family_value = str(method_df["family"].iloc[0])
        mode = str(method_df["mode"].iloc[0])
        label = "Learned" if family_value == "baseline" else f"{mode.title()} {method_df['strength_label'].iloc[0]}"
        color = _family_plot_color(
            MethodSpec(
                name=method_name,
                family=family_value,
                family_label=str(method_df["family_label"].iloc[0]),
                mode=mode,
                strength_label=str(method_df["strength_label"].iloc[0]),
                strength_slug="",
                strength_value=None,
            )
        )
        linestyle = "-" if family_value == "baseline" else ("--" if mode == "direct" else "-")
        linewidth = 2.4 if family_value == "baseline" else 1.8
        plt.plot(method_df["eps_true"], method_df["ber"], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    plt.yscale("log")
    plt.xlabel("Residual CFO")
    plt.ylabel("BER")
    plt.title(f"{family_label}: BER vs CFO")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = output_dir / f"{family}_ber_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_family_snr_plot(output_dir: Path, family: str, family_label: str, ber_snr_df: pd.DataFrame) -> Path | None:
    family_df = ber_snr_df[(ber_snr_df["family"] == family) | (ber_snr_df["family"] == "baseline")]
    if family_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=140, sharey=True)
    for ax, eps in zip(axes, (0.05, 0.10), strict=True):
        eps_df = family_df[np.isclose(family_df["eps_true"], eps)]
        for method_name, method_df in eps_df.groupby("method", sort=False):
            family_value = str(method_df["family"].iloc[0])
            mode = str(method_df["mode"].iloc[0])
            label = "Learned" if family_value == "baseline" else f"{mode.title()} {method_df['strength_label'].iloc[0]}"
            color = _family_plot_color(
                MethodSpec(
                    name=method_name,
                    family=family_value,
                    family_label=str(method_df["family_label"].iloc[0]),
                    mode=mode,
                    strength_label=str(method_df["strength_label"].iloc[0]),
                    strength_slug="",
                    strength_value=None,
                )
            )
            linestyle = "-" if family_value == "baseline" else ("--" if mode == "direct" else "-")
            linewidth = 2.4 if family_value == "baseline" else 1.8
            ax.plot(method_df["ebn0_db"], method_df["ber"], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        ax.set_title(rf"$\delta = {eps:.2f}$")
        ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.grid(True, alpha=0.25)
        ax.set_yscale("log")
    axes[0].set_ylabel("BER")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=7, ncol=2)
    fig.suptitle(f"{family_label}: BER vs SNR")
    path = output_dir / f"{family}_ber_vs_snr.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_family_estimation_plot(
    output_dir: Path,
    family: str,
    family_label: str,
    grid_df: pd.DataFrame,
    *,
    value_column: str,
    title_suffix: str,
    ylabel: str,
    file_suffix: str,
) -> Path | None:
    family_df = grid_df[grid_df["family"] == family]
    if family_df.empty:
        return None
    fig = plt.figure(figsize=(8.2, 5.6), dpi=140)
    for method_name, method_df in family_df.groupby("method", sort=False):
        mode = str(method_df["mode"].iloc[0])
        label = f"{mode.title()} {method_df['strength_label'].iloc[0]}"
        linestyle = "--" if mode == "direct" else "-"
        linewidth = 1.8
        plt.plot(method_df["eps_true"], method_df[value_column], linestyle=linestyle, linewidth=linewidth, label=label)
    plt.plot(grid_df["eps_true"].unique(), grid_df["eps_true"].unique(), linestyle=":", color="#222222", linewidth=1.2, label="ideal")
    plt.xlabel("True residual CFO")
    plt.ylabel(ylabel)
    plt.title(f"{family_label}: {title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = output_dir / f"{family}_{file_suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_scatter_overview(output_dir: Path, raw_df: pd.DataFrame) -> Path | None:
    scatter_df = raw_df[raw_df["scatter_keep"] == 1]
    if scatter_df.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=140, sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        family_df = scatter_df[scatter_df["family"] == family]
        for mode, color in (("direct", "#8C8C8C"), ("residual", "#C8553D")):
            mode_df = family_df[family_df["mode"] == mode]
            ax.scatter(mode_df["eps_true"], mode_df["eps_hat"], s=8, alpha=0.10, color=color, label=mode.title())
        xline = np.linspace(-0.20, 0.20, 41)
        ax.plot(xline, xline, linestyle="--", linewidth=1.1, color="#222222")
        ax.set_title(str(family_df["family_label"].iloc[0]) if not family_df.empty else family)
        ax.set_xlabel("True residual CFO")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Estimated residual CFO")
    axes[-1].legend()
    path = output_dir / "delta_hat_scatter_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(
    output_dir: Path,
    config: object,
    checkpoint_snapshot_path: Path,
    ranking_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> Path:
    lines = [
        "# Temporary Pilot-Side-Information Threshold Study",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Stage 1 checkpoint: `{config.stage2_checkpoint_path}`",
        f"- Stage 1 snapshot: `{checkpoint_snapshot_path}`",
        f"- Training support: `|delta| <= {config.stage_c_cfo:.2f}`",
        f"- Evaluation support: `[{float(np.min(config.ber_eval_cfo)):+.2f}, {float(np.max(config.ber_eval_cfo)):+.2f}]`",
        f"- Residual learner epochs: `{config.stage2_nonlinear_only_epochs}`",
        f"- BER blocks per CFO point: `{DIAGNOSTIC_BER_BLOCKS}`",
        f"- BER blocks per SNR point: `{DIAGNOSTIC_SNR_BLOCKS}`",
        "",
        "## Notes",
        "",
        "- The learned baseline uses the shared complex slicer with no Stage 2 correction.",
        "- Direct coarse-MMSE uses the exact post-V MMSE solve driven by the coarse CFO hint.",
        "- Residual coarse-MMSE trains only a supervised CFO residual predictor on top of that hint.",
        "- In this payload-region Stage 2 package, the transmitted learned waveform does not carry operational pilot symbols.",
        "- For that reason, the `Pilot-Grid Quantized` family is an explicit pilot-side-information surrogate: true CFO quantized onto pilot-style grids.",
        "",
        "## Ranking",
        "",
        "```text",
        ranking_df.head(18).to_string(index=False),
        "```",
        "",
        "## Figures",
        "",
    ]
    for key, path in sorted(plot_paths.items()):
        lines.append(f"- `{key}`: [{path.name}]({path.name})")
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def run_stage2_pilot_sideinfo_thresholds(config: object | None = None) -> dict[str, object]:
    config = build_stage2_pilot_sideinfo_config() if config is None else config
    set_seed(config.base_seed)
    if config.refresh_output_dir:
        refresh_output_dir(config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, learned_tx, learned_rx = _load_stage1_bases(config)
    checkpoint_snapshot_path = _prepare_checkpoint_snapshot(config, checkpoint)
    solver = _build_mmse_solver(config, learned_tx, learned_rx)
    family_specs = _family_specs()
    spec_lookup = {(spec.family, spec.strength_slug): spec for spec in family_specs}

    training_history_frames: list[pd.DataFrame] = []
    training_summary_rows: list[dict[str, float | str]] = []
    residual_models: dict[tuple[str, str], nn.Module] = {}
    for seed_offset, spec in enumerate(family_specs, start=1):
        model, history_df, summary = _train_residual_model(
            config,
            spec,
            learned_tx,
            learned_rx,
            seed_offset=seed_offset,
        )
        residual_models[(spec.family, spec.strength_slug)] = model
        training_history_frames.append(history_df)
        training_summary_rows.append(summary)

    methods = _build_method_specs(family_specs, residual_models)
    ber_df = _evaluate_ber_vs_cfo(config, learned_tx, learned_rx, solver, methods, spec_lookup)
    ber_snr_df = _evaluate_ber_vs_snr(config, learned_tx, learned_rx, solver, methods, spec_lookup)
    estimation_raw_df, estimation_grid_df, estimation_summary_df = _evaluate_estimation_metrics(
        config,
        learned_tx,
        learned_rx,
        solver,
        methods,
        spec_lookup,
    )
    ranking_df = _build_ranking_table(ber_df[ber_df["family"] != "baseline"], estimation_summary_df)
    training_history_df = pd.concat(training_history_frames, ignore_index=True) if training_history_frames else pd.DataFrame()
    training_summary_df = pd.DataFrame(training_summary_rows)

    ber_focus_df = (
        ber_df[ber_df["abs_eps"].isin([0.05, 0.10])]
        .groupby(["method", "family", "family_label", "mode", "strength_label", "abs_eps"], as_index=False)["ber"]
        .mean()
        .pivot(index=["method", "family", "family_label", "mode", "strength_label"], columns="abs_eps", values="ber")
        .reset_index()
        .rename(columns={0.05: "ber_abs_0p05", 0.10: "ber_abs_0p10"})
    )

    output_dir = Path(config.output_dir)
    training_history_df.to_csv(output_dir / "training_history.csv", index=False)
    training_summary_df.to_csv(output_dir / "training_summary.csv", index=False)
    ber_df.to_csv(output_dir / "ber_vs_cfo.csv", index=False)
    ber_snr_df.to_csv(output_dir / "ber_vs_snr.csv", index=False)
    ber_focus_df.to_csv(output_dir / "ber_focus_summary.csv", index=False)
    estimation_raw_df.to_csv(output_dir / "estimation_samples.csv", index=False)
    estimation_grid_df.to_csv(output_dir / "estimation_grid_summary.csv", index=False)
    estimation_summary_df.to_csv(output_dir / "estimation_summary.csv", index=False)
    ranking_df.to_csv(output_dir / "ranking_summary.csv", index=False)

    plot_paths: dict[str, Path] = {}
    for family in FAMILY_ORDER:
        family_label = next(spec.family_label for spec in family_specs if spec.family == family)
        ber_plot = _save_family_ber_plot(output_dir, family, family_label, ber_df)
        if ber_plot is not None:
            plot_paths[f"{family}_ber_vs_cfo"] = ber_plot
        snr_plot = _save_family_snr_plot(output_dir, family, family_label, ber_snr_df)
        if snr_plot is not None:
            plot_paths[f"{family}_ber_vs_snr"] = snr_plot
        coarse_plot = _save_family_estimation_plot(
            output_dir,
            family,
            family_label,
            estimation_grid_df,
            value_column="eps_coarse_mean",
            title_suffix=r"coarse CFO versus true CFO",
            ylabel=r"Mean coarse CFO",
            file_suffix="eps_coarse_vs_true",
        )
        if coarse_plot is not None:
            plot_paths[f"{family}_eps_coarse_vs_true"] = coarse_plot
        hat_plot = _save_family_estimation_plot(
            output_dir,
            family,
            family_label,
            estimation_grid_df,
            value_column="eps_hat_mean",
            title_suffix=r"estimated CFO versus true CFO",
            ylabel=r"Mean estimated CFO",
            file_suffix="eps_hat_vs_true",
        )
        if hat_plot is not None:
            plot_paths[f"{family}_eps_hat_vs_true"] = hat_plot
    scatter_overview = _save_scatter_overview(output_dir, estimation_raw_df)
    if scatter_overview is not None:
        plot_paths["scatter_overview"] = scatter_overview

    report_path = _write_report(output_dir, config, checkpoint_snapshot_path, ranking_df, plot_paths)

    print("\n=== top ranking rows ===")
    print(ranking_df.head(12).to_string(index=False))
    print("\n=== report ===")
    print(report_path)
    return {
        "training_history_df": training_history_df,
        "training_summary_df": training_summary_df,
        "ber_df": ber_df,
        "ber_snr_df": ber_snr_df,
        "ber_focus_df": ber_focus_df,
        "estimation_raw_df": estimation_raw_df,
        "estimation_grid_df": estimation_grid_df,
        "estimation_summary_df": estimation_summary_df,
        "ranking_df": ranking_df,
        "plot_paths": plot_paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_stage2_pilot_sideinfo_thresholds()
