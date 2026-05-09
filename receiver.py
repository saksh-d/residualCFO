from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from channel import apply_residual_cfo, effective_operator, propagate
from comm_core import ExperimentConfig, StageSpec, TrainingResult, normalize_columns, set_seed, stage_specs
from transmitter import (
    _QAM16_DECISION_BITS,
    contiguous_band_bins_with_guard,
    frame_resource_layout,
    make_initial_transceiver,
    make_tx_support_frame,
    pilot_reference_symbol,
    qam16_from_bits,
    qpsk_from_bits,
    sample_training_symbols,
    transmit_symbols,
)


@dataclass
class EvaluationScheme:
    tx_basis: torch.Tensor
    rx_basis: torch.Tensor
    nonlinear_receiver: nn.Module | None = None


@dataclass
class Stage2ReceiverTrainingResult:
    receiver: nn.Module
    learned_tx: torch.Tensor
    learned_rx: torch.Tensor
    history_df: pd.DataFrame
    stage_summary_df: pd.DataFrame


def _progress_enabled(config: ExperimentConfig) -> bool:
    return bool(getattr(config, "terminal_progress_enabled", True))


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _log_progress(config: ExperimentConfig, message: str) -> None:
    if _progress_enabled(config):
        print(message, flush=True)


def qpsk_slicer(symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits_i = (torch.real(symbols) >= 0).to(torch.int64)
    bits_q = (torch.imag(symbols) >= 0).to(torch.int64)
    bits = torch.stack([bits_i, bits_q], dim=-1)
    return bits, qpsk_from_bits(bits)


def qam16_slicer(symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = math.sqrt(10.0)
    re = torch.real(symbols) * scale
    im = torch.imag(symbols) * scale
    table = _QAM16_DECISION_BITS.to(symbols.device)

    def _decision_levels(values: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(values, dtype=torch.int64)
        idx = torch.where(values < -2.0, torch.zeros_like(idx), idx)
        idx = torch.where((values >= -2.0) & (values < 0.0), torch.ones_like(idx), idx)
        idx = torch.where((values >= 0.0) & (values < 2.0), torch.full_like(idx, 2), idx)
        idx = torch.where(values >= 2.0, torch.full_like(idx, 3), idx)
        return idx

    idx_i = _decision_levels(re)
    idx_q = _decision_levels(im)
    bits_i = table[idx_i]
    bits_q = table[idx_q]
    bits = torch.cat([bits_i, bits_q], dim=-1)
    return bits, qam16_from_bits(bits)


def slice_symbols(symbols: torch.Tensor, modulation: str) -> tuple[torch.Tensor, torch.Tensor]:
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_slicer(symbols)
    if modulation == "16QAM":
        return qam16_slicer(symbols)
    raise ValueError(f"Unsupported modulation: {modulation}")


def qam16_labels_from_bits(bits: torch.Tensor) -> torch.Tensor:
    table = _QAM16_DECISION_BITS.to(bits.device)
    table_view = table.view(*([1] * (bits.ndim - 1)), 4, 2)
    i_pairs = bits[..., None, :2].to(torch.int64)
    q_pairs = bits[..., None, 2:].to(torch.int64)
    idx_i = torch.argmax(torch.all(i_pairs == table_view, dim=-1).to(torch.int64), dim=-1)
    idx_q = torch.argmax(torch.all(q_pairs == table_view, dim=-1).to(torch.int64), dim=-1)
    return 4 * idx_i + idx_q


def qam16_bits_from_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.int64)
    idx_i = torch.div(labels, 4, rounding_mode="floor")
    idx_q = torch.remainder(labels, 4)
    table = _QAM16_DECISION_BITS.to(labels.device)
    return torch.cat([table[idx_i], table[idx_q]], dim=-1)


def qam16_symbols_from_labels(labels: torch.Tensor, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    constellation = qam16_constellation_points(labels.device, dtype=dtype)
    return constellation[labels.to(torch.int64)]


def qam16_constellation_points(device: str | torch.device, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    labels = torch.arange(16, device=device, dtype=torch.int64)
    return qam16_from_bits(qam16_bits_from_labels(labels)).to(dtype)


def qam16_symbol_logits_to_bit_logits(logits: torch.Tensor) -> torch.Tensor:
    label_bits = qam16_bits_from_labels(torch.arange(16, device=logits.device, dtype=torch.int64))
    bit_logits: list[torch.Tensor] = []
    for bit_idx in range(4):
        mask_one = label_bits[:, bit_idx].to(torch.bool)
        mask_zero = ~mask_one
        one_score = torch.logsumexp(logits[..., mask_one], dim=-1)
        zero_score = torch.logsumexp(logits[..., mask_zero], dim=-1)
        bit_logits.append(one_score - zero_score)
    return torch.stack(bit_logits, dim=-1).to(torch.float32)


def complex_to_real_features(symbols: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.real(symbols), torch.imag(symbols)], dim=-1).to(torch.float32)


def real_features_to_complex(features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] % 2 != 0:
        raise ValueError(f"Expected an even feature dimension, got {features.shape[-1]}.")
    half = features.shape[-1] // 2
    return torch.complex(features[..., :half], features[..., half:]).to(torch.complex64)


def decode_symbols(y: torch.Tensor, rx_basis: torch.Tensor) -> torch.Tensor:
    return y @ rx_basis.T


def decode_symbols_batched(y: torch.Tensor, rx_basis: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2:
        return decode_symbols(y, rx_basis)
    return torch.einsum("bgm,nm->bgn", y, rx_basis)


def normalized_symbol_mse(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    denom = torch.mean(torch.abs(ref) ** 2).real.clamp_min(1e-12)
    return torch.mean(torch.abs(est - ref) ** 2).real / denom


def _weighted_loss_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    normed_weights = weights.to(values.device, dtype=torch.float32)
    normed_weights = normed_weights / normed_weights.sum().clamp_min(1e-12)
    return torch.sum(values * normed_weights)


def identity_loss(operator: torch.Tensor) -> torch.Tensor:
    N = operator.shape[-1]
    eye = torch.eye(N, device=operator.device, dtype=operator.dtype).unsqueeze(0)
    return torch.mean(torch.sum(torch.abs(operator - eye) ** 2, dim=(-2, -1)).real / N)


def offdiag_energy_per_operator(operator: torch.Tensor, distance_weight_power: float = 0.0) -> torch.Tensor:
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    N = operator.shape[-1]
    eye = torch.eye(N, device=operator.device, dtype=operator.dtype).unsqueeze(0)
    offdiag = operator * (1.0 - eye)
    if distance_weight_power > 0.0:
        idx = torch.arange(N, device=operator.device, dtype=torch.float32)
        distance = torch.abs(idx[:, None] - idx[None, :]).clamp_min(1.0)
        weights = torch.pow(distance, -distance_weight_power).to(operator.dtype)
        weights = weights * (1.0 - eye[0])
        mean_weight = weights[weights != 0].real.mean().clamp_min(1e-12)
        offdiag = offdiag * (weights / mean_weight).unsqueeze(0)
    return torch.sum(torch.abs(offdiag) ** 2, dim=(-2, -1)).real / N


def offdiag_energy_loss(
    operator: torch.Tensor,
    distance_weight_power: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _weighted_loss_mean(offdiag_energy_per_operator(operator, distance_weight_power), weights)


def diag_uniformity_per_operator(operator: torch.Tensor) -> torch.Tensor:
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    diag = torch.diagonal(operator, dim1=-2, dim2=-1)
    diag_mean = diag.mean(dim=-1, keepdim=True)
    return torch.sum(torch.abs(diag - diag_mean) ** 2, dim=-1).real / operator.shape[-1]


def diag_uniformity_loss(operator: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return _weighted_loss_mean(diag_uniformity_per_operator(operator), weights)


def nearest_neighbor_energy_per_operator(operator: torch.Tensor, max_distance: int = 2) -> torch.Tensor:
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    n = operator.shape[-1]
    eye = torch.eye(n, device=operator.device, dtype=operator.dtype).unsqueeze(0)
    idx = torch.arange(n, device=operator.device)
    distance = torch.abs(idx[:, None] - idx[None, :])
    mask = ((distance >= 1) & (distance <= max_distance)).to(operator.dtype).unsqueeze(0)
    local = operator * mask * (1.0 - eye)
    return torch.sum(torch.abs(local) ** 2, dim=(-2, -1)).real / n


def nearest_neighbor_energy_loss(
    operator: torch.Tensor,
    max_distance: int = 2,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _weighted_loss_mean(nearest_neighbor_energy_per_operator(operator, max_distance), weights)


def receiver_fro_loss(rx_basis: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(rx_basis) ** 2).real


def spectral_mask_loss(tx_basis: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    if not config.spectral_constraint_enabled or config.lambda_spec <= 0.0:
        return torch.zeros((), device=tx_basis.device, dtype=torch.float32)

    if config.payload_region_enabled:
        allowed_bins = frame_resource_layout(config).payload_bin_indices
    else:
        allowed_bins = contiguous_band_bins_with_guard(config.M, config.N, guard_bins=config.spectral_guard_bins)
    allowed_mask = torch.zeros(config.M, device=tx_basis.device, dtype=torch.bool)
    allowed_mask[torch.tensor(allowed_bins, device=tx_basis.device, dtype=torch.long)] = True
    spectrum = torch.fft.fft(tx_basis, n=config.M, dim=0)
    total_energy = torch.sum(torch.abs(spectrum) ** 2).real.clamp_min(1e-12)
    outside_energy = torch.sum(torch.abs(spectrum[~allowed_mask]) ** 2).real
    return (outside_energy / total_energy).to(torch.float32)


def offdiag_leakage_ratio(operator: torch.Tensor) -> torch.Tensor:
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    eye = torch.eye(operator.shape[-1], device=operator.device, dtype=operator.dtype).unsqueeze(0)
    total = torch.sum(torch.abs(operator) ** 2, dim=(-2, -1)).real.clamp_min(1e-12)
    diag = torch.sum(torch.abs(operator * eye) ** 2, dim=(-2, -1)).real
    return ((total - diag).clamp_min(0.0) / total).detach()


def diagonal_magnitudes(operator: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.diagonal(operator, dim1=-2, dim2=-1))


def nearest_neighbor_leakage_ratio(operator: torch.Tensor, max_distance: int = 1) -> torch.Tensor:
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    n = operator.shape[-1]
    idx = torch.arange(n, device=operator.device)
    distance = torch.abs(idx[:, None] - idx[None, :])
    mask = ((distance >= 1) & (distance <= max_distance)).to(operator.dtype).unsqueeze(0)
    total = torch.sum(torch.abs(operator) ** 2, dim=(-2, -1)).real.clamp_min(1e-12)
    local_energy = torch.sum(torch.abs(operator * mask) ** 2, dim=(-2, -1)).real
    return (local_energy / total).detach()


def _stage_cfo_support(
    config: ExperimentConfig,
    stage: StageSpec,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not stage.use_cfo_losses or stage.cfo_span <= 0.0 or not config.train_cfo_support:
        return None
    support = torch.tensor(config.train_cfo_support, device=device, dtype=torch.float32)
    weights = torch.tensor(config.train_cfo_support_weights, device=device, dtype=torch.float32)
    mask = torch.abs(support) <= float(stage.cfo_span) + 1e-12
    if not torch.any(mask):
        return None
    support = support[mask]
    weights = weights[mask]
    weights = weights / weights.sum().clamp_min(1e-12)
    order = torch.argsort(support)
    return support[order], weights[order]


def _sample_training_cfo_values(
    config: ExperimentConfig,
    stage: StageSpec,
    batch_size: int,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    if not stage.use_cfo_losses or stage.cfo_span <= 0.0:
        return torch.zeros(batch_size, device=config.device, dtype=torch.float32)

    stage_support = _stage_cfo_support(config, stage, device=config.device)
    if stage_support is not None:
        support, weights = stage_support
        if deterministic:
            repeats = math.ceil(batch_size / support.numel())
            return support.repeat(repeats)[:batch_size]
        sample_idx = torch.multinomial(weights, batch_size, replacement=True)
        return support[sample_idx]

    grid_points = int(config.train_cfo_grid_points)
    if grid_points <= 0:
        return torch.empty(batch_size, device=config.device).uniform_(-stage.cfo_span, stage.cfo_span)

    grid_points = max(3, grid_points)
    if grid_points % 2 == 0:
        grid_points += 1
    base_grid = torch.linspace(-stage.cfo_span, stage.cfo_span, grid_points, device=config.device, dtype=torch.float32)
    repeats = math.ceil(batch_size / grid_points)
    tiled = base_grid.repeat(repeats)[:batch_size]
    if deterministic:
        return tiled
    return tiled[torch.randperm(batch_size, device=config.device)]


def _sample_training_ebn0_values(
    config: ExperimentConfig,
    batch_size: int,
    *,
    deterministic: bool = False,
) -> float | torch.Tensor:
    low = config.train_ebn0_db if config.train_ebn0_db_min is None else float(config.train_ebn0_db_min)
    high = config.train_ebn0_db if config.train_ebn0_db_max is None else float(config.train_ebn0_db_max)
    low, high = sorted((low, high))
    if math.isclose(low, high):
        return float(low)

    if deterministic:
        points = min(5, batch_size)
        base_grid = torch.linspace(low, high, points, device=config.device, dtype=torch.float32)
        repeats = math.ceil(batch_size / points)
        return base_grid.repeat(repeats)[:batch_size]
    return torch.empty(batch_size, device=config.device, dtype=torch.float32).uniform_(low, high)


def _stage_validation_grid(config: ExperimentConfig, stage: StageSpec) -> torch.Tensor:
    if not stage.use_cfo_losses:
        return torch.tensor([0.0], device=config.device, dtype=torch.float32)
    stage_support = _stage_cfo_support(config, stage, device=config.device)
    if stage_support is not None:
        return stage_support[0]
    return torch.linspace(
        -stage.cfo_span,
        stage.cfo_span,
        config.stage_validation_points,
        device=config.device,
        dtype=torch.float32,
    )


def _stage_operator_loss_support(
    config: ExperimentConfig,
    stage: StageSpec,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    stage_support = _stage_cfo_support(config, stage, device=config.device)
    if stage_support is not None:
        return stage_support
    return _sample_training_cfo_values(config, stage, config.operator_batch_size), None


def _build_validation_batch(
    config: ExperimentConfig,
    stage: StageSpec,
) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]:
    batch_size = min(config.train_symbol_batch_size, 128)
    _, symbols = sample_training_symbols(batch_size, config)
    eps = _sample_training_cfo_values(config, stage, batch_size, deterministic=True)
    ebn0_values = _sample_training_ebn0_values(config, batch_size, deterministic=True)
    return symbols, eps, ebn0_values


def _symbol_loss_for_batch(
    config: ExperimentConfig,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    symbols: torch.Tensor,
    eps_values: torch.Tensor,
    ebn0_db_values: float | torch.Tensor | None = None,
) -> torch.Tensor:
    tx_signal = transmit_symbols(symbols, tx_basis)
    if ebn0_db_values is None:
        ebn0_db_values = config.train_ebn0_db
    rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_db_values)
    shat = decode_symbols(rx_signal, rx_basis)
    return normalized_symbol_mse(shat, symbols)


def estimate_residual_cfo_from_pilots(
    config: ExperimentConfig,
    rx_signal: torch.Tensor,
    rx_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not config.frame_structure_enabled:
        batch_size = rx_signal.shape[0]
        zeros = torch.zeros(batch_size, device=rx_signal.device, dtype=torch.float32)
        return zeros, zeros

    layout = frame_resource_layout(config)
    pilot_idx = torch.tensor(layout.pilot_stream_indices, device=rx_signal.device, dtype=torch.long)
    pilot_ref = pilot_reference_symbol(config.modulation, device=config.device)
    cfo_grid = torch.tensor(config.pilot_estimator_cfo_grid, device=rx_signal.device, dtype=torch.float32)
    sample_idx = torch.arange(config.M, device=rx_signal.device, dtype=torch.float32)
    derotation = torch.exp(
        -1j * 2.0 * math.pi * cfo_grid[:, None] * sample_idx[None, :] / float(config.M)
    ).to(torch.complex64)
    compensated = rx_signal[:, None, :] * derotation[None, :, :]
    shat_grid = decode_symbols_batched(compensated, rx_basis)
    pilot_target = torch.full(
        (rx_signal.shape[0], cfo_grid.numel(), pilot_idx.numel()),
        pilot_ref,
        device=rx_signal.device,
        dtype=torch.complex64,
    )
    pilot_mse = torch.mean(torch.abs(shat_grid[:, :, pilot_idx] - pilot_target) ** 2, dim=-1).real
    best_idx = torch.argmin(pilot_mse, dim=1)
    batch_idx = torch.arange(rx_signal.shape[0], device=rx_signal.device, dtype=torch.long)
    return cfo_grid[best_idx], pilot_mse[batch_idx, best_idx]


def _stage_total_loss(
    config: ExperimentConfig,
    tx_basis: torch.Tensor,
    operator_clean: torch.Tensor,
    operator_cfo: torch.Tensor,
    rx_basis: torch.Tensor,
    symbol_loss: torch.Tensor,
    use_cfo_losses: bool,
    operator_loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_0 = identity_loss(operator_clean)
    loss_v = receiver_fro_loss(rx_basis)
    loss_spec = spectral_mask_loss(tx_basis, config)
    if use_cfo_losses:
        loss_off = offdiag_energy_loss(
            operator_cfo,
            config.offdiag_distance_weight_power,
            weights=operator_loss_weights,
        )
        loss_diag = diag_uniformity_loss(operator_cfo, weights=operator_loss_weights)
        loss_nn = nearest_neighbor_energy_loss(
            operator_cfo,
            max_distance=config.local_leakage_max_distance,
            weights=operator_loss_weights,
        )
    else:
        zero = torch.zeros((), device=operator_clean.device, dtype=torch.float32)
        loss_off = zero
        loss_diag = zero
        loss_nn = zero
    total = (
        config.lambda_0 * loss_0
        + config.lambda_1 * loss_off
        + config.lambda_2 * loss_diag
        + config.lambda_3 * loss_v
        + config.lambda_sym * symbol_loss
        + config.lambda_nn * loss_nn
        + config.lambda_spec * loss_spec
    )
    return total, {
        "L0": loss_0,
        "Loff": loss_off,
        "Ldiag": loss_diag,
        "LV": loss_v,
        "Lsym": symbol_loss,
        "Lnn": loss_nn,
        "Lspec": loss_spec,
    }


class RedundantLinearWaveform(nn.Module):
    def __init__(
        self,
        tx_init: torch.Tensor,
        rx_init: torch.Tensor,
        tx_support_frame: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.tx_re = nn.Parameter(torch.real(tx_init).to(torch.float32))
        self.tx_im = nn.Parameter(torch.imag(tx_init).to(torch.float32))
        self.rx_re = nn.Parameter(torch.real(rx_init).to(torch.float32))
        self.rx_im = nn.Parameter(torch.imag(rx_init).to(torch.float32))
        if tx_support_frame is None:
            self.register_buffer("tx_support_re", torch.empty(0, dtype=torch.float32))
            self.register_buffer("tx_support_im", torch.empty(0, dtype=torch.float32))
        else:
            self.register_buffer("tx_support_re", torch.real(tx_support_frame).to(torch.float32))
            self.register_buffer("tx_support_im", torch.imag(tx_support_frame).to(torch.float32))

    def tx_raw(self) -> torch.Tensor:
        return torch.complex(self.tx_re, self.tx_im)

    def tx_support_frame(self) -> torch.Tensor | None:
        if self.tx_support_re.numel() == 0:
            return None
        return torch.complex(self.tx_support_re, self.tx_support_im)

    def rx_raw(self) -> torch.Tensor:
        return torch.complex(self.rx_re, self.rx_im)

    def tx_basis(self) -> torch.Tensor:
        tx_raw = self.tx_raw()
        support_frame = self.tx_support_frame()
        if support_frame is not None:
            tx_raw = support_frame @ tx_raw
        return normalize_columns(tx_raw)

    def rx_basis(self) -> torch.Tensor:
        return self.rx_raw()


class DenseSymbolResidualReceiver(nn.Module):
    def __init__(
        self,
        num_symbols: int,
        hidden_multiplier: int = 4,
        residual_scale_init: float = 0.1,
        use_residual_logit_head: bool = False,
    ) -> None:
        super().__init__()
        if num_symbols <= 0:
            raise ValueError("num_symbols must be positive.")
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least 1.")
        if residual_scale_init <= 0.0:
            raise ValueError("residual_scale_init must be positive.")

        self.num_symbols = int(num_symbols)
        self.feature_dim = 2 * self.num_symbols
        self.hidden_dim = self.feature_dim * int(hidden_multiplier)
        self.use_residual_logit_head = bool(use_residual_logit_head)
        self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.feature_dim)
        self.classifier = nn.Linear(self.hidden_dim, 16 * self.num_symbols) if self.use_residual_logit_head else None
        self.log_residual_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale_init), dtype=torch.float32)))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        if self.classifier is not None:
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def residual_scale(self) -> torch.Tensor:
        return torch.exp(self.log_residual_scale)

    def constellation_logits(self, corrected_symbols: torch.Tensor) -> torch.Tensor:
        constellation = qam16_constellation_points(corrected_symbols.device, dtype=corrected_symbols.dtype)
        distances = torch.abs(corrected_symbols.unsqueeze(-1) - constellation.view(1, 1, -1)) ** 2
        return (-distances.real).to(torch.float32)

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        u0 = complex_to_real_features(z0)
        hidden = F.gelu(self.fc1(u0))
        delta_u = self.fc2(hidden)
        u_hat = u0 + self.residual_scale() * delta_u
        corrected_symbols = real_features_to_complex(u_hat)
        logits = self.constellation_logits(corrected_symbols)
        if self.classifier is not None:
            logits = logits + self.classifier(hidden).reshape(-1, self.num_symbols, 16)
        aux = {
            "eps_hat": torch.zeros(z0.shape[0], device=z0.device, dtype=torch.float32),
            "bit_logits": qam16_symbol_logits_to_bit_logits(logits),
            "cancellation_scale": torch.zeros((), device=z0.device, dtype=torch.float32),
        }
        return corrected_symbols, logits, aux


def stage1_geometry_logits(symbols: torch.Tensor) -> torch.Tensor:
    constellation = qam16_constellation_points(symbols.device, dtype=symbols.dtype)
    distances = torch.abs(symbols.unsqueeze(-1) - constellation.view(1, 1, -1)) ** 2
    return (-distances.real).to(torch.float32)


def logits_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def nearest_constellation_residual(symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_logits = stage1_geometry_logits(symbols)
    labels = torch.argmax(base_logits, dim=-1)
    constellation = qam16_constellation_points(symbols.device, dtype=symbols.dtype)
    nearest = constellation[labels]
    residual = symbols - nearest
    return residual, logits_margin(base_logits)


class LocalSymbolResidualReceiver(nn.Module):
    def __init__(
        self,
        num_symbols: int,
        channels: int = 32,
        kernel_size: int = 5,
        residual_scale_init: float = 0.1,
        cancellation_scale_init: float = 0.1,
        use_confidence_features: bool = True,
        use_symbol_correction_head: bool = True,
        use_residual_logit_head: bool = False,
        max_abs_eps: float = 0.1,
    ) -> None:
        super().__init__()
        if num_symbols <= 0:
            raise ValueError("num_symbols must be positive.")
        if channels < 1:
            raise ValueError("channels must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if residual_scale_init <= 0.0:
            raise ValueError("residual_scale_init must be positive.")
        if cancellation_scale_init <= 0.0:
            raise ValueError("cancellation_scale_init must be positive.")
        if max_abs_eps <= 0.0:
            raise ValueError("max_abs_eps must be positive.")

        self.num_symbols = int(num_symbols)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.use_confidence_features = bool(use_confidence_features)
        self.use_symbol_correction_head = bool(use_symbol_correction_head)
        self.use_residual_logit_head = bool(use_residual_logit_head)
        self.feature_channels = 5 if self.use_confidence_features else 4
        self.max_abs_eps = float(max_abs_eps)
        padding = kernel_size // 2

        self.cfo_head = nn.Sequential(
            nn.Linear(self.feature_channels, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, 1),
        )
        self.pass1_conv1 = nn.Conv1d(self.feature_channels + 1, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass1_conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass1_conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass1_logit_head = nn.Conv1d(self.channels, 16, kernel_size=1) if self.use_residual_logit_head else None

        self.cancel_conv1 = nn.Conv1d(4, self.channels, kernel_size=kernel_size, padding=padding)
        self.cancel_conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.cancel_conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.cancel_head = nn.Conv1d(self.channels, 2, kernel_size=1)

        self.pass2_conv1 = nn.Conv1d(self.feature_channels + 4, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass2_conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass2_conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
        self.pass2_logit_head = nn.Conv1d(self.channels, 16, kernel_size=1) if self.use_residual_logit_head else None
        self.symbol_head = nn.Conv1d(self.channels, 2, kernel_size=1) if self.use_symbol_correction_head else None
        self.log_residual_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale_init), dtype=torch.float32)))
        self.log_cancellation_scale = nn.Parameter(
            torch.log(torch.tensor(float(cancellation_scale_init), dtype=torch.float32))
        )
        nn.init.zeros_(self.cfo_head[-1].weight)
        nn.init.zeros_(self.cfo_head[-1].bias)
        if self.pass1_logit_head is not None:
            nn.init.zeros_(self.pass1_logit_head.weight)
            nn.init.zeros_(self.pass1_logit_head.bias)
        nn.init.zeros_(self.cancel_head.weight)
        nn.init.zeros_(self.cancel_head.bias)
        if self.pass2_logit_head is not None:
            nn.init.zeros_(self.pass2_logit_head.weight)
            nn.init.zeros_(self.pass2_logit_head.bias)
        if self.symbol_head is not None:
            nn.init.zeros_(self.symbol_head.weight)
            nn.init.zeros_(self.symbol_head.bias)

    def residual_scale(self) -> torch.Tensor:
        return torch.exp(self.log_residual_scale)

    def cancellation_scale(self) -> torch.Tensor:
        return torch.exp(self.log_cancellation_scale)

    def _conv_stack(
        self,
        x: torch.Tensor,
        conv1: nn.Conv1d,
        conv2: nn.Conv1d,
        conv3: nn.Conv1d,
    ) -> torch.Tensor:
        hidden = F.gelu(conv1(x))
        hidden = F.gelu(conv2(hidden))
        hidden = F.gelu(conv3(hidden))
        return hidden

    def _feature_map(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, margin = nearest_constellation_residual(z0)
        feature_list = [
            torch.real(z0),
            torch.imag(z0),
            torch.real(residual),
            torch.imag(residual),
        ]
        if self.use_confidence_features:
            feature_list.append(margin)
        features = torch.stack(feature_list, dim=1).to(torch.float32)
        base_logits = stage1_geometry_logits(z0)
        return features, base_logits, margin.to(torch.float32)

    def _estimate_eps(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(features, dim=-1)
        return torch.tanh(self.cfo_head(pooled).squeeze(-1)) * self.max_abs_eps

    def _eps_channel(self, eps_hat: torch.Tensor) -> torch.Tensor:
        return eps_hat.view(-1, 1, 1).expand(-1, 1, self.num_symbols)

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features0, base_logits0, _ = self._feature_map(z0)
        eps_hat = self._estimate_eps(features0)
        eps_channel = self._eps_channel(eps_hat)
        pass1_hidden = self._conv_stack(
            torch.cat([features0, eps_channel], dim=1),
            self.pass1_conv1,
            self.pass1_conv2,
            self.pass1_conv3,
        )
        logits1 = base_logits0
        if self.pass1_logit_head is not None:
            logits1 = logits1 + self.pass1_logit_head(pass1_hidden).permute(0, 2, 1).contiguous()
        labels1 = torch.argmax(logits1, dim=-1)
        tentative1 = qam16_symbols_from_labels(labels1, dtype=z0.dtype)
        pass1_margin = logits_margin(logits1).to(torch.float32)

        cancel_features = torch.stack(
            [
                torch.real(tentative1),
                torch.imag(tentative1),
                pass1_margin,
                eps_channel[:, 0, :],
            ],
            dim=1,
        ).to(torch.float32)
        cancel_hidden = self._conv_stack(
            cancel_features,
            self.cancel_conv1,
            self.cancel_conv2,
            self.cancel_conv3,
        )
        cancel_delta = self.cancel_head(cancel_hidden)
        cancel_complex = torch.complex(cancel_delta[:, 0, :], cancel_delta[:, 1, :]).to(torch.complex64)
        z1 = z0 - self.cancellation_scale() * cancel_complex

        features1, _, _ = self._feature_map(z1)
        pass2_input = torch.cat(
            [
                features1,
                eps_channel,
                torch.real(tentative1).unsqueeze(1).to(torch.float32),
                torch.imag(tentative1).unsqueeze(1).to(torch.float32),
                pass1_margin.unsqueeze(1),
            ],
            dim=1,
        )
        hidden = self._conv_stack(
            pass2_input,
            self.pass2_conv1,
            self.pass2_conv2,
            self.pass2_conv3,
        )

        if self.symbol_head is None:
            corrected_symbols = z1
        else:
            delta = self.symbol_head(hidden)
            delta_u = torch.cat([delta[:, 0, :], delta[:, 1, :]], dim=-1)
            corrected_symbols = real_features_to_complex(
                complex_to_real_features(z1) + self.residual_scale() * delta_u
            )
        logits = stage1_geometry_logits(corrected_symbols)
        if self.pass2_logit_head is not None:
            logits = logits + self.pass2_logit_head(hidden).permute(0, 2, 1).contiguous()
        aux = {
            "eps_hat": eps_hat.to(torch.float32),
            "bit_logits": qam16_symbol_logits_to_bit_logits(logits),
            "cancellation_scale": self.cancellation_scale().to(torch.float32),
            "pass1_logits": logits1,
        }
        return corrected_symbols, logits, aux


class JointStage2Model(nn.Module):
    def __init__(
        self,
        tx_init: torch.Tensor,
        rx_init: torch.Tensor,
        nonlinear_receiver: nn.Module,
        tx_support_frame: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.waveform = RedundantLinearWaveform(tx_init, rx_init, tx_support_frame=tx_support_frame)
        self.nonlinear_receiver = nonlinear_receiver

    def tx_basis(self) -> torch.Tensor:
        return self.waveform.tx_basis()

    def rx_basis(self) -> torch.Tensor:
        return self.waveform.rx_basis()

    def forward(
        self,
        symbols: torch.Tensor,
        eps_values: torch.Tensor,
        config: ExperimentConfig,
        ebn0_db_values: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tx_basis = self.tx_basis()
        rx_basis = self.rx_basis()
        tx_signal = transmit_symbols(symbols, tx_basis)
        rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_db_values)
        z0 = decode_symbols(rx_signal, rx_basis)
        corrected_symbols, logits, aux = self.nonlinear_receiver(z0)
        return corrected_symbols, logits, z0, aux


def train_feasibility_model(
    config: ExperimentConfig,
    tx_init: torch.Tensor | None = None,
    rx_init: torch.Tensor | None = None,
) -> TrainingResult:
    set_seed(config.base_seed)
    if tx_init is None or rx_init is None:
        tx_init, rx_init = make_initial_transceiver(config)

    tx_support_frame = make_tx_support_frame(config)
    model = RedundantLinearWaveform(tx_init, rx_init, tx_support_frame=tx_support_frame).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    stage_summaries: list[dict[str, float | int | str | bool | None]] = []
    global_epoch = 0
    t0 = perf_counter()
    stage_failed = False
    failed_stage: str | None = None
    stop_reason = "Completed all stages."

    _log_progress(
        config,
        (
            f"[Stage1] Training start | modulation={config.modulation} M={config.M} K={config.K} "
            f"N={config.N} R={config.redundancy_dimensions} | epochs="
            f"{config.stage_a_epochs + config.stage_b_epochs + config.stage_c_epochs}"
        ),
    )

    for stage in stage_specs(config):
        best: dict[str, object] | None = None
        validation_eps = _stage_validation_grid(config, stage)
        validation_weights = None
        validation_support = _stage_cfo_support(config, stage, device=config.device)
        if validation_support is not None:
            validation_weights = validation_support[1]
        val_symbols, val_eps_sym, val_ebn0_sym = _build_validation_batch(config, stage)
        _log_progress(
            config,
            f"[Stage1] {stage.name} start | epochs={stage.epochs} cfo_span={stage.cfo_span:.2f}",
        )

        for stage_epoch in range(stage.epochs):
            global_epoch += 1
            tx_basis = model.tx_basis()
            rx_basis = model.rx_basis()
            eps_batch, operator_loss_weights = _stage_operator_loss_support(config, stage)
            eps_sym = _sample_training_cfo_values(config, stage, config.train_symbol_batch_size)
            ebn0_sym = _sample_training_ebn0_values(config, config.train_symbol_batch_size)

            operator_clean = effective_operator(tx_basis, rx_basis, torch.tensor([0.0], device=config.device))
            operator_cfo = effective_operator(tx_basis, rx_basis, eps_batch)
            _, symbols = sample_training_symbols(config.train_symbol_batch_size, config)
            loss_sym = _symbol_loss_for_batch(config, tx_basis, rx_basis, symbols, eps_sym, ebn0_sym)
            total_loss, components = _stage_total_loss(
                config=config,
                tx_basis=tx_basis,
                operator_clean=operator_clean,
                operator_cfo=operator_cfo,
                rx_basis=rx_basis,
                symbol_loss=loss_sym,
                use_cfo_losses=stage.use_cfo_losses,
                operator_loss_weights=operator_loss_weights,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            should_log = (
                stage_epoch == 0
                or stage_epoch + 1 == stage.epochs
                or (stage_epoch + 1) % config.history_log_interval == 0
            )
            if not should_log:
                continue

            with torch.no_grad():
                tx_val = model.tx_basis()
                rx_val = model.rx_basis()
                operator_clean_val = effective_operator(tx_val, rx_val, torch.tensor([0.0], device=config.device))
                operator_stage_val = effective_operator(tx_val, rx_val, validation_eps)
                val_symbol_loss = _symbol_loss_for_batch(
                    config,
                    tx_val,
                    rx_val,
                    val_symbols,
                    val_eps_sym,
                    val_ebn0_sym,
                )
                val_total, val_components = _stage_total_loss(
                    config=config,
                    tx_basis=tx_val,
                    operator_clean=operator_clean_val,
                    operator_cfo=operator_stage_val,
                    rx_basis=rx_val,
                    symbol_loss=val_symbol_loss,
                    use_cfo_losses=stage.use_cfo_losses,
                    operator_loss_weights=validation_weights,
                )
                clean_leakage = offdiag_leakage_ratio(operator_clean_val)[0].item()
                history.append(
                    {
                        "stage": stage.name,
                        "global_epoch": global_epoch,
                        "stage_epoch": stage_epoch + 1,
                        "modulation": config.modulation,
                        "N": config.N,
                        "M": config.M,
                        "train_ebn0_db": config.train_ebn0_db,
                        "eval_ebn0_db": config.eval_ebn0_db,
                        "cfo_span": stage.cfo_span,
                        "train_total": total_loss.item(),
                        "train_L0": components["L0"].item(),
                        "train_Loff": components["Loff"].item(),
                        "train_Ldiag": components["Ldiag"].item(),
                        "train_LV": components["LV"].item(),
                        "train_Lsym": components["Lsym"].item(),
                        "train_Lnn": components["Lnn"].item(),
                        "train_Lspec": components["Lspec"].item(),
                        "val_total": val_total.item(),
                        "val_L0": val_components["L0"].item(),
                        "val_Loff": val_components["Loff"].item(),
                        "val_Ldiag": val_components["Ldiag"].item(),
                        "val_LV": val_components["LV"].item(),
                        "val_Lsym": val_components["Lsym"].item(),
                        "val_Lnn": val_components["Lnn"].item(),
                        "val_Lspec": val_components["Lspec"].item(),
                        "clean_offdiag_leakage": clean_leakage,
                        "rx_fro_norm_sq": receiver_fro_loss(rx_val).item(),
                        "elapsed_s": perf_counter() - t0,
                    }
                )
                elapsed_s = float(history[-1]["elapsed_s"])
                progress_pct = 100.0 * float(stage_epoch + 1) / float(stage.epochs)
                _log_progress(
                    config,
                    (
                        f"[Stage1] {stage.name} {stage_epoch + 1}/{stage.epochs} ({progress_pct:5.1f}%) "
                        f"| global={global_epoch} | train={total_loss.item():.4e} val={val_total.item():.4e} "
                        f"| clean_offdiag={clean_leakage:.4e} | elapsed={_format_duration(elapsed_s)}"
                    ),
                )
                if best is None or history[-1]["val_total"] < best["score"]:
                    best = {
                        "score": history[-1]["val_total"],
                        "global_epoch": global_epoch,
                        "stage_epoch": stage_epoch + 1,
                        "state": copy.deepcopy(model.state_dict()),
                    }

        assert best is not None
        model.load_state_dict(best["state"])
        with torch.no_grad():
            tx_final = model.tx_basis().detach().clone()
            rx_final = model.rx_basis().detach().clone()
            operator_clean = effective_operator(tx_final, rx_final, torch.tensor([0.0], device=config.device))
            operator_stage = effective_operator(tx_final, rx_final, validation_eps)
            stage_symbol_loss = _symbol_loss_for_batch(
                config,
                tx_final,
                rx_final,
                val_symbols,
                val_eps_sym,
                val_ebn0_sym,
            ).item()
            stage_spectral_loss = spectral_mask_loss(tx_final, config).item()
            clean_identity = identity_loss(operator_clean).item()
            clean_leakage = offdiag_leakage_ratio(operator_clean)[0].item()
            stage_record: dict[str, float | int | str | bool | None] = {
                "stage": stage.name,
                "modulation": config.modulation,
                "N": config.N,
                "M": config.M,
                "train_ebn0_db": config.train_ebn0_db,
                "eval_ebn0_db": config.eval_ebn0_db,
                "epochs": stage.epochs,
                "cfo_span": stage.cfo_span,
                "best_global_epoch": int(best["global_epoch"]),
                "best_stage_epoch": int(best["stage_epoch"]),
                "clean_identity_loss": clean_identity,
                "clean_offdiag_leakage": clean_leakage,
                "mean_validation_offdiag": offdiag_leakage_ratio(operator_stage).mean().item(),
                "mean_validation_diag_loss": diag_uniformity_loss(operator_stage).item(),
                "mean_validation_local_leakage": nearest_neighbor_leakage_ratio(
                    operator_stage,
                    max_distance=config.local_leakage_max_distance,
                ).mean().item(),
                "validation_symbol_loss": stage_symbol_loss,
                "validation_spectral_loss": stage_spectral_loss,
                "receiver_fro_norm_sq": receiver_fro_loss(rx_final).item(),
                "stage_failed": False,
                "stop_reason": "",
            }
            if stage.name == "Stage A":
                if clean_identity > config.stage_a_identity_tol or clean_leakage > config.stage_a_leakage_tol:
                    stage_failed = True
                    failed_stage = stage.name
                    stop_reason = (
                        "Stage A failed to produce a strongly diagonal clean operator; "
                        "CFO stages were not run."
                    )
                    stage_record["stage_failed"] = True
                    stage_record["stop_reason"] = stop_reason
                    stage_summaries.append(stage_record)
                    _log_progress(
                        config,
                        (
                            f"[Stage1] {stage.name} failed gate | clean_identity={clean_identity:.4e} "
                            f"clean_leakage={clean_leakage:.4e}"
                        ),
                    )
                    break
            stage_record["stop_reason"] = "Completed."
            stage_summaries.append(stage_record)
            _log_progress(
                config,
                (
                    f"[Stage1] {stage.name} complete | best_epoch={int(best['stage_epoch'])}/{stage.epochs} "
                    f"| clean_identity={clean_identity:.4e} clean_leakage={clean_leakage:.4e} "
                    f"| sym_loss={stage_symbol_loss:.4e}"
                ),
            )

        if stage_failed:
            break

    _log_progress(
        config,
        f"[Stage1] Training complete | status={stop_reason} | elapsed={_format_duration(perf_counter() - t0)}",
    )

    return TrainingResult(
        learned_tx=model.tx_basis().detach().clone(),
        learned_rx=model.rx_basis().detach().clone(),
        history_df=pd.DataFrame(history),
        stage_summary_df=pd.DataFrame(stage_summaries),
        stage_failed=stage_failed,
        failed_stage=failed_stage,
        stop_reason=stop_reason,
    )


def _require_stage2_supported_config(config: ExperimentConfig) -> None:
    if config.modulation != "16QAM":
        raise ValueError("Stage 2 nonlinear receiver support is currently limited to 16QAM.")
    if not config.payload_region_enabled:
        raise ValueError("Stage 2 nonlinear receiver support is currently limited to the structured K-bin setup.")


def _stage2_max_abs_cfo(config: ExperimentConfig) -> float:
    support_max = max((abs(value) for value in config.train_cfo_support), default=0.0)
    select_max = max((abs(value) for value in config.stage2_selection_abs_cfo_points), default=0.0)
    return max(float(config.stage_c_cfo), float(support_max), float(select_max), 1e-3)


def _stage2_dual_head_loss(
    config: ExperimentConfig,
    corrected_symbols: torch.Tensor,
    logits: torch.Tensor,
    bits: torch.Tensor,
    ref_symbols: torch.Tensor,
    baseline_logits: torch.Tensor | None = None,
    bit_logits: torch.Tensor | None = None,
    eps_hat: torch.Tensor | None = None,
    eps_true: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bit_logits = qam16_symbol_logits_to_bit_logits(logits) if bit_logits is None else bit_logits.to(torch.float32)
    labels = qam16_labels_from_bits(bits)
    bit_targets = bits.to(torch.float32)
    loss_ce = torch.zeros((), device=logits.device, dtype=torch.float32)
    loss_bce = torch.zeros((), device=logits.device, dtype=torch.float32)
    if baseline_logits is None:
        if config.stage2_decision_loss == "BIT_BCE":
            bce_terms = F.binary_cross_entropy_with_logits(bit_logits, bit_targets, reduction="none")
            loss_decision = bce_terms.mean()
            loss_bce = loss_decision
        else:
            flat_logits = logits.reshape(-1, 16)
            flat_labels = labels.reshape(-1)
            ce_terms = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(labels.shape)
            loss_decision = ce_terms.mean()
            loss_ce = loss_decision
        loss_guard = torch.zeros((), device=logits.device, dtype=torch.float32)
    else:
        baseline_pred = torch.argmax(baseline_logits, dim=-1)
        confidence = logits_margin(baseline_logits)
        hard_mask = (baseline_pred != labels).to(torch.float32)
        confidence_weight = torch.exp(-confidence).clamp_min(0.25)
        symbol_weights = 1.0 + hard_mask + confidence_weight
        if config.stage2_decision_loss == "BIT_BCE":
            bce_terms = F.binary_cross_entropy_with_logits(bit_logits, bit_targets, reduction="none")
            bit_weights = symbol_weights.unsqueeze(-1)
            loss_decision = torch.sum(bce_terms * bit_weights) / bit_weights.sum().clamp_min(1e-12)
            loss_bce = loss_decision
        else:
            flat_logits = logits.reshape(-1, 16)
            flat_labels = labels.reshape(-1)
            ce_terms = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(labels.shape)
            loss_decision = torch.sum(ce_terms * symbol_weights) / symbol_weights.sum().clamp_min(1e-12)
            loss_ce = loss_decision

        easy_mask = (baseline_pred == labels) & (confidence >= config.stage2_easy_margin_threshold)
        if torch.any(easy_mask):
            if config.stage2_decision_loss == "BIT_BCE":
                baseline_bit_logits = qam16_symbol_logits_to_bit_logits(baseline_logits)
                baseline_true_margin = torch.where(bits.to(torch.bool), baseline_bit_logits, -baseline_bit_logits)
                current_true_margin = torch.where(bits.to(torch.bool), bit_logits, -bit_logits)
                easy_weight = easy_mask.to(torch.float32).unsqueeze(-1)
                loss_guard = torch.sum(
                    F.relu(
                        baseline_true_margin
                        - current_true_margin
                        + config.stage2_easy_consistency_margin
                    )
                    * easy_weight
                ) / easy_weight.sum().clamp_min(1e-12)
            else:
                baseline_true = torch.gather(baseline_logits, -1, labels.unsqueeze(-1)).squeeze(-1)
                current_true = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
                loss_guard = torch.sum(
                    F.relu(baseline_true - current_true + config.stage2_easy_consistency_margin)
                    * easy_mask.to(torch.float32)
                ) / easy_mask.to(torch.float32).sum().clamp_min(1e-12)
        else:
            loss_guard = torch.zeros((), device=logits.device, dtype=torch.float32)
    loss_mse = normalized_symbol_mse(corrected_symbols, ref_symbols)
    if eps_hat is None or eps_true is None:
        loss_eps = torch.zeros((), device=logits.device, dtype=torch.float32)
    else:
        eps_scale = _stage2_max_abs_cfo(config)
        loss_eps = torch.mean(((eps_hat - eps_true.to(torch.float32)) / eps_scale) ** 2)
    total = (
        loss_decision
        + config.stage2_loss_mse_weight * loss_mse
        + config.stage2_noninferiority_weight * loss_guard
        + config.stage2_cfo_loss_weight * loss_eps
    )
    return total, {
        "Ldecision": loss_decision,
        "Lbce": loss_bce,
        "Lce": loss_ce,
        "Lmse": loss_mse,
        "Lguard": loss_guard,
        "Leps": loss_eps,
    }


def _stage2_symbol_error_rate(logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    bit_logits = qam16_symbol_logits_to_bit_logits(logits)
    bits_hat = (bit_logits >= 0.0).to(torch.int64)
    return torch.mean(torch.any(bits_hat != bits, dim=-1).to(torch.float32))


def _stage2_bits_from_outputs(logits: torch.Tensor, aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    if aux is not None and "bit_logits" in aux:
        return (aux["bit_logits"] >= 0.0).to(torch.int64)
    return (qam16_symbol_logits_to_bit_logits(logits) >= 0.0).to(torch.int64)


def _stage2_hard_cfo_weighted_ber(
    config: ExperimentConfig,
    model: JointStage2Model,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
) -> torch.Tensor:
    weights = torch.tensor(config.stage2_selection_abs_cfo_weights, device=config.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)
    total = torch.zeros((), device=config.device, dtype=torch.float32)
    for abs_eps, weight in zip(config.stage2_selection_abs_cfo_points, weights):
        eps_pos = torch.full((symbols.shape[0],), float(abs_eps), device=config.device, dtype=torch.float32)
        eps_neg = torch.full((symbols.shape[0],), -float(abs_eps), device=config.device, dtype=torch.float32)
        _, logits_pos, _, aux_pos = model(symbols, eps_pos, config, ebn0_db_values)
        _, logits_neg, _, aux_neg = model(symbols, eps_neg, config, ebn0_db_values)
        ber_pos = torch.mean((_stage2_bits_from_outputs(logits_pos, aux_pos) != bits).to(torch.float32))
        ber_neg = torch.mean((_stage2_bits_from_outputs(logits_neg, aux_neg) != bits).to(torch.float32))
        total = total + weight * 0.5 * (ber_pos + ber_neg)
    return total


def _stage2_hard_cfo_weighted_loss(
    config: ExperimentConfig,
    model: JointStage2Model,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
) -> torch.Tensor:
    weights = torch.tensor(config.stage2_selection_abs_cfo_weights, device=config.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)
    total = torch.zeros((), device=config.device, dtype=torch.float32)
    for abs_eps, weight in zip(config.stage2_selection_abs_cfo_points, weights):
        for signed_eps in (float(abs_eps), -float(abs_eps)):
            eps_values = torch.full((symbols.shape[0],), signed_eps, device=config.device, dtype=torch.float32)
            corrected_symbols, logits, z0, aux = model(symbols, eps_values, config, ebn0_db_values)
            base_logits = stage1_geometry_logits(z0)
            ce_total, _ = _stage2_dual_head_loss(
                config=config,
                corrected_symbols=corrected_symbols,
                logits=logits,
                bits=bits,
                ref_symbols=symbols,
                baseline_logits=base_logits,
                bit_logits=aux.get("bit_logits"),
                eps_hat=aux.get("eps_hat"),
                eps_true=eps_values,
            )
            total = total + 0.5 * weight * ce_total
    return total


def _stage2_total_loss(
    config: ExperimentConfig,
    model: JointStage2Model,
    z0: torch.Tensor,
    corrected_symbols: torch.Tensor,
    logits: torch.Tensor,
    bits: torch.Tensor,
    ref_symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
    eps_values: torch.Tensor,
    aux: dict[str, torch.Tensor],
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    operator_clean: torch.Tensor,
    operator_cfo: torch.Tensor,
    operator_loss_weights: torch.Tensor | None,
    include_spectral_loss: bool,
    optimize_operator_terms: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    baseline_logits = stage1_geometry_logits(z0)
    dual_total, dual_components = _stage2_dual_head_loss(
        config,
        corrected_symbols,
        logits,
        bits,
        ref_symbols,
        baseline_logits=baseline_logits,
        bit_logits=aux.get("bit_logits"),
        eps_hat=aux.get("eps_hat"),
        eps_true=eps_values,
    )
    loss_0 = identity_loss(operator_clean)
    loss_off = offdiag_energy_loss(
        operator_cfo,
        config.offdiag_distance_weight_power,
        weights=operator_loss_weights,
    )
    loss_diag = diag_uniformity_loss(operator_cfo, weights=operator_loss_weights)
    loss_nn = nearest_neighbor_energy_loss(
        operator_cfo,
        max_distance=config.local_leakage_max_distance,
        weights=operator_loss_weights,
    )
    loss_v = receiver_fro_loss(rx_basis)
    loss_spec = spectral_mask_loss(tx_basis, config) if include_spectral_loss else torch.zeros(
        (),
        device=tx_basis.device,
        dtype=torch.float32,
    )
    loss_hard = _stage2_hard_cfo_weighted_loss(config, model, bits, ref_symbols, ebn0_db_values)
    total = dual_total + config.stage2_hard_cfo_loss_weight * loss_hard
    if optimize_operator_terms:
        total = (
            total
            + config.lambda_0 * loss_0
            + config.lambda_1 * loss_off
            + config.lambda_2 * loss_diag
            + config.lambda_3 * loss_v
            + config.lambda_nn * loss_nn
            + config.lambda_spec * loss_spec
        )
    elif not config.stage2_detector_only_logs_operator_terms:
        loss_0 = torch.zeros_like(loss_0)
        loss_off = torch.zeros_like(loss_off)
        loss_diag = torch.zeros_like(loss_diag)
        loss_v = torch.zeros_like(loss_v)
        loss_nn = torch.zeros_like(loss_nn)
        loss_spec = torch.zeros_like(loss_spec)
    return total, {
        "Ldecision": dual_components["Ldecision"],
        "Lbce": dual_components["Lbce"],
        "Lce": dual_components["Lce"],
        "Lmse": dual_components["Lmse"],
        "Lguard": dual_components["Lguard"],
        "Leps": dual_components["Leps"],
        "Lhard": loss_hard,
        "L0": loss_0,
        "Loff": loss_off,
        "Ldiag": loss_diag,
        "LV": loss_v,
        "Lnn": loss_nn,
        "Lspec": loss_spec,
    }


def _set_stage2_requires_grad(model: JointStage2Model, *, train_tx: bool, train_rx: bool) -> None:
    for param in (model.waveform.tx_re, model.waveform.tx_im):
        param.requires_grad_(train_tx)
    for param in (model.waveform.rx_re, model.waveform.rx_im):
        param.requires_grad_(train_rx)


def _filter_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _copy_stage2_state(model: JointStage2Model) -> dict[str, object]:
    return {
        "waveform": copy.deepcopy(model.waveform.state_dict()),
        "receiver": copy.deepcopy(model.nonlinear_receiver.state_dict()),
    }


def _load_stage2_state(model: JointStage2Model, state: dict[str, object]) -> None:
    model.waveform.load_state_dict(state["waveform"])
    model.nonlinear_receiver.load_state_dict(state["receiver"])


def _recover_stage1_tx_init(config: ExperimentConfig, tx_basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    tx_support_frame = make_tx_support_frame(config)
    if tx_support_frame is None:
        return tx_basis.detach().clone(), None
    tx_init = tx_support_frame.conj().T @ tx_basis
    return tx_init.detach().clone(), tx_support_frame


def _build_stage2_validation_batch(
    config: ExperimentConfig,
    stage: StageSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = min(config.train_symbol_batch_size, 128)
    bits, symbols = sample_training_symbols(batch_size, config)
    eps = _sample_training_cfo_values(config, stage, batch_size, deterministic=True)
    ebn0 = _sample_training_ebn0_values(config, batch_size, deterministic=True)
    return bits, symbols, eps if isinstance(eps, torch.Tensor) else torch.full(
        (batch_size,),
        float(eps),
        device=config.device,
        dtype=torch.float32,
    )


def _stage1_hard_cfo_weighted_ber_for_basis(
    config: ExperimentConfig,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
) -> float:
    weights = torch.tensor(config.stage2_selection_abs_cfo_weights, device=config.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)
    total = torch.zeros((), device=config.device, dtype=torch.float32)
    for abs_eps, weight in zip(config.stage2_selection_abs_cfo_points, weights):
        for signed_eps in (float(abs_eps), -float(abs_eps)):
            eps_values = torch.full((symbols.shape[0],), signed_eps, device=config.device, dtype=torch.float32)
            tx_signal = transmit_symbols(symbols, tx_basis)
            rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_db_values)
            z0 = decode_symbols(rx_signal, rx_basis)
            logits = stage1_geometry_logits(z0)
            ber = torch.mean((qam16_bits_from_labels(torch.argmax(logits, dim=-1)) != bits).to(torch.float32))
            total = total + 0.5 * weight * ber
    return float(total.item())


def _stage2_identity_diagnostic(
    config: ExperimentConfig,
    model: JointStage2Model,
    batch_size: int,
) -> dict[str, float]:
    identity_stage = StageSpec("Identity", 1, config.stage_c_cfo, True)
    bits, symbols = sample_training_symbols(batch_size, config)
    eps_values = _sample_training_cfo_values(config, identity_stage, batch_size, deterministic=True)
    ebn0_values = _sample_training_ebn0_values(config, batch_size, deterministic=True)
    with torch.no_grad():
        corrected_symbols, logits, z0, aux = model(symbols, eps_values, config, ebn0_values)
        stage1_logits = stage1_geometry_logits(z0)
        stage1_bit_logits = qam16_symbol_logits_to_bit_logits(stage1_logits)
        stage1_bits = (stage1_bit_logits >= 0.0).to(torch.int64)
        stage2_bit_logits = aux.get("bit_logits", qam16_symbol_logits_to_bit_logits(logits))
        stage2_bits = (stage2_bit_logits >= 0.0).to(torch.int64)
    return {
        "identity_symbol_mse_to_z0": float(normalized_symbol_mse(corrected_symbols, z0).item()),
        "identity_max_symbol_abs_diff": float(torch.max(torch.abs(corrected_symbols - z0)).item()),
        "identity_max_logit_abs_diff": float(torch.max(torch.abs(logits - stage1_logits)).item()),
        "identity_max_bit_logit_abs_diff": float(torch.max(torch.abs(stage2_bit_logits - stage1_bit_logits)).item()),
        "identity_bit_mismatch_rate_vs_stage1": float(torch.mean((stage2_bits != stage1_bits).to(torch.float32)).item()),
        "identity_label_mismatch_rate_vs_stage1": float(
            torch.mean((torch.argmax(logits, dim=-1) != torch.argmax(stage1_logits, dim=-1)).to(torch.float32)).item()
        ),
        "identity_reference_ber_vs_stage1_bits": float(torch.mean((stage1_bits != bits).to(torch.float32)).item()),
    }


def _run_stage2_training_stage(
    *,
    config: ExperimentConfig,
    model: JointStage2Model,
    stage: StageSpec,
    scheme_name: str,
    epochs: int,
    learning_rate: float,
    baseline_tx_basis: torch.Tensor,
    baseline_rx_basis: torch.Tensor,
    train_tx: bool,
    train_rx: bool,
    include_spectral_loss: bool,
    global_epoch_start: int,
    history: list[dict[str, float | int | str]],
) -> tuple[dict[str, object], dict[str, float | int | str | bool | None], int]:
    optimize_operator_terms = bool(train_tx or train_rx)
    _set_stage2_requires_grad(model, train_tx=train_tx, train_rx=train_rx)
    optimizer = torch.optim.Adam(_filter_trainable_parameters(model), lr=learning_rate)
    best: dict[str, object] | None = None
    t0 = perf_counter()
    validation_eps = _stage_validation_grid(config, stage)
    validation_weights = None
    validation_support = _stage_cfo_support(config, stage, device=config.device)
    if validation_support is not None:
        validation_weights = validation_support[1]
    val_bits, val_symbols = sample_training_symbols(min(config.train_symbol_batch_size, 128), config)
    val_eps = _sample_training_cfo_values(config, stage, val_symbols.shape[0], deterministic=True)
    val_ebn0 = _sample_training_ebn0_values(config, val_symbols.shape[0], deterministic=True)
    val_stage1_hard_ber = _stage1_hard_cfo_weighted_ber_for_basis(
        config,
        tx_basis=baseline_tx_basis,
        rx_basis=baseline_rx_basis,
        bits=val_bits,
        symbols=val_symbols,
        ebn0_db_values=val_ebn0,
    )
    global_epoch = global_epoch_start
    _log_progress(
        config,
        (
            f"[Stage2] {scheme_name}:{stage.name} start | epochs={epochs} cfo_span={stage.cfo_span:.2f} "
            f"| train_tx={train_tx} train_rx={train_rx} optimize_operator_terms={optimize_operator_terms}"
        ),
    )

    for epoch in range(epochs):
        global_epoch += 1
        bits, symbols = sample_training_symbols(config.train_symbol_batch_size, config)
        eps_values = _sample_training_cfo_values(config, stage, config.train_symbol_batch_size)
        ebn0_values = _sample_training_ebn0_values(config, config.train_symbol_batch_size)
        corrected_symbols, logits, z0, aux = model(symbols, eps_values, config, ebn0_values)
        tx_basis = model.tx_basis()
        rx_basis = model.rx_basis()
        eps_batch, operator_loss_weights = _stage_operator_loss_support(config, stage)
        operator_clean = effective_operator(tx_basis, rx_basis, torch.tensor([0.0], device=config.device))
        operator_cfo = effective_operator(tx_basis, rx_basis, eps_batch)
        total_loss, components = _stage2_total_loss(
            config,
            model,
            z0,
            corrected_symbols,
            logits,
            bits,
            symbols,
            ebn0_values,
            eps_values,
            aux,
            tx_basis,
            rx_basis,
            operator_clean,
            operator_cfo,
            operator_loss_weights,
            include_spectral_loss=include_spectral_loss,
            optimize_operator_terms=optimize_operator_terms,
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        should_log = (
            epoch == 0
            or epoch + 1 == epochs
            or (epoch + 1) % config.history_log_interval == 0
        )
        if not should_log:
            continue

        with torch.no_grad():
            corrected_val, logits_val, z0_val, aux_val = model(val_symbols, val_eps, config, val_ebn0)
            tx_val = model.tx_basis()
            rx_val = model.rx_basis()
            operator_clean_val = effective_operator(tx_val, rx_val, torch.tensor([0.0], device=config.device))
            operator_stage_val = effective_operator(tx_val, rx_val, validation_eps)
            val_total, val_components = _stage2_total_loss(
                config,
                model,
                z0_val,
                corrected_val,
                logits_val,
                val_bits,
                val_symbols,
                val_ebn0,
                val_eps,
                aux_val,
                tx_val,
                rx_val,
                operator_clean_val,
                operator_stage_val,
                validation_weights,
                include_spectral_loss=include_spectral_loss,
                optimize_operator_terms=optimize_operator_terms,
            )
            val_bits_hat = _stage2_bits_from_outputs(logits_val, aux_val)
            val_ber = torch.mean((val_bits_hat != val_bits).to(torch.float32)).item()
            val_ser = _stage2_symbol_error_rate(logits_val, val_bits).item()
            row = {
                "scheme": scheme_name,
                "stage": stage.name,
                "global_epoch": global_epoch,
                "stage_epoch": epoch + 1,
                "modulation": config.modulation,
                "N": config.N,
                "M": config.M,
                "train_ebn0_db": config.train_ebn0_db,
                "eval_ebn0_db": config.eval_ebn0_db,
                "cfo_span": stage.cfo_span,
                "train_total": total_loss.item(),
                "train_Ldecision": components["Ldecision"].item(),
                "train_Lbce": components["Lbce"].item(),
                "train_Lce": components["Lce"].item(),
                "train_Lmse": components["Lmse"].item(),
                "train_Lguard": components["Lguard"].item(),
                "train_Leps": components["Leps"].item(),
                "train_Lhard": components["Lhard"].item(),
                "train_L0": components["L0"].item(),
                "train_Loff": components["Loff"].item(),
                "train_Ldiag": components["Ldiag"].item(),
                "train_LV": components["LV"].item(),
                "train_Lnn": components["Lnn"].item(),
                "train_Lspec": components["Lspec"].item(),
                "val_total": val_total.item(),
                "val_Ldecision": val_components["Ldecision"].item(),
                "val_Lbce": val_components["Lbce"].item(),
                "val_Lce": val_components["Lce"].item(),
                "val_Lmse": val_components["Lmse"].item(),
                "val_Lguard": val_components["Lguard"].item(),
                "val_Leps": val_components["Leps"].item(),
                "val_Lhard": val_components["Lhard"].item(),
                "val_L0": val_components["L0"].item(),
                "val_Loff": val_components["Loff"].item(),
                "val_Ldiag": val_components["Ldiag"].item(),
                "val_LV": val_components["LV"].item(),
                "val_Lnn": val_components["Lnn"].item(),
                "val_Lspec": val_components["Lspec"].item(),
                "clean_offdiag_leakage": offdiag_leakage_ratio(operator_clean_val)[0].item(),
                "rx_fro_norm_sq": receiver_fro_loss(rx_val).item(),
                "val_ber": val_ber,
                "val_ser": val_ser,
                "hard_cfo_weighted_ber": _stage2_hard_cfo_weighted_ber(
                    config,
                    model,
                    val_bits,
                    val_symbols,
                    val_ebn0,
                ).item(),
                "baseline_hard_cfo_weighted_ber": val_stage1_hard_ber,
                "residual_scale": model.nonlinear_receiver.residual_scale().item(),
                "cancellation_scale": float(aux_val.get("cancellation_scale", torch.zeros((), device=config.device)).item()),
                "eps_hat_mae": float(
                    torch.mean(torch.abs(aux_val.get("eps_hat", torch.zeros_like(val_eps)) - val_eps.to(torch.float32))).item()
                ),
                "elapsed_s": perf_counter() - t0,
            }
            history.append(row)
            progress_pct = 100.0 * float(epoch + 1) / float(epochs)
            _log_progress(
                config,
                (
                    f"[Stage2] {scheme_name}:{stage.name} {epoch + 1}/{epochs} ({progress_pct:5.1f}%) "
                    f"| global={global_epoch} | val_ber={val_ber:.4e} hard_ber={row['hard_cfo_weighted_ber']:.4e} "
                    f"(stage1 {val_stage1_hard_ber:.4e}) "
                    f"| val_total={val_total.item():.4e} | elapsed={_format_duration(row['elapsed_s'])}"
                ),
            )
            candidate_score = (row["hard_cfo_weighted_ber"], row["val_ber"], row["val_total"])
            if best is None or candidate_score < best["score"]:
                best = {
                    "score": candidate_score,
                    "epoch": epoch + 1,
                    "global_epoch": global_epoch,
                    "state": _copy_stage2_state(model),
                    "summary": row,
                }

    assert best is not None
    _load_stage2_state(model, best["state"])
    with torch.no_grad():
        tx_final = model.tx_basis().detach().clone()
        rx_final = model.rx_basis().detach().clone()
        operator_clean = effective_operator(tx_final, rx_final, torch.tensor([0.0], device=config.device))
        operator_stage = effective_operator(tx_final, rx_final, validation_eps)
        corrected_val, logits_val, z0_val, aux_val = model(val_symbols, val_eps, config, val_ebn0)
        val_total, val_components = _stage2_total_loss(
            config,
            model,
            z0_val,
            corrected_val,
            logits_val,
            val_bits,
            val_symbols,
            val_ebn0,
            val_eps,
            aux_val,
            tx_final,
            rx_final,
            operator_clean,
            operator_stage,
            validation_weights,
            include_spectral_loss=include_spectral_loss,
            optimize_operator_terms=optimize_operator_terms,
        )
        stage_row = {
            "scheme": scheme_name,
            "stage": stage.name,
            "modulation": config.modulation,
            "N": config.N,
            "M": config.M,
            "train_ebn0_db": config.train_ebn0_db,
            "eval_ebn0_db": config.eval_ebn0_db,
            "epochs": epochs,
            "cfo_span": stage.cfo_span,
            "best_global_epoch": int(best["global_epoch"]),
            "best_stage_epoch": int(best["epoch"]),
            "val_total": float(val_total.item()),
            "val_Ldecision": float(val_components["Ldecision"].item()),
            "val_Lbce": float(val_components["Lbce"].item()),
            "val_Lce": float(val_components["Lce"].item()),
            "val_Lmse": float(val_components["Lmse"].item()),
            "val_Lguard": float(val_components["Lguard"].item()),
            "val_Leps": float(val_components["Leps"].item()),
            "val_Lhard": float(val_components["Lhard"].item()),
            "val_L0": float(val_components["L0"].item()),
            "val_Loff": float(val_components["Loff"].item()),
            "val_Ldiag": float(val_components["Ldiag"].item()),
            "val_LV": float(val_components["LV"].item()),
            "val_Lnn": float(val_components["Lnn"].item()),
            "val_Lspec": float(val_components["Lspec"].item()),
            "val_ber": float(torch.mean((_stage2_bits_from_outputs(logits_val, aux_val) != val_bits).to(torch.float32)).item()),
            "val_ser": float(_stage2_symbol_error_rate(logits_val, val_bits).item()),
            "hard_cfo_weighted_ber": float(
                _stage2_hard_cfo_weighted_ber(config, model, val_bits, val_symbols, val_ebn0).item()
            ),
            "baseline_hard_cfo_weighted_ber": float(val_stage1_hard_ber),
            "residual_scale": float(model.nonlinear_receiver.residual_scale().item()),
            "cancellation_scale": float(aux_val.get("cancellation_scale", torch.zeros((), device=config.device)).item()),
            "eps_hat_mae": float(
                torch.mean(torch.abs(aux_val.get("eps_hat", torch.zeros_like(val_eps)) - val_eps.to(torch.float32))).item()
            ),
            "clean_identity_loss": float(identity_loss(operator_clean).item()),
            "clean_offdiag_leakage": float(offdiag_leakage_ratio(operator_clean)[0].item()),
            "mean_validation_offdiag": float(offdiag_leakage_ratio(operator_stage).mean().item()),
            "mean_validation_diag_loss": float(diag_uniformity_loss(operator_stage).item()),
            "mean_validation_local_leakage": float(
                nearest_neighbor_leakage_ratio(operator_stage, max_distance=config.local_leakage_max_distance).mean().item()
            ),
            "validation_symbol_loss": float(normalized_symbol_mse(corrected_val, val_symbols).item()),
            "validation_spectral_loss": float(spectral_mask_loss(tx_final, config).item()),
            "receiver_fro_norm_sq": float(receiver_fro_loss(rx_final).item()),
            "stage_failed": False,
            "stop_reason": "Completed.",
        }
    _log_progress(
        config,
        (
            f"[Stage2] {scheme_name}:{stage.name} complete | best_epoch={int(best['epoch'])}/{epochs} "
            f"| hard_ber={stage_row['hard_cfo_weighted_ber']:.4e} "
            f"| clean_identity={stage_row['clean_identity_loss']:.4e} "
            f"| clean_leakage={stage_row['clean_offdiag_leakage']:.4e}"
        ),
    )
    return best, stage_row, global_epoch


def train_stage2_nonlinear_receiver(
    config: ExperimentConfig,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    scheme_name: str,
    seed_offset: int = 0,
) -> Stage2ReceiverTrainingResult:
    _require_stage2_supported_config(config)
    set_seed(config.base_seed + seed_offset)

    tx_basis = tx_basis.detach().clone().to(config.device)
    rx_basis = rx_basis.detach().clone().to(config.device)
    tx_init, tx_support_frame = _recover_stage1_tx_init(config, tx_basis)
    if config.stage2_detector_arch == "DENSE":
        receiver = DenseSymbolResidualReceiver(
            num_symbols=config.N,
            hidden_multiplier=config.stage2_hidden_multiplier,
            residual_scale_init=config.stage2_residual_scale_init,
            use_residual_logit_head=config.stage2_use_residual_logit_head,
        ).to(config.device)
    else:
        receiver = LocalSymbolResidualReceiver(
            num_symbols=config.N,
            channels=config.stage2_local_channels,
            kernel_size=config.stage2_local_kernel_size,
            residual_scale_init=config.stage2_residual_scale_init,
            cancellation_scale_init=config.stage2_cancellation_scale_init,
            use_confidence_features=config.stage2_use_confidence_features,
            use_symbol_correction_head=config.stage2_use_symbol_correction_head,
            use_residual_logit_head=config.stage2_use_residual_logit_head,
            max_abs_eps=_stage2_max_abs_cfo(config),
        ).to(config.device)
    model = JointStage2Model(
        tx_init=tx_init,
        rx_init=rx_basis,
        nonlinear_receiver=receiver,
        tx_support_frame=tx_support_frame,
    ).to(config.device)
    history: list[dict[str, float | int | str]] = []
    stage_rows: list[dict[str, float | int | str | bool | None]] = []
    global_epoch = 0
    identity_diag = _stage2_identity_diagnostic(config, model, batch_size=min(config.train_symbol_batch_size, 128))
    if (
        identity_diag["identity_bit_mismatch_rate_vs_stage1"] > 0.0
        or identity_diag["identity_label_mismatch_rate_vs_stage1"] > 0.0
        or identity_diag["identity_max_symbol_abs_diff"] > 1e-7
        or identity_diag["identity_max_logit_abs_diff"] > 1e-7
        or identity_diag["identity_max_bit_logit_abs_diff"] > 1e-7
    ):
        raise ValueError(f"Stage 2 identity initialization check failed: {identity_diag}")
    _log_progress(
        config,
        (
            f"[Stage2] Identity diagnostic | symbol_mse={identity_diag['identity_symbol_mse_to_z0']:.4e} "
            f"bit_mismatch={identity_diag['identity_bit_mismatch_rate_vs_stage1']:.4e} "
            f"logit_diff={identity_diag['identity_max_logit_abs_diff']:.4e} "
            f"bit_logit_diff={identity_diag['identity_max_bit_logit_abs_diff']:.4e}"
        ),
    )
    if config.stage2_workflow == "NONLINEAR_ONLY":
        _, stage_row, global_epoch = _run_stage2_training_stage(
            config=config,
            model=model,
            stage=StageSpec("NonlinearOnly", config.stage2_nonlinear_only_epochs, config.stage_c_cfo, True),
            scheme_name=scheme_name,
            epochs=config.stage2_nonlinear_only_epochs,
            learning_rate=config.stage2_nonlinear_only_learning_rate,
            baseline_tx_basis=tx_basis,
            baseline_rx_basis=rx_basis,
            train_tx=False,
            train_rx=False,
            include_spectral_loss=False,
            global_epoch_start=global_epoch,
            history=history,
        )
        stage_row.update(identity_diag)
        stage_rows.append(stage_row)
    else:
        _, warmstart_row, global_epoch = _run_stage2_training_stage(
            config=config,
            model=model,
            stage=StageSpec("Warmstart", config.stage2_warmstart_epochs, config.stage_c_cfo, True),
            scheme_name=scheme_name,
            epochs=config.stage2_warmstart_epochs,
            learning_rate=config.stage2_warmstart_learning_rate,
            baseline_tx_basis=tx_basis,
            baseline_rx_basis=rx_basis,
            train_tx=False,
            train_rx=False,
            include_spectral_loss=False,
            global_epoch_start=global_epoch,
            history=history,
        )
        warmstart_row.update(identity_diag)
        stage_rows.append(warmstart_row)
        baseline_hard_ber = float(warmstart_row["baseline_hard_cfo_weighted_ber"])
        detector_won = warmstart_row["hard_cfo_weighted_ber"] < baseline_hard_ber - config.stage2_reopen_v_gain_tol
        should_reopen_v = config.stage2_reopen_v_after_win and (
            detector_won or not config.stage2_reopen_v_requires_gain
        )
        if should_reopen_v:
            _, reopen_row, global_epoch = _run_stage2_training_stage(
                config=config,
                model=model,
                stage=StageSpec("ReopenV", config.stage2_reopen_v_epochs, config.stage_c_cfo, True),
                scheme_name=scheme_name,
                epochs=config.stage2_reopen_v_epochs,
                learning_rate=config.stage2_reopen_v_learning_rate,
                baseline_tx_basis=tx_basis,
                baseline_rx_basis=rx_basis,
                train_tx=False,
                train_rx=True,
                include_spectral_loss=False,
                global_epoch_start=global_epoch,
                history=history,
            )
            reopen_row.update(identity_diag)
            stage_rows.append(reopen_row)
        else:
            stage_rows.append(
                {
                    "scheme": scheme_name,
                    "stage": "ReopenV",
                    "modulation": config.modulation,
                    "N": config.N,
                    "M": config.M,
                    "train_ebn0_db": config.train_ebn0_db,
                    "eval_ebn0_db": config.eval_ebn0_db,
                    "epochs": config.stage2_reopen_v_epochs,
                    "cfo_span": config.stage_c_cfo,
                    "best_global_epoch": int(global_epoch),
                    "best_stage_epoch": 0,
                    "val_total": float("nan"),
                    "val_Ldecision": float("nan"),
                    "val_Lbce": float("nan"),
                    "val_Lce": float("nan"),
                    "val_Lmse": float("nan"),
                    "val_Lguard": float("nan"),
                    "val_Leps": float("nan"),
                    "val_Lhard": float("nan"),
                    "val_L0": float("nan"),
                    "val_Loff": float("nan"),
                    "val_Ldiag": float("nan"),
                    "val_LV": float("nan"),
                    "val_Lnn": float("nan"),
                    "val_Lspec": float("nan"),
                    "val_ber": float("nan"),
                    "val_ser": float("nan"),
                    "hard_cfo_weighted_ber": float("nan"),
                    "baseline_hard_cfo_weighted_ber": float(baseline_hard_ber),
                    "residual_scale": float(model.nonlinear_receiver.residual_scale().item()),
                    "cancellation_scale": float("nan"),
                    "eps_hat_mae": float("nan"),
                    "clean_identity_loss": float("nan"),
                    "clean_offdiag_leakage": float("nan"),
                    "mean_validation_offdiag": float("nan"),
                    "mean_validation_diag_loss": float("nan"),
                    "mean_validation_local_leakage": float("nan"),
                    "validation_symbol_loss": float("nan"),
                    "validation_spectral_loss": float("nan"),
                    "receiver_fro_norm_sq": float("nan"),
                    "stage_failed": False,
                    "stop_reason": "Skipped because nonlinear-only did not beat stage1_linear on the hard-CFO target.",
                    **identity_diag,
                }
            )
    model.nonlinear_receiver.eval()
    return Stage2ReceiverTrainingResult(
        receiver=model.nonlinear_receiver,
        learned_tx=model.tx_basis().detach().clone(),
        learned_rx=model.rx_basis().detach().clone(),
        history_df=pd.DataFrame(history),
        stage_summary_df=pd.DataFrame(stage_rows),
    )


def detect_scheme_symbols(
    config: ExperimentConfig,
    scheme: EvaluationScheme,
    rx_signal: torch.Tensor,
    eps_tensor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    rx_basis = scheme.rx_basis
    pilot_symbol_mse = float("nan")
    cfo_est_mae = float("nan")
    cfo_est_bias = float("nan")
    mean_estimated_eps = float("nan")

    if config.frame_structure_enabled and config.pilot_estimation_enabled and not config.payload_region_enabled:
        estimated_eps, pilot_mse = estimate_residual_cfo_from_pilots(config, rx_signal, rx_basis)
        rx_signal = apply_residual_cfo(rx_signal, -estimated_eps)
        pilot_symbol_mse = float(pilot_mse.mean().item())
        if eps_tensor is not None:
            cfo_error = estimated_eps - eps_tensor
            cfo_est_mae = float(torch.mean(torch.abs(cfo_error)).item())
            cfo_est_bias = float(torch.mean(cfo_error).item())
            mean_estimated_eps = float(torch.mean(estimated_eps).item())

    z0 = decode_symbols(rx_signal, rx_basis)
    logits = None
    if scheme.nonlinear_receiver is not None:
        corrected_symbols, logits, aux = scheme.nonlinear_receiver(z0)
        if config.modulation != "16QAM":
            raise ValueError("Nonlinear receiver inference is currently implemented only for 16QAM.")
        bits_hat_full = _stage2_bits_from_outputs(logits, aux)
    else:
        corrected_symbols = z0
        aux = {}
        bits_hat_full, _ = slice_symbols(corrected_symbols, config.modulation)

    if config.frame_structure_enabled and config.pilot_estimation_enabled and not config.payload_region_enabled:
        layout = frame_resource_layout(config)
        data_idx = torch.tensor(layout.data_stream_indices, device=rx_signal.device, dtype=torch.long)
        shat_data = corrected_symbols[:, data_idx]
        bits_hat = bits_hat_full[:, data_idx, :]
    else:
        shat_data = corrected_symbols
        bits_hat = bits_hat_full

    return {
        "symbol_estimates": shat_data,
        "bits_hat": bits_hat,
        "pilot_symbol_mse": pilot_symbol_mse,
        "cfo_est_mae": cfo_est_mae,
        "cfo_est_bias": cfo_est_bias,
        "mean_estimated_eps": mean_estimated_eps,
        "stage2_eps_hat_mean": float(aux.get("eps_hat", torch.zeros(1, device=rx_signal.device)).mean().item()),
    }


def evaluate_single(
    config: ExperimentConfig,
    scheme: EvaluationScheme,
    eps: float,
    ebn0_db: float,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    noise: torch.Tensor,
) -> dict[str, float]:
    eps_tensor = torch.full((symbols.shape[0],), float(eps), device=symbols.device, dtype=torch.float32)
    tx_signal = transmit_symbols(symbols, scheme.tx_basis)
    rx_signal, _ = propagate(tx_signal, eps_tensor, config, ebn0_db=ebn0_db, noise=noise)
    outputs = detect_scheme_symbols(config, scheme, rx_signal, eps_tensor=eps_tensor)
    shat_data = outputs["symbol_estimates"]
    bits_hat = outputs["bits_hat"]

    if config.frame_structure_enabled and config.pilot_estimation_enabled and not config.payload_region_enabled:
        layout = frame_resource_layout(config)
        data_idx = torch.tensor(layout.data_stream_indices, device=symbols.device, dtype=torch.long)
        ref_data = symbols[:, data_idx]
    else:
        ref_data = symbols

    bit_errors = (bits_hat != bits).sum().item()
    bit_total = bits.numel()
    symbol_errors = torch.any(bits_hat != bits, dim=-1).sum().item()
    evm = torch.sqrt(
        torch.mean(torch.abs(shat_data - ref_data) ** 2).real
        / torch.mean(torch.abs(ref_data) ** 2).real.clamp_min(1e-12)
    ).item()
    return {
        "ber": bit_errors / bit_total,
        "ser": symbol_errors / (bits.shape[0] * bits.shape[1]),
        "evm": evm,
        "bit_errors": bit_errors,
        "bit_total": bit_total,
        "pilot_symbol_mse": float(outputs["pilot_symbol_mse"]),
        "cfo_est_mae": float(outputs["cfo_est_mae"]),
        "cfo_est_bias": float(outputs["cfo_est_bias"]),
        "mean_estimated_eps": float(outputs["mean_estimated_eps"]),
    }


def evaluate_scheme_set(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    cfo_points: np.ndarray,
    ebn0_db: float,
    num_blocks: int,
    batch_size: int,
    seed: int = 12_345,
    progress_label: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if progress_label:
        _log_progress(
            config,
            (
                f"[Eval] {progress_label} start | methods={len(schemes)} cfo_points={len(cfo_points)} "
                f"blocks={num_blocks} batch={batch_size} ebn0={float(ebn0_db):.1f} dB"
            ),
        )
    for eps_idx, eps in enumerate(cfo_points):
        set_seed(seed + 1000 * eps_idx)
        remaining = num_blocks
        processed = 0
        progress_marks = {max(1, int(round(num_blocks * frac))) for frac in (0.25, 0.5, 0.75, 1.0)}
        if progress_label:
            _log_progress(
                config,
                f"[Eval] {progress_label} CFO {eps_idx + 1}/{len(cfo_points)} | eps={float(eps):+.3f}",
            )
        accum = {
            name: {
                "bit_errors": 0,
                "bit_total": 0,
                "ser_sum": 0.0,
                "evm_sum": 0.0,
                "batches": 0,
                "pilot_symbol_mse_sum": 0.0,
                "cfo_est_mae_sum": 0.0,
                "cfo_est_bias_sum": 0.0,
                "mean_estimated_eps_sum": 0.0,
                "pilot_batches": 0,
            }
            for name in schemes
        }
        while remaining > 0:
            bsz = min(batch_size, remaining)
            bits, symbols = sample_training_symbols(bsz, config)
            noise_var = 1.0 / (config.bits_per_symbol * (10 ** (float(ebn0_db) / 10.0)))
            sigma = np.sqrt(noise_var / 2.0)
            noise = sigma * (
                torch.randn(bsz, config.M, device=config.device)
                + 1j * torch.randn(bsz, config.M, device=config.device)
            ).to(torch.complex64)
            for name, scheme in schemes.items():
                metrics = evaluate_single(
                    config=config,
                    scheme=scheme,
                    eps=float(eps),
                    ebn0_db=ebn0_db,
                    bits=bits,
                    symbols=symbols,
                    noise=noise,
                )
                accum[name]["bit_errors"] += metrics["bit_errors"]
                accum[name]["bit_total"] += metrics["bit_total"]
                accum[name]["ser_sum"] += metrics["ser"]
                accum[name]["evm_sum"] += metrics["evm"]
                accum[name]["batches"] += 1
                if np.isfinite(metrics["pilot_symbol_mse"]):
                    accum[name]["pilot_symbol_mse_sum"] += metrics["pilot_symbol_mse"]
                    accum[name]["cfo_est_mae_sum"] += metrics["cfo_est_mae"]
                    accum[name]["cfo_est_bias_sum"] += metrics["cfo_est_bias"]
                    accum[name]["mean_estimated_eps_sum"] += metrics["mean_estimated_eps"]
                    accum[name]["pilot_batches"] += 1
            remaining -= bsz
            processed += bsz
            if progress_label and processed in progress_marks:
                _log_progress(
                    config,
                    (
                        f"[Eval] {progress_label} CFO {eps_idx + 1}/{len(cfo_points)} "
                        f"| {processed}/{num_blocks} blocks ({100.0 * processed / num_blocks:5.1f}%)"
                    ),
                )
        for name in schemes:
            entry = accum[name]
            pilot_count = max(entry["pilot_batches"], 1)
            rows.append(
                {
                    "method": name,
                    "modulation": config.modulation,
                    "N": config.N,
                    "M": config.M,
                    "K": config.K,
                    "R": config.redundancy_dimensions,
                    "N_data": config.N_data,
                    "N_pilots": config.N_pilots,
                    "N_guard": config.N_guard,
                    "payload_region_fraction": config.payload_region_fraction,
                    "payload_fraction": config.payload_fraction,
                    "train_ebn0_db": config.train_ebn0_db,
                    "eval_ebn0_db": float(ebn0_db),
                    "eps": float(eps),
                    "ber": entry["bit_errors"] / entry["bit_total"],
                    "ser": entry["ser_sum"] / entry["batches"],
                    "evm": entry["evm_sum"] / entry["batches"],
                    "bit_errors": entry["bit_errors"],
                    "bit_total": entry["bit_total"],
                    "pilot_symbol_mse": (
                        entry["pilot_symbol_mse_sum"] / pilot_count if entry["pilot_batches"] > 0 else np.nan
                    ),
                    "cfo_est_mae": entry["cfo_est_mae_sum"] / pilot_count if entry["pilot_batches"] > 0 else np.nan,
                    "cfo_est_bias": entry["cfo_est_bias_sum"] / pilot_count if entry["pilot_batches"] > 0 else np.nan,
                    "mean_estimated_eps": (
                        entry["mean_estimated_eps_sum"] / pilot_count if entry["pilot_batches"] > 0 else np.nan
                    ),
                }
            )
    if progress_label:
        _log_progress(config, f"[Eval] {progress_label} complete")
    return pd.DataFrame(rows)
