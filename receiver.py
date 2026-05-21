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
from comm_core import ExperimentConfig, StageSpec, TrainingResult, ebn0_to_noise_variance, normalize_columns, set_seed, stage_specs
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
    oracle_pre_v_cfo: bool = False
    oracle_post_v_mmse: bool = False
    oracle_mmse_alpha: float | None = None
    oracle_mmse_eps_scale: float | None = None
    oracle_mmse_eps_noise_std_rel: float | None = None
    oracle_mmse_eps_noise_seed_offset: int = 0
    oracle_eps_conditioning: bool = False


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


def oracle_symbol_mmse_equalize(
    z0: torch.Tensor,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    eps_values: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    operator = effective_operator(tx_basis, rx_basis, eps_values)
    ah = torch.conj(torch.transpose(operator, -2, -1))
    eye = torch.eye(operator.shape[-1], device=operator.device, dtype=operator.dtype).unsqueeze(0)
    lhs = ah @ operator + float(alpha) * eye
    rhs = ah @ z0.unsqueeze(-1)
    return torch.linalg.solve(lhs, rhs).squeeze(-1)


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


def papr_smooth_loss(tx_basis: torch.Tensor, symbols: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    if not config.papr_constraint_enabled or config.lambda_papr <= 0.0:
        return torch.zeros((), device=tx_basis.device, dtype=torch.float32)

    tx_signal = transmit_symbols(symbols, tx_basis)
    power = torch.abs(tx_signal) ** 2
    normalized_power = power / power.mean(dim=1, keepdim=True).clamp_min(1e-12)
    beta = float(config.papr_smoothmax_beta)
    smooth_peak = (
        torch.logsumexp(beta * normalized_power, dim=1) - math.log(float(tx_signal.shape[1]))
    ) / beta
    return smooth_peak.mean().to(torch.float32)


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
    symbols: torch.Tensor,
    symbol_loss: torch.Tensor,
    use_cfo_losses: bool,
    operator_loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_0 = identity_loss(operator_clean)
    loss_v = receiver_fro_loss(rx_basis)
    loss_spec = spectral_mask_loss(tx_basis, config)
    loss_papr = papr_smooth_loss(tx_basis, symbols, config)
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
        + config.lambda_papr * loss_papr
    )
    return total, {
        "L0": loss_0,
        "Loff": loss_off,
        "Ldiag": loss_diag,
        "LV": loss_v,
        "Lsym": symbol_loss,
        "Lnn": loss_nn,
        "Lspec": loss_spec,
        "Lpapr": loss_papr,
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


def _complex_phase_features(symbols: torch.Tensor) -> torch.Tensor:
    phase = torch.angle(symbols).to(torch.float32)
    return torch.stack(
        [
            torch.real(symbols).to(torch.float32),
            torch.imag(symbols).to(torch.float32),
            torch.abs(symbols).to(torch.float32),
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=1,
    )


def stage1_geometry_logits(symbols: torch.Tensor) -> torch.Tensor:
    constellation = qam16_constellation_points(symbols.device, dtype=symbols.dtype)
    distances = torch.abs(symbols.unsqueeze(-1) - constellation.view(1, 1, -1)) ** 2
    return (-distances.real).to(torch.float32)


def logits_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


class CfoEstimatorMmseReceiver(nn.Module):
    def __init__(
        self,
        tx_basis: torch.Tensor,
        rx_basis: torch.Tensor,
        num_symbols: int,
        bits_per_symbol: int,
        architecture: str = "LOCAL",
        channels: int = 32,
        hidden_multiplier: int = 4,
        kernel_size: int = 5,
        max_abs_eps: float = 0.1,
        alpha_scale: float = 1.0,
        sideinfo_enabled: bool = False,
        sideinfo_mode: str = "NONE",
        sideinfo_scale: float = 0.0,
        sideinfo_residual_enabled: bool = False,
    ) -> None:
        super().__init__()
        if num_symbols <= 0:
            raise ValueError("num_symbols must be positive.")
        if bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be positive.")
        if channels < 1:
            raise ValueError("channels must be positive.")
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least 1.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if max_abs_eps <= 0.0:
            raise ValueError("max_abs_eps must be positive.")
        if alpha_scale <= 0.0:
            raise ValueError("alpha_scale must be positive.")

        architecture = architecture.upper()
        if architecture not in {"DENSE", "LOCAL"}:
            raise ValueError(f"Unsupported Stage 2 architecture: {architecture}")

        self.register_buffer("tx_basis_buffer", tx_basis.detach().clone().to(torch.complex64))
        self.register_buffer("rx_basis_buffer", rx_basis.detach().clone().to(torch.complex64))
        self.num_symbols = int(num_symbols)
        self.bits_per_symbol = int(bits_per_symbol)
        self.architecture = architecture
        self.channels = int(channels)
        self.hidden_multiplier = int(hidden_multiplier)
        self.kernel_size = int(kernel_size)
        self.feature_channels = 5
        self.max_abs_eps = float(max_abs_eps)
        self.alpha_scale = float(alpha_scale)
        self.sideinfo_enabled = bool(sideinfo_enabled)
        self.sideinfo_mode = str(sideinfo_mode).upper()
        self.sideinfo_scale = float(sideinfo_scale)
        self.sideinfo_residual_enabled = bool(sideinfo_residual_enabled)
        self.blend_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        if self.sideinfo_mode not in {"NONE", "SCALED_TRUE"}:
            raise ValueError(f"Unsupported Stage 2 side-information mode: {self.sideinfo_mode}")
        if not self.sideinfo_enabled:
            self.sideinfo_mode = "NONE"
            self.sideinfo_scale = 0.0
            self.sideinfo_residual_enabled = False
        if self.sideinfo_residual_enabled and self.sideinfo_mode == "NONE":
            raise ValueError("Stage 2 residual-on-coarse requires a concrete side-information mode.")

        if self.architecture == "DENSE":
            feature_dim = self.feature_channels * self.num_symbols
            hidden_dim = feature_dim * self.hidden_multiplier
            self.dense_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.dense_head[-1].weight)
            nn.init.zeros_(self.dense_head[-1].bias)
            if self.sideinfo_residual_enabled:
                self.dense_sideinfo_head = nn.Sequential(
                    nn.Linear(feature_dim + 1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.dense_sideinfo_head[-1].weight)
                nn.init.zeros_(self.dense_sideinfo_head[-1].bias)
            else:
                self.dense_sideinfo_head = None
            self.conv1 = None
            self.conv2 = None
            self.conv3 = None
            self.pool_head = None
            self.pool_sideinfo_head = None
        else:
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(self.feature_channels, self.channels, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
            self.conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=kernel_size, padding=padding)
            self.pool_head = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, 1),
            )
            nn.init.zeros_(self.pool_head[-1].weight)
            nn.init.zeros_(self.pool_head[-1].bias)
            if self.sideinfo_residual_enabled:
                self.pool_sideinfo_head = nn.Sequential(
                    nn.Linear(self.channels + 1, self.channels),
                    nn.GELU(),
                    nn.Linear(self.channels, 1),
                )
                nn.init.zeros_(self.pool_sideinfo_head[-1].weight)
                nn.init.zeros_(self.pool_sideinfo_head[-1].bias)
            else:
                self.pool_sideinfo_head = None
            self.dense_head = None
            self.dense_sideinfo_head = None

    def tx_basis(self) -> torch.Tensor:
        return self.tx_basis_buffer

    def rx_basis(self) -> torch.Tensor:
        return self.rx_basis_buffer

    def residual_scale(self) -> torch.Tensor:
        return torch.tanh(self.blend_param)

    def cancellation_scale(self) -> torch.Tensor:
        return torch.zeros((), device=self.blend_param.device, dtype=torch.float32)

    def _coarse_eps_from_true(self, eps_true: torch.Tensor) -> torch.Tensor:
        eps_true = eps_true.to(torch.float32)
        if self.sideinfo_mode == "NONE":
            raise ValueError("Stage 2 coarse CFO requested without an enabled side-information mode.")
        if self.sideinfo_mode == "SCALED_TRUE":
            coarse = self.sideinfo_scale * eps_true
        else:
            raise ValueError(f"Unsupported Stage 2 side-information mode: {self.sideinfo_mode}")
        return torch.clamp(coarse, min=-self.max_abs_eps, max=self.max_abs_eps)

    def _estimate_eps(self, z0: torch.Tensor) -> torch.Tensor:
        features = _complex_phase_features(z0)
        if self.architecture == "DENSE":
            assert self.dense_head is not None
            flat = torch.flatten(features, start_dim=1)
            return torch.tanh(self.dense_head(flat).squeeze(-1)) * self.max_abs_eps
        assert self.conv1 is not None and self.conv2 is not None and self.conv3 is not None and self.pool_head is not None
        hidden = F.gelu(self.conv1(features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = F.gelu(self.conv3(hidden))
        pooled = torch.mean(hidden, dim=-1)
        return torch.tanh(self.pool_head(pooled).squeeze(-1)) * self.max_abs_eps

    def _estimate_eps_from_coarse(self, z0: torch.Tensor, eps_coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps_coarse = eps_coarse.to(device=z0.device, dtype=torch.float32)
        features = _complex_phase_features(z0)
        if self.architecture == "DENSE":
            assert self.dense_sideinfo_head is not None
            flat = torch.flatten(features, start_dim=1)
            sideinfo_input = torch.cat([flat, eps_coarse.unsqueeze(-1)], dim=-1)
            residual = torch.tanh(self.dense_sideinfo_head(sideinfo_input).squeeze(-1)) * self.max_abs_eps
        else:
            assert self.conv1 is not None and self.conv2 is not None and self.conv3 is not None and self.pool_sideinfo_head is not None
            hidden = F.gelu(self.conv1(features))
            hidden = F.gelu(self.conv2(hidden))
            hidden = F.gelu(self.conv3(hidden))
            pooled = torch.mean(hidden, dim=-1)
            sideinfo_input = torch.cat([pooled, eps_coarse.unsqueeze(-1)], dim=-1)
            residual = torch.tanh(self.pool_sideinfo_head(sideinfo_input).squeeze(-1)) * self.max_abs_eps
        eps_hat = torch.clamp(eps_coarse + residual, min=-self.max_abs_eps, max=self.max_abs_eps)
        return eps_hat, residual

    def _alpha_values(self, ebn0_db_values: float | torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(ebn0_db_values, torch.Tensor):
            noise_var = 1.0 / (
                self.bits_per_symbol * torch.pow(10.0, ebn0_db_values.to(device=device, dtype=torch.float32) / 10.0)
            )
            return self.alpha_scale * noise_var.to(torch.float32)
        return torch.full(
            (batch_size,),
            float(self.alpha_scale * ebn0_to_noise_variance(float(ebn0_db_values), self.bits_per_symbol)),
            device=device,
            dtype=torch.float32,
        )

    def _mmse_solve(
        self,
        z0: torch.Tensor,
        eps_hat: torch.Tensor,
        ebn0_db_values: float | torch.Tensor,
    ) -> torch.Tensor:
        operator = effective_operator(self.tx_basis(), self.rx_basis(), eps_hat)
        ah = torch.conj(torch.transpose(operator, -2, -1))
        alpha = self._alpha_values(ebn0_db_values, z0.shape[0], z0.device).view(-1, 1, 1)
        eye = torch.eye(self.num_symbols, device=z0.device, dtype=operator.dtype).unsqueeze(0)
        lhs = ah @ operator + alpha.to(operator.dtype) * eye
        rhs = ah @ z0.unsqueeze(-1)
        return torch.linalg.solve(lhs, rhs).squeeze(-1)

    def forward(
        self,
        z0: torch.Tensor,
        ebn0_db_values: float | torch.Tensor,
        eps_override: torch.Tensor | None = None,
        eps_true: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        eps_coarse = None
        eps_residual = None
        if eps_override is None:
            if self.sideinfo_enabled:
                if eps_true is None:
                    raise ValueError("Side-information-conditioned Stage 2 requires the true CFO tensor to build the coarse seed.")
                eps_coarse = self._coarse_eps_from_true(eps_true)
                if self.sideinfo_residual_enabled:
                    eps_hat, eps_residual = self._estimate_eps_from_coarse(z0, eps_coarse)
                else:
                    eps_hat = eps_coarse
                    eps_residual = torch.zeros_like(eps_hat)
            else:
                eps_hat = self._estimate_eps(z0)
        else:
            eps_hat = eps_override.to(device=z0.device, dtype=torch.float32)
        solved_symbols = self._mmse_solve(z0, eps_hat, ebn0_db_values)
        if self.sideinfo_enabled:
            corrected_symbols = solved_symbols
            solver_blend = torch.ones((), device=z0.device, dtype=torch.float32)
        else:
            blend = self.residual_scale().to(z0.dtype)
            corrected_symbols = z0 + blend * (solved_symbols - z0)
            solver_blend = self.residual_scale().to(torch.float32)
        logits = stage1_geometry_logits(corrected_symbols)
        aux = {
            "eps_hat": eps_hat.to(torch.float32),
            "bit_logits": qam16_symbol_logits_to_bit_logits(logits),
            "cancellation_scale": self.cancellation_scale().to(torch.float32),
            "solver_blend": solver_blend,
        }
        if eps_coarse is not None:
            aux["eps_coarse"] = eps_coarse.to(torch.float32)
        if eps_residual is not None:
            aux["eps_residual"] = eps_residual.to(torch.float32)
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
        oracle_eps_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tx_basis = self.tx_basis()
        rx_basis = self.rx_basis()
        tx_signal = transmit_symbols(symbols, tx_basis)
        rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_db_values)
        z0 = decode_symbols(rx_signal, rx_basis)
        corrected_symbols, logits, aux = self.nonlinear_receiver(
            z0,
            ebn0_db_values=ebn0_db_values,
            eps_override=oracle_eps_override,
            eps_true=eps_values,
        )
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
                symbols=symbols,
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
                    symbols=val_symbols,
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
                        "train_Lpapr": components["Lpapr"].item(),
                        "val_total": val_total.item(),
                        "val_L0": val_components["L0"].item(),
                        "val_Loff": val_components["Loff"].item(),
                        "val_Ldiag": val_components["Ldiag"].item(),
                        "val_LV": val_components["LV"].item(),
                        "val_Lsym": val_components["Lsym"].item(),
                        "val_Lnn": val_components["Lnn"].item(),
                        "val_Lspec": val_components["Lspec"].item(),
                        "val_Lpapr": val_components["Lpapr"].item(),
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
                "validation_papr_loss": papr_smooth_loss(tx_final, val_symbols, config).item(),
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


def _stage2_training_abs_cfo_scale(config: ExperimentConfig) -> float:
    support_max = max((abs(value) for value in config.train_cfo_support), default=0.0)
    return max(float(config.stage_c_cfo), float(support_max), 1e-3)


def _stage2_estimator_max_abs_cfo(config: ExperimentConfig) -> float:
    if config.stage2_estimator_max_abs_cfo is not None:
        return float(config.stage2_estimator_max_abs_cfo)
    select_max = max((abs(value) for value in config.stage2_selection_abs_cfo_points), default=0.0)
    return max(_stage2_training_abs_cfo_scale(config), float(select_max), 1e-3)


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
    loss_mse = normalized_symbol_mse(corrected_symbols, ref_symbols)
    if eps_hat is None or eps_true is None:
        loss_eps = torch.zeros((), device=logits.device, dtype=torch.float32)
    else:
        eps_scale = _stage2_training_abs_cfo_scale(config)
        loss_eps = torch.mean(((eps_hat - eps_true.to(torch.float32)) / eps_scale) ** 2)
    total = loss_decision + config.stage2_loss_mse_weight * loss_mse + config.stage2_cfo_loss_weight * loss_eps
    return total, {
        "Ldecision": loss_decision,
        "Lbce": loss_bce,
        "Lce": loss_ce,
        "Lmse": loss_mse,
        "Lguard": loss_guard,
        "Leps": loss_eps,
    }


def _bits_from_complex_symbols(symbols: torch.Tensor, modulation: str) -> torch.Tensor:
    bits_hat, _ = slice_symbols(symbols, modulation)
    return bits_hat


def _stage2_symbol_error_rate(symbols: torch.Tensor, bits: torch.Tensor, modulation: str) -> torch.Tensor:
    bits_hat = _bits_from_complex_symbols(symbols, modulation)
    return torch.mean(torch.any(bits_hat != bits, dim=-1).to(torch.float32))


def _stage2_bits_from_outputs(symbols: torch.Tensor, modulation: str) -> torch.Tensor:
    return _bits_from_complex_symbols(symbols, modulation)


def _stage2_hard_cfo_weighted_ber(
    config: ExperimentConfig,
    model: JointStage2Model,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
    oracle_eps_conditioning: bool = False,
) -> torch.Tensor:
    weights = torch.tensor(config.stage2_selection_abs_cfo_weights, device=config.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)
    total = torch.zeros((), device=config.device, dtype=torch.float32)
    for abs_eps, weight in zip(config.stage2_selection_abs_cfo_points, weights):
        eps_pos = torch.full((symbols.shape[0],), float(abs_eps), device=config.device, dtype=torch.float32)
        eps_neg = torch.full((symbols.shape[0],), -float(abs_eps), device=config.device, dtype=torch.float32)
        oracle_eps_pos = eps_pos if oracle_eps_conditioning else None
        oracle_eps_neg = eps_neg if oracle_eps_conditioning else None
        corrected_pos, _, _, _ = model(symbols, eps_pos, config, ebn0_db_values, oracle_eps_override=oracle_eps_pos)
        corrected_neg, _, _, _ = model(symbols, eps_neg, config, ebn0_db_values, oracle_eps_override=oracle_eps_neg)
        ber_pos = torch.mean((_stage2_bits_from_outputs(corrected_pos, config.modulation) != bits).to(torch.float32))
        ber_neg = torch.mean((_stage2_bits_from_outputs(corrected_neg, config.modulation) != bits).to(torch.float32))
        total = total + weight * 0.5 * (ber_pos + ber_neg)
    return total


def _stage2_hard_cfo_weighted_loss(
    config: ExperimentConfig,
    model: JointStage2Model,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    ebn0_db_values: float | torch.Tensor,
    oracle_eps_conditioning: bool = False,
) -> torch.Tensor:
    weights = torch.tensor(config.stage2_selection_abs_cfo_weights, device=config.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)
    total = torch.zeros((), device=config.device, dtype=torch.float32)
    for abs_eps, weight in zip(config.stage2_selection_abs_cfo_points, weights):
        for signed_eps in (float(abs_eps), -float(abs_eps)):
            eps_values = torch.full((symbols.shape[0],), signed_eps, device=config.device, dtype=torch.float32)
            oracle_eps = eps_values if oracle_eps_conditioning else None
            corrected_symbols, logits, z0, aux = model(
                symbols,
                eps_values,
                config,
                ebn0_db_values,
                oracle_eps_override=oracle_eps,
            )
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
    oracle_eps_conditioning: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dual_total, dual_components = _stage2_dual_head_loss(
        config,
        corrected_symbols,
        logits,
        bits,
        ref_symbols,
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
    loss_hard = torch.zeros((), device=tx_basis.device, dtype=torch.float32)
    total = dual_total
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
            ber = torch.mean((_bits_from_complex_symbols(z0, config.modulation) != bits).to(torch.float32))
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
        if config.stage2_sideinfo_enabled:
            reference_eps = model.nonlinear_receiver._coarse_eps_from_true(eps_values)
            reference_symbols = model.nonlinear_receiver._mmse_solve(z0, reference_eps, ebn0_values)
            eps_hat = aux.get("eps_hat", torch.zeros_like(reference_eps))
            return {
                "identity_symbol_mse_to_z0": float(normalized_symbol_mse(corrected_symbols, z0).item()),
                "identity_max_symbol_abs_diff": float(torch.max(torch.abs(corrected_symbols - reference_symbols)).item()),
                "identity_max_logit_abs_diff": 0.0,
                "identity_max_bit_logit_abs_diff": 0.0,
                "identity_bit_mismatch_rate_vs_stage1": float("nan"),
                "identity_label_mismatch_rate_vs_stage1": float("nan"),
                "identity_reference_ber_vs_stage1_bits": float(torch.mean((_bits_from_complex_symbols(z0, config.modulation) != bits).to(torch.float32)).item()),
                "sideinfo_init_eps_mae_to_reference": float(torch.mean(torch.abs(eps_hat - reference_eps)).item()),
                "sideinfo_init_reference_symbol_mse": float(normalized_symbol_mse(reference_symbols, corrected_symbols).item()),
            }
        stage1_logits = stage1_geometry_logits(z0)
        stage1_bit_logits = qam16_symbol_logits_to_bit_logits(stage1_logits)
        stage1_bits = _bits_from_complex_symbols(z0, config.modulation)
        stage2_bit_logits = aux.get("bit_logits", qam16_symbol_logits_to_bit_logits(logits))
        stage2_bits = _bits_from_complex_symbols(corrected_symbols, config.modulation)
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
    oracle_eps_conditioning: bool,
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
        oracle_eps = eps_values if oracle_eps_conditioning else None
        corrected_symbols, logits, z0, aux = model(
            symbols,
            eps_values,
            config,
            ebn0_values,
            oracle_eps_override=oracle_eps,
        )
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
            oracle_eps_conditioning=oracle_eps_conditioning,
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
            oracle_val_eps = val_eps if oracle_eps_conditioning else None
            corrected_val, logits_val, z0_val, aux_val = model(
                val_symbols,
                val_eps,
                config,
                val_ebn0,
                oracle_eps_override=oracle_val_eps,
            )
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
                oracle_eps_conditioning=oracle_eps_conditioning,
            )
            val_bits_hat = _stage2_bits_from_outputs(corrected_val, config.modulation)
            val_ber = torch.mean((val_bits_hat != val_bits).to(torch.float32)).item()
            val_ser = _stage2_symbol_error_rate(corrected_val, val_bits, config.modulation).item()
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
                    oracle_eps_conditioning=oracle_eps_conditioning,
                ).item(),
                "baseline_hard_cfo_weighted_ber": val_stage1_hard_ber,
                "residual_scale": float(
                    aux_val.get("solver_blend", model.nonlinear_receiver.residual_scale().to(torch.float32)).mean().item()
                ),
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
        oracle_val_eps = val_eps if oracle_eps_conditioning else None
        corrected_val, logits_val, z0_val, aux_val = model(
            val_symbols,
            val_eps,
            config,
            val_ebn0,
            oracle_eps_override=oracle_val_eps,
        )
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
            oracle_eps_conditioning=oracle_eps_conditioning,
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
            "val_ber": float(torch.mean((_stage2_bits_from_outputs(corrected_val, config.modulation) != val_bits).to(torch.float32)).item()),
            "val_ser": float(_stage2_symbol_error_rate(corrected_val, val_bits, config.modulation).item()),
            "hard_cfo_weighted_ber": float(
                _stage2_hard_cfo_weighted_ber(
                    config,
                    model,
                    val_bits,
                    val_symbols,
                    val_ebn0,
                    oracle_eps_conditioning=oracle_eps_conditioning,
                ).item()
            ),
            "baseline_hard_cfo_weighted_ber": float(val_stage1_hard_ber),
            "residual_scale": float(
                aux_val.get("solver_blend", model.nonlinear_receiver.residual_scale().to(torch.float32)).mean().item()
            ),
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
    oracle_eps_conditioning: bool = False,
) -> Stage2ReceiverTrainingResult:
    _require_stage2_supported_config(config)
    set_seed(config.base_seed + seed_offset)

    tx_basis = tx_basis.detach().clone().to(config.device)
    rx_basis = rx_basis.detach().clone().to(config.device)
    tx_init, tx_support_frame = _recover_stage1_tx_init(config, tx_basis)
    receiver = CfoEstimatorMmseReceiver(
        tx_basis=tx_basis,
        rx_basis=rx_basis,
        num_symbols=config.N,
        bits_per_symbol=config.bits_per_symbol,
        architecture=config.stage2_detector_arch,
        channels=config.stage2_local_channels,
        hidden_multiplier=config.stage2_hidden_multiplier,
        kernel_size=config.stage2_local_kernel_size,
        max_abs_eps=_stage2_estimator_max_abs_cfo(config),
        sideinfo_enabled=config.stage2_sideinfo_enabled,
        sideinfo_mode=config.stage2_sideinfo_mode,
        sideinfo_scale=config.stage2_sideinfo_scale,
        sideinfo_residual_enabled=config.stage2_sideinfo_residual_enabled,
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
    if config.stage2_sideinfo_enabled:
        if (
            identity_diag["identity_max_symbol_abs_diff"] > 1e-7
            or identity_diag.get("sideinfo_init_eps_mae_to_reference", 0.0) > 1e-7
            or identity_diag.get("sideinfo_init_reference_symbol_mse", 0.0) > 1e-7
        ):
            raise ValueError(f"Stage 2 side-information initialization check failed: {identity_diag}")
        _log_progress(
            config,
            (
                f"[Stage2] Side-info init diagnostic | coarse_mode={config.stage2_sideinfo_mode} "
                f"scale={config.stage2_sideinfo_scale:.3f} "
                f"| ref_symbol_mse={identity_diag['sideinfo_init_reference_symbol_mse']:.4e} "
                f"eps_mae={identity_diag['sideinfo_init_eps_mae_to_reference']:.4e}"
            ),
        )
    else:
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
    _, stage_row, global_epoch = _run_stage2_training_stage(
        config=config,
        model=model,
        stage=StageSpec("FrozenPostV", config.stage2_nonlinear_only_epochs, config.stage_c_cfo, True),
        scheme_name=scheme_name,
        epochs=config.stage2_nonlinear_only_epochs,
        learning_rate=config.stage2_nonlinear_only_learning_rate,
        baseline_tx_basis=tx_basis,
        baseline_rx_basis=rx_basis,
        train_tx=False,
        train_rx=False,
        include_spectral_loss=False,
        oracle_eps_conditioning=oracle_eps_conditioning,
        global_epoch_start=global_epoch,
        history=history,
    )
    stage_row.update(identity_diag)
    stage_rows.append(stage_row)
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
    ebn0_db: float | torch.Tensor | None = None,
    eval_seed: int | None = None,
) -> dict[str, torch.Tensor | float]:
    rx_basis = scheme.rx_basis
    pilot_symbol_mse = float("nan")
    cfo_est_mae = float("nan")
    cfo_est_bias = float("nan")
    mean_estimated_eps = float("nan")
    stage2_eps_hat_mae = float("nan")
    stage2_eps_hat_bias = float("nan")
    stage2_eps_hat_mean = float("nan")

    if config.frame_structure_enabled and config.pilot_estimation_enabled and not config.payload_region_enabled:
        estimated_eps, pilot_mse = estimate_residual_cfo_from_pilots(config, rx_signal, rx_basis)
        rx_signal = apply_residual_cfo(rx_signal, -estimated_eps)
        pilot_symbol_mse = float(pilot_mse.mean().item())
        if eps_tensor is not None:
            cfo_error = estimated_eps - eps_tensor
            cfo_est_mae = float(torch.mean(torch.abs(cfo_error)).item())
            cfo_est_bias = float(torch.mean(cfo_error).item())
            mean_estimated_eps = float(torch.mean(estimated_eps).item())

    if scheme.oracle_pre_v_cfo:
        if eps_tensor is None:
            raise ValueError("oracle_pre_v_cfo requires the true residual CFO tensor.")
        rx_signal = apply_residual_cfo(rx_signal, -eps_tensor)

    z0 = decode_symbols(rx_signal, rx_basis)
    logits = None
    if scheme.oracle_post_v_mmse:
        if eps_tensor is None:
            raise ValueError("oracle_post_v_mmse requires the true residual CFO tensor.")
        if ebn0_db is None:
            raise ValueError("oracle_post_v_mmse requires the active Eb/N0 for regularization.")
        alpha = scheme.oracle_mmse_alpha
        if alpha is None:
            if isinstance(ebn0_db, torch.Tensor):
                alpha = float(torch.mean(1.0 / (config.bits_per_symbol * torch.pow(10.0, ebn0_db.to(torch.float32) / 10.0))).item())
            else:
                alpha = ebn0_to_noise_variance(float(ebn0_db), config.bits_per_symbol)
        eps_for_mmse = eps_tensor
        if scheme.oracle_mmse_eps_scale is not None:
            eps_for_mmse = scheme.oracle_mmse_eps_scale * eps_tensor
        if scheme.oracle_mmse_eps_noise_std_rel is not None:
            noise_std_rel = float(scheme.oracle_mmse_eps_noise_std_rel)
            if noise_std_rel < 0.0:
                raise ValueError("oracle_mmse_eps_noise_std_rel must be non-negative.")
            if noise_std_rel > 0.0:
                noise_std = noise_std_rel * torch.abs(eps_tensor.to(torch.float32))
                if eval_seed is not None:
                    generator_device = eps_tensor.device if eps_tensor.is_cuda else torch.device("cpu")
                    generator = torch.Generator(device=generator_device)
                    generator.manual_seed(int(eval_seed) + int(scheme.oracle_mmse_eps_noise_seed_offset))
                    eps_noise = torch.randn(
                        eps_tensor.shape,
                        generator=generator,
                        device=eps_tensor.device,
                        dtype=torch.float32,
                    )
                else:
                    eps_noise = torch.randn(eps_tensor.shape, device=eps_tensor.device, dtype=torch.float32)
                eps_for_mmse = eps_for_mmse.to(torch.float32) + noise_std * eps_noise
                eps_for_mmse = eps_for_mmse.to(eps_tensor.dtype)
        corrected_symbols = oracle_symbol_mmse_equalize(z0, scheme.tx_basis, scheme.rx_basis, eps_for_mmse, alpha)
        aux = {}
        bits_hat_full, _ = slice_symbols(corrected_symbols, config.modulation)
    elif scheme.nonlinear_receiver is not None:
        oracle_eps = eps_tensor if scheme.oracle_eps_conditioning else None
        if ebn0_db is None:
            raise ValueError("Nonlinear Stage 2 inference requires the active Eb/N0 for regularization.")
        corrected_symbols, logits, aux = scheme.nonlinear_receiver(
            z0,
            ebn0_db_values=ebn0_db,
            eps_override=oracle_eps,
            eps_true=eps_tensor,
        )
        if config.modulation != "16QAM":
            raise ValueError("Nonlinear receiver inference is currently implemented only for 16QAM.")
        bits_hat_full = _stage2_bits_from_outputs(corrected_symbols, config.modulation)
        if eps_tensor is not None and "eps_hat" in aux:
            eps_error = aux["eps_hat"].to(torch.float32) - eps_tensor.to(torch.float32)
            stage2_eps_hat_mae = float(torch.mean(torch.abs(eps_error)).item())
            stage2_eps_hat_bias = float(torch.mean(eps_error).item())
            stage2_eps_hat_mean = float(torch.mean(aux["eps_hat"].to(torch.float32)).item())
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
        "stage2_eps_hat_mae": stage2_eps_hat_mae,
        "stage2_eps_hat_bias": stage2_eps_hat_bias,
        "stage2_eps_hat_mean": stage2_eps_hat_mean,
    }


def evaluate_single(
    config: ExperimentConfig,
    scheme: EvaluationScheme,
    eps: float,
    ebn0_db: float,
    bits: torch.Tensor,
    symbols: torch.Tensor,
    noise: torch.Tensor,
    eval_seed: int | None = None,
) -> dict[str, float]:
    eps_tensor = torch.full((symbols.shape[0],), float(eps), device=symbols.device, dtype=torch.float32)
    tx_signal = transmit_symbols(symbols, scheme.tx_basis)
    rx_signal, _ = propagate(tx_signal, eps_tensor, config, ebn0_db=ebn0_db, noise=noise)
    outputs = detect_scheme_symbols(
        config,
        scheme,
        rx_signal,
        eps_tensor=eps_tensor,
        ebn0_db=ebn0_db,
        eval_seed=eval_seed,
    )
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
        "stage2_eps_hat_mae": float(outputs["stage2_eps_hat_mae"]),
        "stage2_eps_hat_bias": float(outputs["stage2_eps_hat_bias"]),
        "stage2_eps_hat_mean": float(outputs["stage2_eps_hat_mean"]),
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
                "stage2_eps_hat_mae_sum": 0.0,
                "stage2_eps_hat_bias_sum": 0.0,
                "stage2_eps_hat_mean_sum": 0.0,
                "pilot_batches": 0,
                "stage2_batches": 0,
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
            batch_eval_seed = seed + 1_000_000 * eps_idx + processed
            for name, scheme in schemes.items():
                metrics = evaluate_single(
                    config=config,
                    scheme=scheme,
                    eps=float(eps),
                    ebn0_db=ebn0_db,
                    bits=bits,
                    symbols=symbols,
                    noise=noise,
                    eval_seed=batch_eval_seed,
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
                if np.isfinite(metrics["stage2_eps_hat_mae"]):
                    accum[name]["stage2_eps_hat_mae_sum"] += metrics["stage2_eps_hat_mae"]
                    accum[name]["stage2_eps_hat_bias_sum"] += metrics["stage2_eps_hat_bias"]
                    accum[name]["stage2_eps_hat_mean_sum"] += metrics["stage2_eps_hat_mean"]
                    accum[name]["stage2_batches"] += 1
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
            stage2_count = max(entry["stage2_batches"], 1)
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
                    "stage2_eps_hat_mae": (
                        entry["stage2_eps_hat_mae_sum"] / stage2_count if entry["stage2_batches"] > 0 else np.nan
                    ),
                    "stage2_eps_hat_bias": (
                        entry["stage2_eps_hat_bias_sum"] / stage2_count if entry["stage2_batches"] > 0 else np.nan
                    ),
                    "stage2_eps_hat_mean": (
                        entry["stage2_eps_hat_mean_sum"] / stage2_count if entry["stage2_batches"] > 0 else np.nan
                    ),
                }
            )
    if progress_label:
        _log_progress(config, f"[Eval] {progress_label} complete")
    return pd.DataFrame(rows)
