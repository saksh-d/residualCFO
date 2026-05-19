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

from channel import propagate
from comm_core import ExperimentConfig, log_terminal_progress, set_seed
from receiver import (
    EvaluationScheme,
    _sample_condition_noise_like,
    decode_symbols,
    normalized_symbol_mse,
    qam16_constellation_points,
    qam16_labels_from_bits,
    qam16_slicer,
    qam16_symbol_logits_to_bit_logits,
    slice_symbols,
    stage1_geometry_logits,
)
from transmitter import sample_training_symbols, transmit_symbols


DEFAULT_CONDITION_SIGMA = 0.005
DEFAULT_CONDITION_SENSITIVITY = (0.0, 0.0025, 0.0050, 0.0100, 0.0200, 0.0300, 0.0500)
DEFAULT_SUMMARY_DELTA_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
DEFAULT_SNR_GRID = (0.0, 5.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_VALIDATION_DELTAS = (0.0, 0.05, 0.10)
DEFAULT_PHASE2_SIGMAS = (0.0, 0.0025, 0.0050, 0.0100, 0.0200)


def _inverse_sigmoid(prob: float) -> float:
    prob = min(max(float(prob), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(prob / (1.0 - prob))


def _complex_gain(real_part: torch.Tensor, imag_part: torch.Tensor) -> torch.Tensor:
    return torch.complex(real_part, imag_part).to(torch.complex64)


def _normalized_batch_mse(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    denom = torch.mean(torch.abs(ref) ** 2, dim=-1).real.clamp_min(1.0e-12)
    numer = torch.mean(torch.abs(est - ref) ** 2, dim=-1).real
    return torch.mean(numer / denom)


def _normalized_reference_mse(est: torch.Tensor, ref: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    denom = torch.mean(torch.abs(anchor) ** 2, dim=-1).real.clamp_min(1.0e-12)
    numer = torch.mean(torch.abs(est - ref) ** 2, dim=-1).real
    return torch.mean(numer / denom)


def correction_norm(est: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    numer = torch.linalg.vector_norm(est - z0, dim=-1)
    denom = torch.linalg.vector_norm(z0, dim=-1).clamp_min(1.0e-12)
    return torch.mean((numer / denom).real)


def _branch_rms(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values.to(torch.float32) ** 2) + 1.0e-12)


def _constellation_logits(symbols: torch.Tensor, temp_cls: float) -> torch.Tensor:
    constellation = qam16_constellation_points(symbols.device, dtype=symbols.dtype)
    dist_sq = torch.abs(symbols[..., None] - constellation[None, None, :]) ** 2
    return -(dist_sq.real.to(torch.float32) / float(temp_cls))


@dataclass(frozen=True)
class GRRNetArchitectureConfig:
    conditioned: bool = True
    rank: int = 16
    num_symbols: int = 45
    local_channels: int = 32
    conv_hidden_channels: int = 64
    conv_kernel_size: int = 3
    conv_dilations: tuple[int, ...] = (1, 2, 4)
    alpha_init: float = 0.05
    final_gain_enabled: bool = True
    eval_condition_sigma: float = DEFAULT_CONDITION_SIGMA

    def __post_init__(self) -> None:
        if self.rank not in (4, 8, 16, 32):
            raise ValueError("GRR-Net rank must be one of 4, 8, 16, or 32.")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be positive.")
        if self.local_channels < 1 or self.conv_hidden_channels < 1:
            raise ValueError("Channel counts must be positive.")
        if self.conv_kernel_size < 1 or self.conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer.")
        if tuple(self.conv_dilations) != (1, 2, 4):
            raise ValueError("GRR-Net expects three dilated Conv1d layers with dilations (1, 2, 4).")

    @property
    def feature_dim(self) -> int:
        return 14 if self.conditioned else 10


@dataclass(frozen=True)
class GRRNetLossConfig:
    temp_cls: float = 0.1
    cls_weight: float = 0.3
    identity_weight: float = 0.1
    correction_weight: float = 1.0e-4
    identity_delta_threshold: float = 0.02
    weight_decay: float = 0.0


@dataclass(frozen=True)
class GRRNetPhaseConfig:
    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    delta_span: float
    ebn0_choices: tuple[float, ...]
    sigma_condition: float | tuple[float, ...]


@dataclass(frozen=True)
class GRRNetValidationConfig:
    delta_values: tuple[float, ...] = DEFAULT_VALIDATION_DELTAS
    ebn0_db: float = 10.0
    sigma_condition: float = DEFAULT_CONDITION_SIGMA
    num_blocks: int = 1024
    batch_size: int = 256


@dataclass
class GRRNetTrainingBundle:
    receiver: "GlobalResidualRefiner"
    history_df: pd.DataFrame
    stage_summary_df: pd.DataFrame


class GlobalResidualRefiner(nn.Module):
    def __init__(self, architecture_config: GRRNetArchitectureConfig) -> None:
        super().__init__()
        self.conditioned = bool(architecture_config.conditioned)
        self.rank = int(architecture_config.rank)
        self.num_symbols = int(architecture_config.num_symbols)
        self.feature_dim = int(architecture_config.feature_dim)
        self.local_channels = int(architecture_config.local_channels)
        self.eval_condition_sigma = float(architecture_config.eval_condition_sigma)
        self.final_gain_enabled = bool(architecture_config.final_gain_enabled)

        self.local_proj = nn.Linear(self.feature_dim, self.local_channels)
        self.global_down = nn.Linear(self.num_symbols * self.feature_dim, self.rank)
        self.global_up = nn.Linear(self.rank, self.num_symbols * self.local_channels)
        self.conv1 = nn.Conv1d(self.local_channels, architecture_config.conv_hidden_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(architecture_config.conv_hidden_channels, architecture_config.conv_hidden_channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(architecture_config.conv_hidden_channels, architecture_config.conv_hidden_channels, kernel_size=3, dilation=4, padding=4)
        self.conv_out = nn.Conv1d(architecture_config.conv_hidden_channels, 2, kernel_size=1)

        alpha_prob = min(max(float(architecture_config.alpha_init) / 0.2, 1.0e-4), 1.0 - 1.0e-4)
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_sigmoid(alpha_prob), dtype=torch.float32))
        self.final_gain_re = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.final_gain_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def parameter_summary(self) -> dict[str, float | bool | int]:
        return {
            "conditioned": self.conditioned,
            "rank": self.rank,
            "alpha": float((0.2 * torch.sigmoid(self.raw_alpha)).item()),
            "final_gain_re": float(self.final_gain_re.item()),
            "final_gain_im": float(self.final_gain_im.item()),
        }

    def _gain(self) -> torch.Tensor:
        if not self.final_gain_enabled:
            return torch.ones((), device=self.final_gain_re.device, dtype=torch.complex64)
        return _complex_gain(self.final_gain_re, self.final_gain_im)

    def _build_features(self, z0: torch.Tensor, delta_condition: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditioned and delta_condition is None:
            raise ValueError("Conditioned GRR-Net requires a condition tensor.")
        phase = torch.angle(z0).to(torch.float32)
        _, q_nearest = qam16_slicer(z0)
        error = z0 - q_nearest
        feature_list = [
            torch.real(z0).to(torch.float32),
            torch.imag(z0).to(torch.float32),
            torch.abs(z0).to(torch.float32),
            torch.cos(phase),
            torch.sin(phase),
            torch.real(q_nearest).to(torch.float32),
            torch.imag(q_nearest).to(torch.float32),
            torch.real(error).to(torch.float32),
            torch.imag(error).to(torch.float32),
            (torch.abs(error) ** 2).to(torch.float32),
        ]
        if self.conditioned:
            assert delta_condition is not None
            cond = delta_condition.to(device=z0.device, dtype=torch.float32)
            cond_base = cond[:, None].expand(-1, z0.shape[1])
            feature_list.extend(
                [
                    cond_base,
                    cond_base ** 2,
                    torch.sin(math.pi * cond_base),
                    torch.cos(math.pi * cond_base),
                ]
            )
        return torch.stack(feature_list, dim=-1).to(torch.float32)

    def forward_refine(
        self,
        z0: torch.Tensor,
        delta_condition: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]] | torch.Tensor:
        features = self._build_features(z0, delta_condition)
        local = F.gelu(self.local_proj(features))
        flat = features.reshape(features.shape[0], -1)
        global_mixed = F.gelu(self.global_down(flat))
        global_mixed = self.global_up(global_mixed).reshape(features.shape[0], self.num_symbols, self.local_channels)
        combined = local + global_mixed
        conv_in = combined.transpose(1, 2)
        conv_hidden1 = F.gelu(self.conv1(conv_in))
        conv_hidden2 = F.gelu(self.conv2(conv_hidden1))
        conv_hidden3 = F.gelu(self.conv3(conv_hidden2))
        residual_real = self.conv_out(conv_hidden3)
        residual_complex = torch.complex(residual_real[:, 0, :], residual_real[:, 1, :]).to(torch.complex64)

        alpha = 0.2 * torch.sigmoid(self.raw_alpha)
        pre_gain = z0 + alpha.to(torch.complex64) * residual_complex
        corrected = self._gain() * pre_gain

        aux = {
            "alpha": alpha.detach().to(torch.float32),
            "condition_feature": None if delta_condition is None else delta_condition.detach().to(torch.float32),
            "local_norm": _branch_rms(local).detach().to(torch.float32),
            "global_norm": _branch_rms(global_mixed).detach().to(torch.float32),
            "combined_norm": _branch_rms(combined).detach().to(torch.float32),
            "conv_norm": correction_norm(z0 + residual_complex, z0).detach().to(torch.float32),
            "correction_norm": correction_norm(corrected, z0).detach().to(torch.float32),
        }
        return (corrected, aux) if return_aux else corrected

    def forward(
        self,
        z0: torch.Tensor,
        *,
        ebn0_db_values: float | torch.Tensor | None = None,
        eps_override: torch.Tensor | None = None,
        eps_true: torch.Tensor | None = None,
        eval_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        delta_condition = None
        condition_mae = torch.tensor(float("nan"), device=z0.device, dtype=torch.float32)
        if self.conditioned:
            base_condition = eps_override if eps_override is not None else eps_true
            if base_condition is None:
                raise ValueError("Conditioned GRR-Net requires a residual-CFO condition tensor.")
            base_condition = base_condition.to(device=z0.device, dtype=torch.float32)
            delta_condition = base_condition + _sample_condition_noise_like(
                base_condition,
                self.eval_condition_sigma,
                eval_seed=eval_seed,
                seed_offset=707,
            )
            condition_mae = torch.mean(torch.abs(delta_condition - base_condition))
        corrected, aux = self.forward_refine(z0, delta_condition, return_aux=True)
        logits = stage1_geometry_logits(corrected)
        aux["bit_logits"] = qam16_symbol_logits_to_bit_logits(logits)
        aux["condition_mae"] = condition_mae.detach().to(torch.float32)
        return corrected, logits, aux


def _sample_training_batch(
    config: ExperimentConfig,
    W_tx: torch.Tensor,
    V: torch.Tensor,
    *,
    batch_size: int,
    delta_span: float,
    ebn0_choices: tuple[float, ...],
    sigma_condition: float | tuple[float, ...],
    fixed_delta: float | None = None,
    fixed_ebn0_db: float | None = None,
) -> dict[str, torch.Tensor]:
    true_bits, true_symbols = sample_training_symbols(batch_size, config)
    if fixed_delta is None:
        delta_true = (2.0 * torch.rand(batch_size, device=config.device) - 1.0) * float(delta_span)
    else:
        delta_true = torch.full((batch_size,), float(fixed_delta), device=config.device, dtype=torch.float32)
    if fixed_ebn0_db is None:
        ebn0_idx = torch.randint(0, len(ebn0_choices), (batch_size,), device=config.device)
        ebn0_values = torch.tensor(ebn0_choices, device=config.device, dtype=torch.float32)[ebn0_idx]
    else:
        ebn0_values = torch.full((batch_size,), float(fixed_ebn0_db), device=config.device, dtype=torch.float32)
    x = transmit_symbols(true_symbols, W_tx)
    y, _ = propagate(x, delta_true, config, ebn0_db=ebn0_values)
    z0 = decode_symbols(y, V)
    if isinstance(sigma_condition, tuple):
        sigma_idx = torch.randint(0, len(sigma_condition), (batch_size,), device=config.device)
        sigma_values = torch.tensor(sigma_condition, device=config.device, dtype=torch.float32)[sigma_idx]
    else:
        sigma_values = torch.full((batch_size,), float(sigma_condition), device=config.device, dtype=torch.float32)
    delta_condition = delta_true + sigma_values * torch.randn_like(delta_true)
    return {
        "true_bits": true_bits,
        "true_symbols": true_symbols,
        "delta_true": delta_true.to(torch.float32),
        "ebn0_values": ebn0_values.to(torch.float32),
        "z0": z0.to(torch.complex64),
        "delta_condition": delta_condition.to(torch.float32),
        "sigma_values": sigma_values.to(torch.float32),
    }


def _grrnet_loss(
    receiver: GlobalResidualRefiner,
    batch: dict[str, torch.Tensor],
    *,
    loss_config: GRRNetLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    corrected, aux = receiver.forward_refine(
        batch["z0"],
        batch["delta_condition"] if receiver.conditioned else None,
        return_aux=True,
    )
    logits = _constellation_logits(corrected, temp_cls=loss_config.temp_cls)
    labels = qam16_labels_from_bits(batch["true_bits"])
    loss_sym = _normalized_batch_mse(corrected, batch["true_symbols"])
    loss_cls = F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1))
    mask = torch.abs(batch["delta_true"]) < float(loss_config.identity_delta_threshold)
    if bool(mask.any()):
        loss_id = _normalized_reference_mse(corrected[mask], batch["z0"][mask], batch["z0"][mask])
    else:
        loss_id = torch.zeros((), device=corrected.device, dtype=torch.float32)
    loss_corr = _normalized_reference_mse(corrected, batch["z0"], batch["z0"])
    total = (
        loss_sym
        + float(loss_config.cls_weight) * loss_cls
        + float(loss_config.identity_weight) * loss_id
        + float(loss_config.correction_weight) * loss_corr
    )
    metrics = {
        "train_total": float(total.item()),
        "train_Lsym": float(loss_sym.item()),
        "train_Lcls": float(loss_cls.item()),
        "train_Lid": float(loss_id.item()),
        "train_Lcorr": float(loss_corr.item()),
        "correction_norm": float(aux["correction_norm"]),
        "alpha": float(aux["alpha"].item()),
        "local_norm": float(aux["local_norm"].item()),
        "global_norm": float(aux["global_norm"].item()),
        "combined_norm": float(aux["combined_norm"].item()),
        "conv_norm": float(aux["conv_norm"].item()),
        "baseline_evm": float(normalized_symbol_mse(batch["z0"], batch["true_symbols"]).item()),
        "corrected_evm": float(normalized_symbol_mse(corrected, batch["true_symbols"]).item()),
    }
    return total, metrics


def evaluate_grrnet_scheme(
    config: ExperimentConfig,
    W_tx: torch.Tensor,
    V: torch.Tensor,
    receiver: GlobalResidualRefiner | None,
    *,
    method_name: str,
    delta_value: float,
    ebn0_db: float,
    sigma_condition: float,
    num_blocks: int,
    batch_size: int,
    seed: int,
    capture_points: int = 0,
) -> dict[str, object]:
    set_seed(seed)
    total_bits = 0
    bit_errors = 0
    total_batches = 0
    evm_sum = 0.0
    correction_sum = 0.0
    condition_mae_sum = 0.0
    alpha_sum = 0.0
    local_sum = 0.0
    global_sum = 0.0
    combined_sum = 0.0
    conv_sum = 0.0
    constellation_points: list[tuple[float, float]] = []

    remaining = int(num_blocks)
    batch_index = 0
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        remaining -= current_batch
        batch_index += 1
        true_bits, true_symbols = sample_training_symbols(current_batch, config)
        delta_true = torch.full((current_batch,), float(delta_value), device=config.device, dtype=torch.float32)
        x = transmit_symbols(true_symbols, W_tx)
        y, _ = propagate(x, delta_true, config, ebn0_db=float(ebn0_db))
        z0 = decode_symbols(y, V)
        if receiver is None:
            corrected = z0
            aux = {
                "alpha": torch.tensor(0.0, device=config.device),
                "condition_mae": torch.tensor(float("nan"), device=config.device),
                "local_norm": torch.tensor(0.0, device=config.device),
                "global_norm": torch.tensor(0.0, device=config.device),
                "combined_norm": torch.tensor(0.0, device=config.device),
                "conv_norm": torch.tensor(0.0, device=config.device),
                "correction_norm": torch.tensor(0.0, device=config.device),
            }
        else:
            delta_condition = delta_true + float(sigma_condition) * torch.randn_like(delta_true)
            with torch.no_grad():
                corrected, aux = receiver.forward_refine(
                    z0,
                    delta_condition if receiver.conditioned else None,
                    return_aux=True,
                )
            if receiver.conditioned:
                aux["condition_mae"] = torch.mean(torch.abs(delta_condition - delta_true)).to(torch.float32)
            else:
                aux["condition_mae"] = torch.tensor(float("nan"), device=config.device)

        bits_hat, _ = slice_symbols(corrected, config.modulation)
        total_bits += true_bits.numel()
        bit_errors += int(torch.sum(bits_hat != true_bits).item())
        total_batches += current_batch
        evm_sum += float(normalized_symbol_mse(corrected, true_symbols).item()) * current_batch
        correction_sum += float(correction_norm(corrected, z0).item()) * current_batch
        alpha_sum += float(aux["alpha"].item()) * current_batch
        local_sum += float(aux["local_norm"].item()) * current_batch
        global_sum += float(aux["global_norm"].item()) * current_batch
        combined_sum += float(aux["combined_norm"].item()) * current_batch
        conv_sum += float(aux["conv_norm"].item()) * current_batch
        if torch.isfinite(aux["condition_mae"]):
            condition_mae_sum += float(aux["condition_mae"].item()) * current_batch

        if capture_points > 0 and len(constellation_points) < capture_points:
            flattened = corrected.reshape(-1)
            take_count = min(capture_points - len(constellation_points), flattened.numel())
            for point in flattened[:take_count]:
                constellation_points.append((float(torch.real(point).item()), float(torch.imag(point).item())))

    denom = max(1, total_batches)
    return {
        "method": method_name,
        "delta": float(delta_value),
        "BER": float(bit_errors / max(1, total_bits)),
        "EVM": float(evm_sum / denom),
        "correction_norm": float(correction_sum / denom),
        "condition_mae": float(condition_mae_sum / denom) if receiver is not None and receiver.conditioned else float("nan"),
        "alpha": float(alpha_sum / denom),
        "local_norm": float(local_sum / denom),
        "global_norm": float(global_sum / denom),
        "combined_norm": float(combined_sum / denom),
        "conv_norm": float(conv_sum / denom),
        "constellation_points": constellation_points,
    }


def _evaluate_validation_grid(
    config: ExperimentConfig,
    W_tx: torch.Tensor,
    V: torch.Tensor,
    receiver: GlobalResidualRefiner,
    *,
    validation_config: GRRNetValidationConfig,
    seed: int,
) -> dict[str, float]:
    rows = []
    for delta_idx, delta_value in enumerate(validation_config.delta_values):
        rows.append(
            evaluate_grrnet_scheme(
                config,
                W_tx,
                V,
                receiver,
                method_name="Validation",
                delta_value=float(delta_value),
                ebn0_db=float(validation_config.ebn0_db),
                sigma_condition=float(validation_config.sigma_condition),
                num_blocks=int(validation_config.num_blocks),
                batch_size=int(validation_config.batch_size),
                seed=int(seed) + 10_000 * delta_idx,
            )
        )
    frame = pd.DataFrame(rows)
    hard_mask = frame["delta"] >= 0.05
    val_ber = float(frame["BER"].mean())
    val_evm = float(frame["EVM"].mean())
    hard_cfo_weighted_ber = float(frame.loc[hard_mask, "BER"].mean()) if hard_mask.any() else val_ber
    val_total = val_ber + 0.25 * val_evm
    return {
        "val_total": val_total,
        "val_ber": val_ber,
        "val_evm": val_evm,
        "hard_cfo_weighted_ber": hard_cfo_weighted_ber,
    }


def run_grrnet_sanity_check(
    config: ExperimentConfig,
    W_tx: torch.Tensor,
    V: torch.Tensor,
    *,
    architecture_config: GRRNetArchitectureConfig,
    loss_config: GRRNetLossConfig,
    epochs: int,
    batch_size: int = 512,
    seed_offset: int = 0,
    scheme_name: str = "LearnedGRRNetSanity",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool]]:
    set_seed(int(config.base_seed) + int(seed_offset))
    receiver = GlobalResidualRefiner(architecture_config).to(config.device)
    optimizer = torch.optim.Adam(
        receiver.parameters(),
        lr=1.0e-3,
        weight_decay=float(loss_config.weight_decay),
    )
    batch = _sample_training_batch(
        config,
        W_tx,
        V,
        batch_size=int(batch_size),
        delta_span=0.10,
        ebn0_choices=(20.0,),
        sigma_condition=(DEFAULT_CONDITION_SIGMA if architecture_config.conditioned else 0.0),
        fixed_delta=0.10,
        fixed_ebn0_db=20.0,
    )
    history_rows: list[dict[str, object]] = []
    best_evm = float("inf")
    best_epoch = 0
    start = perf_counter()
    for epoch_idx in range(int(epochs)):
        receiver.train()
        optimizer.zero_grad(set_to_none=True)
        total, metrics = _grrnet_loss(receiver, batch, loss_config=loss_config)
        total.backward()
        optimizer.step()
        corrected, _ = receiver.forward_refine(
            batch["z0"],
            batch["delta_condition"] if receiver.conditioned else None,
            return_aux=True,
        )
        corrected_evm = float(normalized_symbol_mse(corrected, batch["true_symbols"]).item())
        baseline_evm = float(normalized_symbol_mse(batch["z0"], batch["true_symbols"]).item())
        if corrected_evm < best_evm:
            best_evm = corrected_evm
            best_epoch = epoch_idx + 1
        history_rows.append(
            {
                "scheme": scheme_name,
                "stage": "Phase0SanityOverfit",
                "phase": "Phase0SanityOverfit",
                "global_epoch": epoch_idx + 1,
                "train_total": metrics["train_total"],
                "train_Lsym": metrics["train_Lsym"],
                "train_Lcls": metrics["train_Lcls"],
                "train_Lid": metrics["train_Lid"],
                "train_Lcorr": metrics["train_Lcorr"],
                "val_total": float("nan"),
                "val_ber": float("nan"),
                "val_evm": corrected_evm,
                "baseline_evm": baseline_evm,
                "correction_norm": metrics["correction_norm"],
                "alpha": metrics["alpha"],
                "local_norm": metrics["local_norm"],
                "global_norm": metrics["global_norm"],
                "combined_norm": metrics["combined_norm"],
                "conv_norm": metrics["conv_norm"],
            }
        )
    final_evm = float(history_rows[-1]["val_evm"]) if history_rows else float("nan")
    baseline_evm = float(history_rows[-1]["baseline_evm"]) if history_rows else float("nan")
    summary_df = pd.DataFrame(
        [
            {
                "scheme": scheme_name,
                "stage": "Phase0SanityOverfit",
                "best_global_epoch": best_epoch,
                "best_stage_epoch": best_epoch,
                "val_total": best_evm,
                "val_ber": float("nan"),
                "val_evm": best_evm,
                "hard_cfo_weighted_ber": float("nan"),
                "elapsed_seconds": float(perf_counter() - start),
                "alpha": receiver.parameter_summary()["alpha"],
                "final_gain_re": receiver.parameter_summary()["final_gain_re"],
                "final_gain_im": receiver.parameter_summary()["final_gain_im"],
            }
        ]
    )
    summary = {
        "baseline_evm": baseline_evm,
        "best_evm": best_evm,
        "final_evm": final_evm,
        "passed": bool(final_evm < baseline_evm),
    }
    return pd.DataFrame(history_rows), summary_df, summary


def train_grrnet_receiver(
    config: ExperimentConfig,
    W_tx: torch.Tensor,
    V: torch.Tensor,
    *,
    scheme_name: str,
    architecture_config: GRRNetArchitectureConfig,
    phase_configs: tuple[GRRNetPhaseConfig, ...],
    validation_config: GRRNetValidationConfig,
    loss_config: GRRNetLossConfig,
    seed_offset: int = 0,
) -> GRRNetTrainingBundle:
    set_seed(int(config.base_seed) + int(seed_offset))
    receiver = GlobalResidualRefiner(architecture_config).to(config.device)
    history_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    global_epoch = 0

    for phase_idx, phase in enumerate(phase_configs):
        optimizer = torch.optim.Adam(
            receiver.parameters(),
            lr=float(phase.learning_rate),
            weight_decay=float(loss_config.weight_decay),
        )
        best_val_total = float("inf")
        best_row: dict[str, object] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        phase_start = perf_counter()

        for epoch_idx in range(int(phase.epochs)):
            receiver.train()
            batch = _sample_training_batch(
                config,
                W_tx,
                V,
                batch_size=int(phase.batch_size),
                delta_span=float(phase.delta_span),
                ebn0_choices=tuple(float(value) for value in phase.ebn0_choices),
                sigma_condition=phase.sigma_condition,
            )
            optimizer.zero_grad(set_to_none=True)
            total, metrics = _grrnet_loss(receiver, batch, loss_config=loss_config)
            total.backward()
            optimizer.step()
            global_epoch += 1

            val_metrics = _evaluate_validation_grid(
                config,
                W_tx,
                V,
                receiver,
                validation_config=validation_config,
                seed=int(config.base_seed) + 40_000 + 1_000 * phase_idx + epoch_idx,
            )
            params = receiver.parameter_summary()
            history_rows.append(
                {
                    "scheme": scheme_name,
                    "stage": phase.name,
                    "phase": phase.name,
                    "global_epoch": global_epoch,
                    "train_total": metrics["train_total"],
                    "train_Lsym": metrics["train_Lsym"],
                    "train_Lcls": metrics["train_Lcls"],
                    "train_Lid": metrics["train_Lid"],
                    "train_Lcorr": metrics["train_Lcorr"],
                    "val_total": val_metrics["val_total"],
                    "val_ber": val_metrics["val_ber"],
                    "val_evm": val_metrics["val_evm"],
                    "baseline_evm": metrics["baseline_evm"],
                    "correction_norm": metrics["correction_norm"],
                    "alpha": float(params["alpha"]),
                    "local_norm": metrics["local_norm"],
                    "global_norm": metrics["global_norm"],
                    "combined_norm": metrics["combined_norm"],
                    "conv_norm": metrics["conv_norm"],
                }
            )
            if val_metrics["val_total"] < best_val_total:
                best_val_total = val_metrics["val_total"]
                best_state = copy.deepcopy(receiver.state_dict())
                best_row = {
                    "scheme": scheme_name,
                    "stage": phase.name,
                    "best_global_epoch": global_epoch,
                    "best_stage_epoch": epoch_idx + 1,
                    "val_total": val_metrics["val_total"],
                    "val_ber": val_metrics["val_ber"],
                    "val_evm": val_metrics["val_evm"],
                    "hard_cfo_weighted_ber": val_metrics["hard_cfo_weighted_ber"],
                    "elapsed_seconds": float(perf_counter() - phase_start),
                    "alpha": float(params["alpha"]),
                    "final_gain_re": float(params["final_gain_re"]),
                    "final_gain_im": float(params["final_gain_im"]),
                    "conditioned": bool(params["conditioned"]),
                    "rank": int(params["rank"]),
                }
            if epoch_idx in {0, int(phase.epochs) - 1} or (int(phase.epochs) >= 20 and (epoch_idx + 1) % 25 == 0):
                log_terminal_progress(
                    config,
                    f"[GRRNet] {scheme_name} {phase.name} epoch={epoch_idx + 1:03d}/{int(phase.epochs):03d} "
                    f"train={metrics['train_total']:.4e} val={val_metrics['val_total']:.4e} "
                    f"ber={val_metrics['val_ber']:.4e} evm={val_metrics['val_evm']:.4e} alpha={float(params['alpha']):.4f}",
                )
        if best_row is None or best_state is None:
            raise RuntimeError(f"GRR-Net phase {phase.name} did not record a best validation row.")
        receiver.load_state_dict(best_state)
        stage_rows.append(best_row)

    return GRRNetTrainingBundle(
        receiver=receiver.eval(),
        history_df=pd.DataFrame(history_rows),
        stage_summary_df=pd.DataFrame(stage_rows),
    )


def build_grrnet_schemes(
    ofdm_tx: torch.Tensor,
    ofdm_rx: torch.Tensor,
    learned_tx: torch.Tensor,
    learned_rx: torch.Tensor,
    ofdm_receiver: GlobalResidualRefiner,
    learned_receiver: GlobalResidualRefiner,
) -> dict[str, EvaluationScheme]:
    return {
        "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
        "OFDMGRRNet": EvaluationScheme(
            tx_basis=ofdm_tx,
            rx_basis=ofdm_rx,
            nonlinear_receiver=ofdm_receiver.eval(),
            oracle_eps_conditioning=bool(ofdm_receiver.conditioned),
        ),
        "Learned": EvaluationScheme(tx_basis=learned_tx, rx_basis=learned_rx, nonlinear_receiver=None),
        "LearnedGRRNet": EvaluationScheme(
            tx_basis=learned_tx,
            rx_basis=learned_rx,
            nonlinear_receiver=learned_receiver.eval(),
            oracle_eps_conditioning=bool(learned_receiver.conditioned),
        ),
    }
