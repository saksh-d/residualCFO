from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from comm_core import ExperimentConfig, normalize_columns, normalize_tone_pattern


_QAM16_LEVELS = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=torch.float32)
_QAM16_DECISION_BITS = torch.tensor(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ],
    dtype=torch.int64,
)


@dataclass(frozen=True)
class FrameResourceLayout:
    active_shift_bins: np.ndarray
    guard_shift_bins: np.ndarray
    active_bin_indices: np.ndarray
    guard_bin_indices: np.ndarray
    payload_shift_bins: np.ndarray
    payload_bin_indices: np.ndarray
    data_stream_indices: np.ndarray
    pilot_stream_indices: np.ndarray
    data_shift_bins: np.ndarray
    pilot_shift_bins: np.ndarray
    data_bin_indices: np.ndarray
    pilot_bin_indices: np.ndarray


def _fftshift_to_native_bins(shift_bins: np.ndarray, M: int) -> np.ndarray:
    return ((shift_bins + M // 2) % M).astype(int)


def active_fft_bins(M: int, N: int, pattern: str) -> np.ndarray:
    pattern = normalize_tone_pattern(pattern)
    if N > M:
        raise ValueError(f"N must be <= M for tone placement, got N={N}, M={M}")
    if pattern == "CONTIGUOUS":
        start = (M - N) // 2
        shift_bins = np.arange(start, start + N, dtype=int)
    else:
        shift_bins = np.floor(np.arange(N, dtype=float) * M / N + 0.5).astype(int)
    return _fftshift_to_native_bins(shift_bins, M)


def contiguous_band_bins_with_guard(M: int, N: int, guard_bins: int = 1) -> np.ndarray:
    if N > M:
        raise ValueError(f"N must be <= M for tone placement, got N={N}, M={M}")
    start = (M - N) // 2
    stop = start + N
    shift_bins = np.arange(start - guard_bins, stop + guard_bins, dtype=int)
    shift_bins = np.mod(shift_bins, M)
    return _fftshift_to_native_bins(shift_bins, M)


def _centered_active_shift_bins(M: int, N_active: int) -> np.ndarray:
    if N_active > M:
        raise ValueError(f"N_active must be <= M, got N_active={N_active}, M={M}")
    start = (M - N_active) // 2
    return np.arange(start, start + N_active, dtype=int)


def _evenly_spaced_stream_indices(N_active: int, N_pilots: int) -> np.ndarray:
    positions = np.linspace(0, N_active - 1, N_pilots + 2, dtype=float)[1:-1]
    indices = np.round(positions).astype(int)
    indices = np.clip(indices, 0, N_active - 1)
    indices = np.unique(indices)
    if indices.size != N_pilots:
        indices = np.linspace(1, N_active - 2, N_pilots, dtype=int)
        indices = np.unique(indices)
    if indices.size != N_pilots:
        raise ValueError("Unable to generate the requested number of pilot stream indices.")
    return np.sort(indices)


def frame_resource_layout(config: ExperimentConfig) -> FrameResourceLayout:
    if not config.frame_structure_enabled:
        raise ValueError("frame_resource_layout requires frame_structure_enabled=True.")

    active_shift_bins = _centered_active_shift_bins(config.M, config.active_resource_bins)
    guard_shift_bins = np.setdiff1d(np.arange(config.M, dtype=int), active_shift_bins, assume_unique=True)
    active_bin_indices = _fftshift_to_native_bins(active_shift_bins, config.M)
    guard_bin_indices = _fftshift_to_native_bins(guard_shift_bins, config.M)

    if config.payload_region_enabled:
        pilot_edge = config.N_pilots // 2
        payload_start = pilot_edge
        payload_stop = payload_start + config.K
        payload_shift_bins = active_shift_bins[payload_start:payload_stop]
        if payload_shift_bins.size != config.K:
            raise ValueError("Unable to allocate the requested contiguous payload region.")
        payload_bin_indices = _fftshift_to_native_bins(payload_shift_bins, config.M)
        if config.N_pilots > 0:
            left_pilots = active_shift_bins[:pilot_edge]
            right_pilots = active_shift_bins[payload_stop:]
            pilot_shift_bins = np.concatenate([left_pilots, right_pilots]).astype(int)
        else:
            pilot_shift_bins = np.array([], dtype=int)
        pilot_bin_indices = _fftshift_to_native_bins(pilot_shift_bins, config.M)
        data_start = (config.K - config.N_data) // 2
        data_stream_indices = np.arange(data_start, data_start + config.N_data, dtype=int)
        data_shift_bins = payload_shift_bins[data_stream_indices]
        data_bin_indices = payload_bin_indices[data_stream_indices]
        pilot_stream_indices = np.array([], dtype=int)
        if pilot_shift_bins.size > 0:
            left_indices = np.arange(pilot_edge, dtype=int)
            right_indices = np.arange(payload_stop, active_shift_bins.size, dtype=int)
            pilot_stream_indices = np.concatenate([left_indices, right_indices]).astype(int)
    else:
        payload_shift_bins = active_shift_bins
        payload_bin_indices = active_bin_indices
        pilot_stream_indices = _evenly_spaced_stream_indices(config.N_active, config.N_pilots)
        data_stream_indices = np.setdiff1d(np.arange(config.N_active, dtype=int), pilot_stream_indices, assume_unique=True)
        data_shift_bins = active_shift_bins[data_stream_indices]
        pilot_shift_bins = active_shift_bins[pilot_stream_indices]
        data_bin_indices = active_bin_indices[data_stream_indices]
        pilot_bin_indices = active_bin_indices[pilot_stream_indices]
    return FrameResourceLayout(
        active_shift_bins=active_shift_bins,
        guard_shift_bins=guard_shift_bins,
        active_bin_indices=active_bin_indices,
        guard_bin_indices=guard_bin_indices,
        payload_shift_bins=payload_shift_bins,
        payload_bin_indices=payload_bin_indices,
        data_stream_indices=data_stream_indices,
        pilot_stream_indices=pilot_stream_indices,
        data_shift_bins=data_shift_bins,
        pilot_shift_bins=pilot_shift_bins,
        data_bin_indices=data_bin_indices,
        pilot_bin_indices=pilot_bin_indices,
    )


def pilot_reference_symbol(modulation: str, device: str = "cpu") -> torch.Tensor:
    modulation = modulation.upper()
    if modulation == "QPSK":
        return torch.tensor((1.0 + 1.0j) / math.sqrt(2.0), device=device, dtype=torch.complex64)
    if modulation == "16QAM":
        return torch.tensor((1.0 + 1.0j) / math.sqrt(10.0), device=device, dtype=torch.complex64)
    raise ValueError(f"Unsupported modulation: {modulation}")


def build_partial_dft_frame(
    M: int,
    active_bins: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    n = torch.arange(M, device=device, dtype=torch.float32)
    k = torch.tensor(active_bins, device=device, dtype=torch.float32)
    n_grid, k_grid = torch.meshgrid(n, k, indexing="ij")
    return torch.exp(1j * 2 * math.pi * n_grid * k_grid / M).to(dtype) / math.sqrt(M)


def make_tx_support_frame(config: ExperimentConfig) -> torch.Tensor | None:
    if not config.payload_region_enabled:
        return None
    layout = frame_resource_layout(config)
    return build_partial_dft_frame(config.M, layout.payload_bin_indices, device=config.device)


def make_initial_transceiver(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.payload_region_enabled:
        support_frame = make_tx_support_frame(config)
        assert support_frame is not None
        coeff_bins = active_fft_bins(config.K, config.N, config.learned_init_pattern)
        tx_init = build_partial_dft_frame(config.K, coeff_bins, device=config.device)
        tx_basis = normalize_columns(support_frame @ tx_init)
        rx_init = tx_basis.conj().T.contiguous()
        return tx_init, rx_init
    active_bins = active_fft_bins(config.M, config.N, config.learned_init_pattern)
    tx_init = build_partial_dft_frame(config.M, active_bins, device=config.device)
    rx_init = tx_init.conj().T.contiguous()
    return tx_init, rx_init


def make_ofdm_baseline_transceiver(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.payload_region_enabled:
        active_bins = frame_resource_layout(config).data_bin_indices
    elif config.frame_structure_enabled:
        active_bins = frame_resource_layout(config).active_bin_indices
    else:
        active_bins = active_fft_bins(config.M, config.N, "CONTIGUOUS")
    tx_basis = build_partial_dft_frame(config.M, active_bins, device=config.device)
    rx_basis = tx_basis.conj().T.contiguous()
    return tx_basis, rx_basis


def qpsk_from_bits(bits: torch.Tensor) -> torch.Tensor:
    re = 2 * bits[..., 0].to(torch.float32) - 1
    im = 2 * bits[..., 1].to(torch.float32) - 1
    return ((re + 1j * im) / math.sqrt(2.0)).to(torch.complex64)


def qam16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    idx_i = (2 * bits[..., 0] + bits[..., 1]).to(torch.int64)
    idx_q = (2 * bits[..., 2] + bits[..., 3]).to(torch.int64)
    levels = _QAM16_LEVELS.to(bits.device)
    re = levels[idx_i]
    im = levels[idx_q]
    return ((re + 1j * im) / math.sqrt(10.0)).to(torch.complex64)


def symbols_from_bits(bits: torch.Tensor, modulation: str) -> torch.Tensor:
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_from_bits(bits)
    if modulation == "16QAM":
        return qam16_from_bits(bits)
    raise ValueError(f"Unsupported modulation: {modulation}")


def random_symbols(
    batch_size: int,
    N: int,
    modulation: str,
    bits_per_symbol: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (batch_size, N, bits_per_symbol), device=device)
    return bits, symbols_from_bits(bits, modulation)


def prepare_tx_basis(tx_basis: torch.Tensor) -> torch.Tensor:
    return normalize_columns(tx_basis)


def transmit_symbols(symbols: torch.Tensor, tx_basis: torch.Tensor) -> torch.Tensor:
    prepared = prepare_tx_basis(tx_basis)
    return symbols @ prepared.T


def sample_training_symbols(batch_size: int, config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.payload_region_enabled:
        return random_symbols(
            batch_size=batch_size,
            N=config.N,
            modulation=config.modulation,
            bits_per_symbol=config.bits_per_symbol,
            device=config.device,
        )
    if config.frame_structure_enabled:
        bits, data_symbols = random_symbols(
            batch_size=batch_size,
            N=config.N_data,
            modulation=config.modulation,
            bits_per_symbol=config.bits_per_symbol,
            device=config.device,
        )
        layout = frame_resource_layout(config)
        symbols = torch.zeros(batch_size, config.N_active, device=config.device, dtype=torch.complex64)
        pilot_symbol = pilot_reference_symbol(config.modulation, device=config.device)
        symbols[:, torch.tensor(layout.data_stream_indices, device=config.device, dtype=torch.long)] = data_symbols
        symbols[:, torch.tensor(layout.pilot_stream_indices, device=config.device, dtype=torch.long)] = pilot_symbol
        return bits, symbols
    return random_symbols(
        batch_size=batch_size,
        N=config.N,
        modulation=config.modulation,
        bits_per_symbol=config.bits_per_symbol,
        device=config.device,
    )
