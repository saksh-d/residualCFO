from __future__ import annotations

import math

import torch

from comm_core import ExperimentConfig, ebn0_to_noise_variance


def apply_residual_cfo(x: torch.Tensor, eps_values: torch.Tensor | float) -> torch.Tensor:
    if not isinstance(eps_values, torch.Tensor):
        eps_values = torch.full((x.shape[0],), float(eps_values), device=x.device, dtype=torch.float32)
    eps_values = eps_values.to(device=x.device, dtype=torch.float32)
    n = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    phase = torch.exp(1j * 2 * math.pi * eps_values[:, None] * n[None, :] / x.shape[1]).to(torch.complex64)
    return x * phase


def add_awgn(
    x: torch.Tensor,
    ebn0_db: float | torch.Tensor,
    config: ExperimentConfig,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(ebn0_db, torch.Tensor):
        ebn0_tensor = ebn0_db.to(device=x.device, dtype=torch.float32)
        ebn0_linear = torch.pow(10.0, ebn0_tensor / 10.0)
        sigma = torch.sqrt((1.0 / (config.bits_per_symbol * ebn0_linear)).clamp_min(1e-12) / 2.0)
        sigma = sigma[:, None]
    else:
        noise_var = ebn0_to_noise_variance(ebn0_db, config.bits_per_symbol)
        sigma = math.sqrt(noise_var / 2.0)
    if noise is None:
        noise = sigma * (torch.randn_like(torch.real(x)) + 1j * torch.randn_like(torch.real(x)))
    return (x + noise).to(torch.complex64), noise.to(torch.complex64)


def propagate(
    x: torch.Tensor,
    eps_values: torch.Tensor | float,
    config: ExperimentConfig,
    ebn0_db: float | torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    y = apply_residual_cfo(x, eps_values)
    if ebn0_db is not None:
        y, noise = add_awgn(y, ebn0_db, config, noise=noise)
    return y, noise


def effective_operator(
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    eps_values: torch.Tensor | list[float] | float,
) -> torch.Tensor:
    if not isinstance(eps_values, torch.Tensor):
        eps_values = torch.tensor(eps_values, device=tx_basis.device, dtype=torch.float32)
    if eps_values.ndim == 0:
        eps_values = eps_values[None]
    eps_values = eps_values.to(device=tx_basis.device, dtype=torch.float32)
    n = torch.arange(tx_basis.shape[0], device=tx_basis.device, dtype=torch.float32)
    phase = torch.exp(1j * 2 * math.pi * eps_values[:, None] * n[None, :] / tx_basis.shape[0]).to(torch.complex64)
    return torch.einsum("mk,bm,nm->bkn", tx_basis, phase, rx_basis)
