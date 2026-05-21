from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from channel import effective_operator, propagate
from comm_core import (
    ExperimentConfig,
    ExperimentResult,
    TrainingResult,
    annotate_stage1_summary_df,
    normalize_columns,
)
from receiver import (
    EvaluationScheme,
    detect_scheme_symbols,
    decode_symbols,
    diagonal_magnitudes,
    evaluate_scheme_set,
    identity_loss,
    nearest_neighbor_leakage_ratio,
    offdiag_leakage_ratio,
    receiver_fro_loss,
)
from transmitter import frame_resource_layout, make_ofdm_baseline_transceiver, sample_training_symbols, transmit_symbols


METHOD_ORDER = (
    "OFDM",
    "OFDMNonlinear",
    "Learned",
    "LearnedOraclePreV",
    "LearnedOracleMMSE",
    "LearnedOracleMMSEScale1p00",
    "LearnedOracleMMSEScale0p90",
    "LearnedOracleMMSEScale0p75",
    "LearnedOracleMMSEScale0p50",
    "LearnedOracleMMSEScale0p25",
    "LearnedOracleMMSENoise0p00",
    "LearnedOracleMMSENoise0p05",
    "LearnedOracleMMSENoise0p10",
    "LearnedOracleMMSENoise0p20",
    "LearnedOracleMMSENoise0p40",
    "LearnedNonlinear",
    "LearnedNonlinearOracleEps",
    "LearnedSpectral",
    "LearnedPAPR",
)
METHOD_DISPLAY_NAMES = {
    "OFDM": "Classical OFDM",
    "OFDMNonlinear": "OFDM + Stage 2",
    "Learned": "Learned Basis",
    "LearnedOraclePreV": "Learned + Oracle Pre-V CFO",
    "LearnedOracleMMSE": "Learned + Oracle MMSE",
    "LearnedNonlinear": "Learned + Stage 2",
    "LearnedNonlinearOracleEps": "Learned + Stage 2 (True delta)",
    "LearnedSpectral": "Learned + Spectral Mask",
    "LearnedPAPR": "Learned + PAPR Penalty",
}
METHOD_COLORS = {
    "OFDM": "#3A5F8A",
    "OFDMNonlinear": "#79A7D3",
    "Learned": "#C05A2B",
    "LearnedOraclePreV": "#7E6AA2",
    "LearnedOracleMMSE": "#2D8A5F",
    "LearnedNonlinear": "#E39A5F",
    "LearnedNonlinearOracleEps": "#C74B50",
    "LearnedSpectral": "#2D8A5F",
    "LearnedPAPR": "#5A8F29",
}
ORACLE_MMSE_SCALE_PREFIX = "LearnedOracleMMSEScale"
ORACLE_MMSE_SCALE_COLORS = {
    "1p00": "#1B5E20",
    "0p90": "#2E7D32",
    "0p75": "#43A047",
    "0p50": "#66BB6A",
    "0p25": "#A5D6A7",
}
ORACLE_MMSE_NOISE_PREFIX = "LearnedOracleMMSENoise"
ORACLE_MMSE_NOISE_COLORS = {
    "0p00": "#1D4E89",
    "0p05": "#2563B8",
    "0p10": "#3B82F6",
    "0p20": "#60A5FA",
    "0p40": "#93C5FD",
}


def _progress_enabled(config: ExperimentConfig) -> bool:
    return bool(getattr(config, "terminal_progress_enabled", True))


def _log_progress(config: ExperimentConfig, message: str) -> None:
    if _progress_enabled(config):
        print(message, flush=True)


def _stage2_decision_loss_label(config: ExperimentConfig) -> str:
    return r"$L_{\mathrm{BCE}}$" if config.stage2_decision_loss == "BIT_BCE" else r"$L_{\mathrm{CE}}$"


def _oracle_mmse_scale_suffix(method: str) -> str | None:
    if not method.startswith(ORACLE_MMSE_SCALE_PREFIX):
        return None
    suffix = method[len(ORACLE_MMSE_SCALE_PREFIX):]
    return suffix or None


def _oracle_mmse_scale_value(method: str) -> float | None:
    suffix = _oracle_mmse_scale_suffix(method)
    if suffix is None:
        return None
    try:
        return float(suffix.replace("p", "."))
    except ValueError:
        return None


def _oracle_mmse_noise_suffix(method: str) -> str | None:
    if not method.startswith(ORACLE_MMSE_NOISE_PREFIX):
        return None
    suffix = method[len(ORACLE_MMSE_NOISE_PREFIX):]
    return suffix or None


def _oracle_mmse_noise_value(method: str) -> float | None:
    suffix = _oracle_mmse_noise_suffix(method)
    if suffix is None:
        return None
    try:
        return float(suffix.replace("p", "."))
    except ValueError:
        return None


def method_display_name(method: str) -> str:
    scale = _oracle_mmse_scale_value(method)
    if scale is not None:
        return f"Learned + Oracle MMSE ({scale:.2f}x delta)"
    noise_level = _oracle_mmse_noise_value(method)
    if noise_level is not None:
        return f"Learned + Oracle MMSE (noise sigma = {noise_level:.2f}|delta|)"
    return METHOD_DISPLAY_NAMES.get(method, method)


def method_color(method: str) -> str:
    suffix = _oracle_mmse_scale_suffix(method)
    if suffix is not None:
        return ORACLE_MMSE_SCALE_COLORS.get(suffix, "#2D8A5F")
    noise_suffix = _oracle_mmse_noise_suffix(method)
    if noise_suffix is not None:
        return ORACLE_MMSE_NOISE_COLORS.get(noise_suffix, "#2563B8")
    return METHOD_COLORS.get(method, "#444444")


def ordered_methods(
    methods: list[str] | tuple[str, ...] | set[str],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    available = list(methods)
    forced = []
    if preferred_order is not None:
        forced = [method for method in preferred_order if method in available]
    preferred = forced + [method for method in METHOD_ORDER if method in available and method not in forced]
    scale_methods = [
        method
        for method in available
        if method not in preferred and method not in METHOD_ORDER and _oracle_mmse_scale_value(method) is not None
    ]
    scale_methods.sort(
        key=lambda method: (-float(_oracle_mmse_scale_value(method) or 0.0), method),
    )
    noise_methods = [
        method
        for method in available
        if method not in preferred and method not in METHOD_ORDER and _oracle_mmse_noise_value(method) is not None
    ]
    noise_methods.sort(
        key=lambda method: (float(_oracle_mmse_noise_value(method) or 0.0), method),
    )
    extras = sorted(
        method
        for method in available
        if (
            method not in preferred
            and method not in METHOD_ORDER
            and _oracle_mmse_scale_value(method) is None
            and _oracle_mmse_noise_value(method) is None
        )
    )
    return preferred + scale_methods + noise_methods + extras


def linear_method_order(
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    return [name for name in ordered_methods(schemes.keys(), preferred_order) if schemes[name].nonlinear_receiver is None]


def transmitter_plot_method_order(
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    available = set(schemes.keys())
    if preferred_order is not None:
        return ordered_methods(available, preferred_order)
    order: list[str] = []
    for pair in (("OFDM", "OFDMNonlinear"), ("Learned", "LearnedNonlinear"), ("LearnedSpectral", "LearnedPAPR")):
        for method in pair:
            if method in available:
                order.append(method)
                break
    extras = [method for method in ordered_methods(available, preferred_order) if method not in order]
    return order + extras


def average_metrics_by_abs_cfo(ber_df: pd.DataFrame) -> pd.DataFrame:
    if ber_df.empty:
        return ber_df.copy()
    frame = ber_df.copy()
    frame["eps"] = np.abs(frame["eps"].to_numpy(dtype=float))
    numeric_cols = [
        col
        for col in (
            "ber",
            "ser",
            "evm",
            "pilot_symbol_mse",
            "cfo_est_mae",
            "cfo_est_bias",
            "mean_estimated_eps",
            "stage2_eps_hat_mae",
            "stage2_eps_hat_bias",
            "stage2_eps_hat_mean",
        )
        if col in frame.columns
    ]
    base_cols = {
        "modulation": "first",
        "N": "first",
        "M": "first",
        "K": "first",
        "R": "first",
        "N_data": "first",
        "N_pilots": "first",
        "N_guard": "first",
        "payload_region_fraction": "first",
        "payload_fraction": "first",
        "train_ebn0_db": "first",
        "eval_ebn0_db": "first",
    }
    grouped = frame.groupby(["method", "eps"], as_index=False).agg({**base_cols, **{col: "mean" for col in numeric_cols}})
    grouped["method_label"] = grouped["method"].map(method_display_name)
    return grouped.sort_values(["method", "eps"], kind="stable").reset_index(drop=True)


def redundancy_ratio_tick_label(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{rounded:.1f}"
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def integrated_log10_metric(x_values: np.ndarray, metric_values: np.ndarray, floor: float = 1e-8) -> float:
    x_values = np.asarray(x_values, dtype=float)
    metric_values = np.asarray(metric_values, dtype=float)
    return float(np.trapezoid(np.log10(np.maximum(metric_values, floor)), x_values))


def integrated_even_metric(
    x_values: np.ndarray,
    metric_values: np.ndarray,
    *,
    x_min: float = 0.0,
    x_max: float | None = None,
) -> float:
    frame = pd.DataFrame(
        {
            "abs_x": np.abs(np.asarray(x_values, dtype=float)),
            "metric": np.asarray(metric_values, dtype=float),
        }
    )
    grouped = frame.groupby("abs_x", sort=True, as_index=False)["metric"].mean()
    mask = grouped["abs_x"] >= float(x_min) - 1e-12
    if x_max is not None:
        mask &= grouped["abs_x"] <= float(x_max) + 1e-12
    grouped = grouped.loc[mask].sort_values("abs_x", kind="stable")
    if len(grouped) <= 1:
        return float(grouped["metric"].iloc[0]) if len(grouped) == 1 else 0.0
    return float(np.trapezoid(grouped["metric"].to_numpy(), grouped["abs_x"].to_numpy()))


def robustness_thresholds(modulation: str) -> tuple[float, ...]:
    modulation = modulation.upper()
    if modulation == "QPSK":
        return (1e-4, 1e-3)
    if modulation == "16QAM":
        return (1e-2, 1e-1)
    raise ValueError(f"Unsupported modulation: {modulation}")


def robustness_column_name(threshold: float) -> str:
    return f"robust_window_ber_le_{threshold:g}"


def robustness_window(eps_values: np.ndarray, ber_values: np.ndarray, threshold: float) -> float:
    eps_values = np.asarray(eps_values, dtype=float)
    ber_values = np.asarray(ber_values, dtype=float)
    radii = np.unique(np.round(np.abs(eps_values), decimals=12))
    best_radius = 0.0
    for radius in np.sort(radii):
        mask = np.abs(eps_values) <= radius + 1e-12
        if np.all(ber_values[mask] <= threshold):
            best_radius = float(radius)
        else:
            break
    return best_radius


def resolved_basis_plot_indices(config: ExperimentConfig, num_streams: int) -> list[int]:
    if config.basis_plot_indices is not None:
        indices = [idx for idx in config.basis_plot_indices if 0 <= idx < num_streams]
        if indices:
            return indices
    count = min(5, num_streams)
    if count == num_streams:
        return list(range(num_streams))
    return sorted(set(np.linspace(0, num_streams - 1, num=count, dtype=int).tolist()))


def random_basis_plot_indices(config: ExperimentConfig, num_streams: int) -> list[int]:
    rng = np.random.default_rng(config.random_basis_seed)
    count = min(config.random_basis_plot_count, num_streams)
    return sorted(rng.choice(num_streams, size=count, replace=False).tolist())


def basis_spectra_db(basis: torch.Tensor, fft_len: int) -> np.ndarray:
    spec = torch.fft.fftshift(torch.fft.fft(basis, n=fft_len, dim=0), dim=0)
    mag = torch.abs(spec).detach().cpu().numpy()
    mag = mag / max(float(mag.max()), 1e-12)
    return 20 * np.log10(np.maximum(mag, 1e-6))


def build_scheme_dict(
    config: ExperimentConfig,
    training_result: TrainingResult,
) -> dict[str, EvaluationScheme]:
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)
    return {
        "OFDM": EvaluationScheme(tx_basis=ofdm_tx, rx_basis=ofdm_rx, nonlinear_receiver=None),
        "Learned": EvaluationScheme(
            tx_basis=training_result.learned_tx,
            rx_basis=training_result.learned_rx,
            nonlinear_receiver=None,
        ),
    }


def build_operator_df(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps_tensor = torch.tensor(config.operator_eval_cfo, device=config.device, dtype=torch.float32)
    rows: list[dict[str, float | int | str]] = []
    diag_rows: list[dict[str, float | int | str]] = []
    for method in linear_method_order(schemes, preferred_order):
        scheme = schemes[method]
        tx_basis = scheme.tx_basis
        rx_basis = scheme.rx_basis
        operator = effective_operator(tx_basis, rx_basis, eps_tensor)
        diag_mag = diagonal_magnitudes(operator).detach().cpu().numpy()
        leakage = offdiag_leakage_ratio(operator).cpu().numpy()
        nn_leakage = nearest_neighbor_leakage_ratio(operator).cpu().numpy()
        clean_identity = identity_loss(
            effective_operator(tx_basis, rx_basis, torch.tensor([0.0], device=config.device))
        ).item()
        rx_norm_sq = receiver_fro_loss(rx_basis).item()
        for idx, eps in enumerate(config.operator_eval_cfo):
            rows.append(
                {
                    "method": method,
                    "method_label": method_display_name(method),
                    "modulation": config.modulation,
                    "N": config.N,
                    "M": config.M,
                    "K": config.K,
                    "R": config.redundancy_dimensions,
                    "train_ebn0_db": config.train_ebn0_db,
                    "eval_ebn0_db": config.eval_ebn0_db,
                    "eps": float(eps),
                    "offdiag_leakage": float(leakage[idx]),
                    "nearest_neighbor_leakage": float(nn_leakage[idx]),
                    "residual_far_leakage": float(max(leakage[idx] - nn_leakage[idx], 0.0)),
                    "diag_mean_magnitude": float(diag_mag[idx].mean()),
                    "diag_std_magnitude": float(diag_mag[idx].std()),
                    "diag_min_magnitude": float(diag_mag[idx].min()),
                    "diag_max_magnitude": float(diag_mag[idx].max()),
                    "clean_identity_loss": float(clean_identity),
                    "receiver_fro_norm_sq": float(rx_norm_sq),
                }
            )
            if method.startswith("Learned"):
                for stream_idx, magnitude in enumerate(diag_mag[idx]):
                    diag_rows.append(
                        {
                            "method": method,
                            "method_label": method_display_name(method),
                            "eps": float(eps),
                            "stream": int(stream_idx),
                            "diag_magnitude": float(magnitude),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def build_summary_df(
    config: ExperimentConfig,
    training_result: TrainingResult,
    operator_df: pd.DataFrame,
    ber_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    thresholds = robustness_thresholds(config.modulation)
    for method in ordered_methods(ber_df["method"].unique().tolist(), preferred_order):
        ber_group = average_metrics_by_abs_cfo(ber_df[ber_df["method"] == method]).sort_values("eps")
        op_group = operator_df[operator_df["method"] == method]

        def _nearest_metric(eps: float, column: str) -> float:
            idx = int(np.argmin(np.abs(ber_group["eps"].values - eps)))
            return float(ber_group.iloc[idx][column])

        clean_identity = float("nan")
        clean_offdiag = float("nan")
        clean_nearest_neighbor = float("nan")
        receiver_norm_sq = float("nan")
        integrated_offdiag = float("nan")
        integrated_nearest_neighbor = float("nan")
        small_mid_offdiag = float("nan")
        small_mid_nearest_neighbor = float("nan")
        if not op_group.empty:
            clean_row = op_group.iloc[int(np.argmin(np.abs(op_group["eps"].values)))]
            clean_identity = float(clean_row["clean_identity_loss"])
            clean_offdiag = float(clean_row["offdiag_leakage"])
            clean_nearest_neighbor = float(clean_row["nearest_neighbor_leakage"])
            receiver_norm_sq = float(clean_row["receiver_fro_norm_sq"])
            integrated_offdiag = integrated_even_metric(
                op_group["eps"].values,
                op_group["offdiag_leakage"].values,
            )
            integrated_nearest_neighbor = integrated_even_metric(
                op_group["eps"].values,
                op_group["nearest_neighbor_leakage"].values,
            )
            small_mid_offdiag = integrated_even_metric(
                op_group["eps"].values,
                op_group["offdiag_leakage"].values,
                x_min=0.02,
                x_max=0.12,
            )
            small_mid_nearest_neighbor = integrated_even_metric(
                op_group["eps"].values,
                op_group["nearest_neighbor_leakage"].values,
                x_min=0.02,
                x_max=0.12,
            )

        rows.append(
            {
                "method": method,
                "method_label": method_display_name(method),
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
                "eval_ebn0_db": config.eval_ebn0_db,
                "stage_failed": training_result.stage_failed if method.startswith("Learned") else False,
                "failed_stage": training_result.failed_stage if method.startswith("Learned") else "",
                "stop_reason": (
                    training_result.stop_reason
                    if method.endswith("Nonlinear") or method.startswith("Learned")
                    else "Fixed baseline."
                ),
                "clean_identity_loss": clean_identity,
                "clean_offdiag_leakage": clean_offdiag,
                "clean_nearest_neighbor_leakage": clean_nearest_neighbor,
                "receiver_fro_norm_sq": receiver_norm_sq,
                "integrated_log10_ber": integrated_log10_metric(ber_group["eps"].values, ber_group["ber"].values),
                "integrated_offdiag_leakage": integrated_offdiag,
                "integrated_nearest_neighbor_leakage": integrated_nearest_neighbor,
                "small_mid_integrated_offdiag_leakage": small_mid_offdiag,
                "small_mid_integrated_nearest_neighbor_leakage": small_mid_nearest_neighbor,
                "ber_at_0": _nearest_metric(0.0, "ber"),
                "ber_at_0p05": _nearest_metric(0.05, "ber"),
                "ber_at_0p10": _nearest_metric(0.10, "ber"),
                "evm_at_0": _nearest_metric(0.0, "evm"),
                "evm_at_0p05": _nearest_metric(0.05, "evm"),
                "evm_at_0p10": _nearest_metric(0.10, "evm"),
                "pilot_symbol_mse_at_0p10": _nearest_metric(0.10, "pilot_symbol_mse"),
                "cfo_est_mae_at_0p10": _nearest_metric(0.10, "cfo_est_mae"),
                "stage2_eps_hat_mae_at_0p10": _nearest_metric(0.10, "stage2_eps_hat_mae"),
                "stage2_eps_hat_bias_at_0p10": _nearest_metric(0.10, "stage2_eps_hat_bias"),
            }
        )
        for threshold in thresholds:
            rows[-1][robustness_column_name(threshold)] = robustness_window(
                ber_group["eps"].values,
                ber_group["ber"].values,
                threshold,
            )
    summary_df = pd.DataFrame(rows)
    if config.stage2_enabled:
        summary_df["stage1_checkpoint_source_path"] = (
            str(config.stage2_checkpoint_source_path or config.stage2_checkpoint_path or "")
        )
        summary_df["stage1_checkpoint_snapshot_path"] = str(config.stage2_checkpoint_snapshot_path or "")
        summary_df["stage1_checkpoint_hash_sha256"] = str(config.stage2_checkpoint_hash_sha256)
        summary_df["stage1_checkpoint_file_sha256"] = str(config.stage2_checkpoint_file_sha256)
        summary_df["stage1_checkpoint_format"] = str(config.stage2_checkpoint_format)
        summary_df["stage1_checkpoint_acceptance_passed"] = config.stage2_checkpoint_acceptance_passed
        summary_df["stage1_checkpoint_clean_identity_loss"] = config.stage2_checkpoint_clean_identity_loss
        summary_df["stage1_checkpoint_clean_offdiag_leakage"] = config.stage2_checkpoint_clean_offdiag_leakage
        summary_df["stage1_checkpoint_learned_ber_at_0"] = config.stage2_checkpoint_learned_ber_at_0
        summary_df["stage1_checkpoint_ofdm_ber_at_0"] = config.stage2_checkpoint_ofdm_ber_at_0
        return summary_df
    return annotate_stage1_summary_df(config, training_result, summary_df)


def build_ber_snr_df(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cfo_idx, eps in enumerate(config.snr_sweep_cfo_values):
        for snr_idx, ebn0_db in enumerate(np.asarray(config.snr_sweep_ebn0_db_grid, dtype=float)):
            frame = evaluate_scheme_set(
                config=config,
                schemes=schemes,
                cfo_points=np.array([eps], dtype=float),
                ebn0_db=float(ebn0_db),
                num_blocks=config.ber_blocks,
                batch_size=config.ber_batch_size,
                seed=54_321 + 1000 * cfo_idx + snr_idx,
            ).copy()
            frame["method_label"] = frame["method"].map(method_display_name)
            frame["snr_sweep_cfo"] = float(eps)
            frame["snr_sweep_ebn0_db"] = float(ebn0_db)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_snr_summary_df(
    ber_snr_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if ber_snr_df.empty:
        return pd.DataFrame(rows)
    for eps, cfo_group in ber_snr_df.groupby("eps"):
        for method in ordered_methods(cfo_group["method"].unique().tolist(), preferred_order):
            method_group = cfo_group[cfo_group["method"] == method].sort_values("eval_ebn0_db")
            rows.append(
                {
                    "method": method,
                    "method_label": method_display_name(method),
                    "eps": float(eps),
                    "integrated_log10_ber_vs_snr": integrated_log10_metric(
                        method_group["eval_ebn0_db"].values,
                        method_group["ber"].values,
                    ),
                    "ber_at_lowest_snr": float(method_group.iloc[0]["ber"]),
                    "ber_at_highest_snr": float(method_group.iloc[-1]["ber"]),
                }
            )
    return pd.DataFrame(rows)


def build_constellation_df(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    _, symbols = sample_training_symbols(config.constellation_num_blocks, config)
    noise_var = 1.0 / (config.bits_per_symbol * (10 ** (float(config.eval_ebn0_db) / 10.0)))
    sigma = np.sqrt(noise_var / 2.0)
    noise = sigma * (
        torch.randn(config.constellation_num_blocks, config.M, device=config.device)
        + 1j * torch.randn(config.constellation_num_blocks, config.M, device=config.device)
    ).to(torch.complex64)

    flat_ref = symbols.reshape(-1).detach().cpu().numpy()
    point_count = min(config.constellation_plot_points, flat_ref.size)
    rng = np.random.default_rng(config.base_seed + 7_777)
    sample_indices = np.sort(rng.choice(flat_ref.size, size=point_count, replace=False))

    rows: list[pd.DataFrame] = []
    for eps in config.constellation_cfo:
        eps_tensor = torch.full(
            (config.constellation_num_blocks,),
            float(eps),
            device=config.device,
            dtype=torch.float32,
        )
        for method in ordered_methods(schemes.keys(), preferred_order):
            scheme = schemes[method]
            tx_signal = transmit_symbols(symbols, scheme.tx_basis)
            rx_signal, _ = propagate(tx_signal, eps_tensor, config, ebn0_db=config.eval_ebn0_db, noise=noise)
            outputs = detect_scheme_symbols(
                config,
                scheme,
                rx_signal,
                eps_tensor=eps_tensor,
                ebn0_db=config.eval_ebn0_db,
            )
            shat = outputs["symbol_estimates"].reshape(-1).detach().cpu().numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "method_label": method_display_name(method),
                        "eps": float(eps),
                        "ref_real": np.real(flat_ref[sample_indices]),
                        "ref_imag": np.imag(flat_ref[sample_indices]),
                        "est_real": np.real(shat[sample_indices]),
                        "est_imag": np.imag(shat[sample_indices]),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def occupied_bandwidth(psd: np.ndarray, freq_axis: np.ndarray, fraction: float) -> float:
    power = np.asarray(psd, dtype=float)
    power = power / max(power.sum(), 1e-12)
    order = np.argsort(np.abs(freq_axis))
    cumulative = np.cumsum(power[order])
    cutoff_rank = int(np.searchsorted(cumulative, fraction, side="left"))
    cutoff_rank = min(cutoff_rank, len(order) - 1)
    edge_freq = abs(freq_axis[order[cutoff_rank]])
    bin_width = abs(freq_axis[1] - freq_axis[0]) if len(freq_axis) > 1 else 0.0
    return float(2.0 * (edge_freq + 0.5 * bin_width))


def build_spectral_outputs(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    freq_axis = np.linspace(-0.5, 0.5, config.spectral_fft_len, endpoint=False)
    method_names = ordered_methods(schemes.keys(), preferred_order)
    psd_accum = {method: np.zeros(config.spectral_fft_len, dtype=np.float64) for method in method_names}
    papr_frames: list[pd.DataFrame] = []

    remaining = max(config.spectral_eval_blocks, config.papr_eval_blocks)
    processed = 0
    while processed < remaining:
        batch_size = min(config.ber_batch_size, remaining - processed)
        _, symbols = sample_training_symbols(batch_size, config)
        for method in method_names:
            scheme = schemes[method]
            tx_signal = transmit_symbols(symbols, scheme.tx_basis)

            if processed < config.spectral_eval_blocks:
                spectral_take = min(batch_size, config.spectral_eval_blocks - processed)
                spectral_batch = tx_signal[:spectral_take]
                spectrum = torch.fft.fftshift(
                    torch.fft.fft(spectral_batch, n=config.spectral_fft_len, dim=1),
                    dim=1,
                )
                psd_accum[method] += torch.sum(torch.abs(spectrum) ** 2, dim=0).detach().cpu().numpy()

            if processed < config.papr_eval_blocks:
                papr_take = min(batch_size, config.papr_eval_blocks - processed)
                papr_batch = tx_signal[:papr_take]
                power = torch.abs(papr_batch) ** 2
                papr_db = (
                    10.0
                    * torch.log10(power.max(dim=1).values / power.mean(dim=1).clamp_min(1e-12))
                ).detach().cpu().numpy()
                papr_frames.append(
                    pd.DataFrame(
                        {
                            "method": method,
                            "method_label": method_display_name(method),
                            "papr_db": papr_db,
                        }
                    )
                )
        processed += batch_size

    psd_norm = {}
    peak_reference = 1e-12
    for method, power in psd_accum.items():
        psd_norm[method] = power / max(power.sum(), 1e-12)
        peak_reference = max(peak_reference, float(psd_norm[method].max()))

    spectral_rows: list[pd.DataFrame] = []
    spectral_summary_rows: list[dict[str, float | str]] = []
    nominal_bandwidth = config.payload_bin_count / config.M if config.frame_structure_enabled else config.N / config.M
    for method in method_names:
        method_psd = psd_norm[method]
        spectral_rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "method_label": method_display_name(method),
                    "frequency": freq_axis,
                    "psd": method_psd,
                    "psd_db": 10.0 * np.log10(np.maximum(method_psd / peak_reference, 1e-12)),
                }
            )
        )
        occupied_bw = occupied_bandwidth(method_psd, freq_axis, config.occupied_bandwidth_fraction)
        inband_mask = np.abs(freq_axis) <= nominal_bandwidth / 2.0
        oob_ratio = float(1.0 - method_psd[inband_mask].sum())
        spectral_summary_rows.append(
            {
                "method": method,
                "method_label": method_display_name(method),
                "occupied_bandwidth_fraction": config.occupied_bandwidth_fraction,
                "occupied_bandwidth": occupied_bw,
                "nominal_ofdm_bandwidth": nominal_bandwidth,
                "out_of_band_power_ratio": oob_ratio,
            }
        )

    papr_summary_rows: list[dict[str, float | str]] = []
    papr_df = pd.concat(papr_frames, ignore_index=True)
    for method in method_names:
        method_papr = papr_df[papr_df["method"] == method]["papr_db"].to_numpy()
        papr_summary_rows.append(
            {
                "method": method,
                "method_label": method_display_name(method),
                "papr_mean_db": float(method_papr.mean()),
                "papr_p95_db": float(np.quantile(method_papr, 0.95)),
                "papr_max_db": float(method_papr.max()),
            }
        )

    return (
        pd.concat(spectral_rows, ignore_index=True),
        pd.DataFrame(spectral_summary_rows),
        papr_df,
        pd.DataFrame(papr_summary_rows),
    )


def save_artifacts(
    config: ExperimentConfig,
    training_result: TrainingResult,
    operator_df: pd.DataFrame,
    diagonal_df: pd.DataFrame,
    ber_df: pd.DataFrame,
    ber_snr_df: pd.DataFrame,
    constellation_df: pd.DataFrame,
    spectral_df: pd.DataFrame,
    spectral_summary_df: pd.DataFrame,
    papr_df: pd.DataFrame,
    papr_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    snr_summary_df: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "history_csv": config.output_dir / "stage_training_history.csv",
        "stage_summary_csv": config.output_dir / "stage_summary.csv",
        "operator_csv": config.output_dir / "operator_diagnostics.csv",
        "diagonal_csv": config.output_dir / "diagonal_magnitudes.csv",
        "ber_csv": config.output_dir / "ber_vs_cfo.csv",
        "ber_snr_csv": config.output_dir / "ber_vs_snr_by_cfo.csv",
        "constellation_csv": config.output_dir / "constellation_snapshots.csv",
        "spectral_csv": config.output_dir / "spectral_psd.csv",
        "spectral_summary_csv": config.output_dir / "spectral_summary.csv",
        "papr_csv": config.output_dir / "papr_samples.csv",
        "papr_summary_csv": config.output_dir / "papr_summary.csv",
        "summary_csv": config.output_dir / "summary_metrics.csv",
        "snr_summary_csv": config.output_dir / "ber_vs_snr_slice_summary.csv",
    }
    training_result.history_df.to_csv(paths["history_csv"], index=False)
    training_result.stage_summary_df.to_csv(paths["stage_summary_csv"], index=False)
    operator_df.to_csv(paths["operator_csv"], index=False)
    diagonal_df.to_csv(paths["diagonal_csv"], index=False)
    ber_df.to_csv(paths["ber_csv"], index=False)
    ber_snr_df.to_csv(paths["ber_snr_csv"], index=False)
    constellation_df.to_csv(paths["constellation_csv"], index=False)
    spectral_df.to_csv(paths["spectral_csv"], index=False)
    spectral_summary_df.to_csv(paths["spectral_summary_csv"], index=False)
    papr_df.to_csv(paths["papr_csv"], index=False)
    papr_summary_df.to_csv(paths["papr_summary_csv"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    snr_summary_df.to_csv(paths["snr_summary_csv"], index=False)
    return paths


def build_frame_resource_df(config: ExperimentConfig) -> pd.DataFrame:
    if not config.frame_structure_enabled:
        return pd.DataFrame()

    layout = frame_resource_layout(config)
    rows = [
        {
            "modulation": config.modulation,
            "M": config.M,
            "N": config.N,
            "K": config.K,
            "R": config.redundancy_dimensions,
            "N_active": config.N_active,
            "N_data": config.N_data,
            "N_pilots": config.N_pilots,
            "N_guard": config.N_guard,
            "active_fraction": config.active_resource_bins / config.M,
            "payload_region_fraction": config.payload_region_fraction,
            "payload_fraction": config.payload_fraction,
            "pilot_fraction": config.N_pilots / config.M,
            "guard_fraction": config.N_guard / config.M,
            "data_stream_indices": ",".join(str(int(v)) for v in layout.data_stream_indices),
            "pilot_stream_indices": ",".join(str(int(v)) for v in layout.pilot_stream_indices),
            "active_shift_bins": ",".join(str(int(v)) for v in layout.active_shift_bins),
            "payload_shift_bins": ",".join(str(int(v)) for v in layout.payload_shift_bins),
            "data_shift_bins": ",".join(str(int(v)) for v in layout.data_shift_bins),
            "pilot_shift_bins": ",".join(str(int(v)) for v in layout.pilot_shift_bins),
            "guard_shift_bins": ",".join(str(int(v)) for v in layout.guard_shift_bins),
            "active_bin_indices": ",".join(str(int(v)) for v in layout.active_bin_indices),
            "payload_bin_indices": ",".join(str(int(v)) for v in layout.payload_bin_indices),
            "data_bin_indices": ",".join(str(int(v)) for v in layout.data_bin_indices),
            "pilot_bin_indices": ",".join(str(int(v)) for v in layout.pilot_bin_indices),
            "guard_bin_indices": ",".join(str(int(v)) for v in layout.guard_bin_indices),
        }
    ]
    return pd.DataFrame(rows)


def build_frame_pilot_summary_df(ber_df: pd.DataFrame) -> pd.DataFrame:
    if ber_df.empty or "pilot_symbol_mse" not in ber_df.columns or not ber_df["pilot_symbol_mse"].notna().any():
        return pd.DataFrame()
    columns = [
        "method",
        "method_label",
        "modulation",
        "N",
        "M",
        "N_data",
        "N_pilots",
        "N_guard",
        "eval_ebn0_db",
        "eps",
        "mean_estimated_eps",
        "cfo_est_mae",
        "cfo_est_bias",
        "pilot_symbol_mse",
    ]
    available_columns = [col for col in columns if col in ber_df.columns]
    return ber_df[available_columns].copy().sort_values(["method", "eps"], kind="stable").reset_index(drop=True)


def save_frame_structured_artifacts(
    output_dir: Path,
    resource_df: pd.DataFrame,
    pilot_summary_df: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not resource_df.empty:
        paths["resource_csv"] = output_dir / "resource_usage_summary.csv"
        resource_df.to_csv(paths["resource_csv"], index=False)
    if not pilot_summary_df.empty:
        paths["pilot_summary_csv"] = output_dir / "pilot_estimation_summary.csv"
        pilot_summary_df.to_csv(paths["pilot_summary_csv"], index=False)
    return paths


def plot_frame_resource_map(config: ExperimentConfig) -> Path | None:
    if not config.frame_structure_enabled:
        return None
    layout = frame_resource_layout(config)
    categories = np.zeros(config.M, dtype=int)
    categories[layout.guard_shift_bins] = 0
    if config.payload_region_enabled:
        categories[layout.payload_shift_bins] = 1
        categories[layout.data_shift_bins] = 2
        categories[layout.pilot_shift_bins] = 3
        cmap = plt.matplotlib.colors.ListedColormap(["#D5DBE4", "#E8C7A4", "#C05A2B", "#3A5F8A"])
        legend_handles = [
            plt.matplotlib.patches.Patch(color="#C05A2B", label="OFDM data tones"),
            plt.matplotlib.patches.Patch(color="#E8C7A4", label="Learned-only shaping bins"),
            plt.matplotlib.patches.Patch(color="#3A5F8A", label="Reserved pilot tones"),
            plt.matplotlib.patches.Patch(color="#D5DBE4", label="Guard"),
        ]
        title = (
            f"Frame resource map | {config.modulation}, M={config.M}, K={config.K}, "
            f"N={config.N}, pilots={config.N_pilots}, guards={config.N_guard}"
        )
        vmax = 3
    else:
        categories[layout.data_shift_bins] = 1
        categories[layout.pilot_shift_bins] = 2
        cmap = plt.matplotlib.colors.ListedColormap(["#D5DBE4", "#C05A2B", "#3A5F8A"])
        legend_handles = [
            plt.matplotlib.patches.Patch(color="#C05A2B", label="Data"),
            plt.matplotlib.patches.Patch(color="#3A5F8A", label="Pilot"),
            plt.matplotlib.patches.Patch(color="#D5DBE4", label="Guard"),
        ]
        title = (
            f"Frame resource map | {config.modulation}, M={config.M}, data={config.N_data}, "
            f"pilots={config.N_pilots}, guards={config.N_guard}"
        )
        vmax = 2
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.4), dpi=130, constrained_layout=True)
    ax.imshow(categories[None, :], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks(np.arange(config.M))
    shift_labels = [str(int(idx - config.M // 2)) for idx in range(config.M)]
    ax.set_xticklabels(shift_labels, fontsize=7)
    ax.set_xlabel("Centered FFT bin index")
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    path = config.output_dir / "frame_resource_map.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pilot_estimation_summary(config: ExperimentConfig, pilot_summary_df: pd.DataFrame) -> Path | None:
    if pilot_summary_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=130, constrained_layout=True)
    for method in ordered_methods(pilot_summary_df["method"].unique().tolist()):
        group = pilot_summary_df[pilot_summary_df["method"] == method].sort_values("eps")
        if group.empty:
            continue
        axes[0].plot(
            group["eps"],
            group["cfo_est_mae"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
        axes[1].plot(
            group["eps"],
            group["pilot_symbol_mse"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    axes[0].set_title("Residual CFO estimation error")
    axes[0].set_xlabel("True residual CFO")
    axes[0].set_ylabel(r"$|\hat{\delta} - \delta|$")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Pilot reconstruction MSE")
    axes[1].set_xlabel("True residual CFO")
    axes[1].set_ylabel("Pilot MSE")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        f"Pilot-aided estimation summary | {config.modulation}, M={config.M}, active={config.N_active}",
        fontsize=12,
    )
    path = config.output_dir / "pilot_estimation_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown_report(
    config: ExperimentConfig,
    result: ExperimentResult,
    output_dir: Path,
    plot_files: list[str] | tuple[str, ...],
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    extra_metadata = {} if extra_metadata is None else dict(extra_metadata)
    output_dir = Path(output_dir)

    def relpath(path: Path) -> str:
        try:
            return str(path.relative_to(output_dir))
        except ValueError:
            return os.path.relpath(path, output_dir)

    def coerce_lines(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
        return [str(value)]

    title = str(extra_metadata.get("title", f"Residual-CFO experiment report: {config.output_dir.name}"))
    overview_lines = coerce_lines(extra_metadata.get("overview_lines"))
    control_lines = coerce_lines(extra_metadata.get("control_lines"))

    csv_entries: list[tuple[str, Path]] = []
    for name, path in result.artifact_paths.items():
        path = Path(path)
        if path.suffix.lower() == ".csv":
            csv_entries.append((name, path))
    csv_entries.sort(key=lambda item: item[0])

    checkpoint_entries: list[tuple[str, Path]] = []
    for name, path in result.artifact_paths.items():
        path = Path(path)
        if path.suffix.lower() in {".pt", ".pth"}:
            checkpoint_entries.append((name, path))
    checkpoint_entries.sort(key=lambda item: item[0])

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Modulation: `{config.modulation}`")
    if overview_lines:
        lines.append("")
        lines.extend(overview_lines)
    if control_lines:
        lines.append("")
        lines.append("## Configuration")
        lines.append("")
        lines.extend(control_lines)

    lines.append("")
    lines.append("## Mapping Summary")
    lines.append("")
    lines.append(result.mapping_markdown)
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(result.summary_markdown)

    if csv_entries:
        lines.append("")
        lines.append("## Data Artifacts")
        lines.append("")
        for name, path in csv_entries:
            lines.append(f"- `{name}`: [{path.name}]({relpath(path)})")

    if checkpoint_entries:
        lines.append("")
        lines.append("## Model Artifacts")
        lines.append("")
        for name, path in checkpoint_entries:
            lines.append(f"- `{name}`: [{path.name}]({relpath(path)})")

    figure_paths = [output_dir / name for name in plot_files if (output_dir / name).exists()]
    if figure_paths:
        lines.append("")
        lines.append("## Figures")
        lines.append("")
        for path in figure_paths:
            lines.append(f"### {path.stem.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"[{path.name}]({relpath(path)})")
            lines.append("")
            lines.append(f"![{path.name}]({relpath(path)})")
            lines.append("")

    report_body = "\n".join(lines).rstrip() + "\n"
    report_path = output_dir / "report.md"
    report_path.write_text(report_body)
    if config.stage2_enabled:
        uppercase_report_path = output_dir / "report.MD"
        uppercase_report_path.write_text(report_body)
    return report_path


def build_redundancy_ablation_summary(ablation_df: pd.DataFrame) -> pd.DataFrame:
    if ablation_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str | bool]] = []
    group_cols = ["modulation", "N", "M", "M_over_N", "train_ebn0_db", "eval_ebn0_db", "eps"]
    for _, group in ablation_df.groupby(group_cols, sort=True):
        learned = group[group["method"] == "Learned"].iloc[0]
        ofdm = group[group["method"] == "OFDM"].iloc[0]
        rows.append(
            {
                "modulation": learned["modulation"],
                "N": int(learned["N"]),
                "M": int(learned["M"]),
                "M_over_N": float(learned["M_over_N"]),
                "train_ebn0_db": float(learned["train_ebn0_db"]),
                "eval_ebn0_db": float(learned["eval_ebn0_db"]),
                "eps": float(learned["eps"]),
                "learned_ber": float(learned["ber"]),
                "ofdm_ber": float(ofdm["ber"]),
                "learned_ser": float(learned["ser"]),
                "ofdm_ser": float(ofdm["ser"]),
                "learned_evm": float(learned["evm"]),
                "ofdm_evm": float(ofdm["evm"]),
                "learned_stage_failed": bool(learned["stage_failed"]),
                "learned_failed_stage": str(learned["failed_stage"]),
                "learned_stop_reason": str(learned["stop_reason"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["M_over_N", "M"], kind="stable").reset_index(drop=True)


def save_redundancy_ablation_artifacts(
    output_dir: Path,
    ablation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ablation_csv": output_dir / "redundancy_ablation.csv",
        "ablation_summary_csv": output_dir / "redundancy_ablation_summary.csv",
    }
    ablation_df.to_csv(paths["ablation_csv"], index=False)
    summary_df.to_csv(paths["ablation_summary_csv"], index=False)
    return paths


def _summary_row_for_method(
    result: ExperimentResult,
    source_method: str,
    output_method: str,
    lambda_spec: float | None = None,
    lambda_papr: float | None = None,
) -> dict[str, float | int | str | bool]:
    summary_row = result.summary_df[result.summary_df["method"] == source_method].iloc[0]
    spectral_row = result.spectral_summary_df[result.spectral_summary_df["method"] == source_method].iloc[0]
    papr_row = result.papr_summary_df[result.papr_summary_df["method"] == source_method].iloc[0]
    row: dict[str, float | int | str | bool] = {
        "method": output_method,
        "method_label": method_display_name(output_method),
        "modulation": summary_row["modulation"],
        "N": int(summary_row["N"]),
        "M": int(summary_row["M"]),
        "train_ebn0_db": float(summary_row["train_ebn0_db"]),
        "eval_ebn0_db": float(summary_row["eval_ebn0_db"]),
        "stage_failed": bool(summary_row["stage_failed"]),
        "failed_stage": str(summary_row["failed_stage"]),
        "stop_reason": str(summary_row["stop_reason"]),
        "integrated_log10_ber": float(summary_row["integrated_log10_ber"]),
        "ber_at_0": float(summary_row["ber_at_0"]),
        "ber_at_0p05": float(summary_row["ber_at_0p05"]),
        "ber_at_0p10": float(summary_row["ber_at_0p10"]),
        "evm_at_0p10": float(summary_row["evm_at_0p10"]),
        "clean_identity_loss": float(summary_row["clean_identity_loss"]),
        "clean_offdiag_leakage": float(summary_row["clean_offdiag_leakage"]),
        "stage1_acceptance_passed": bool(summary_row.get("stage1_acceptance_passed", not summary_row["stage_failed"])),
        "occupied_bandwidth": float(spectral_row["occupied_bandwidth"]),
        "out_of_band_power_ratio": float(spectral_row["out_of_band_power_ratio"]),
        "papr_mean_db": float(papr_row["papr_mean_db"]),
        "papr_p95_db": float(papr_row["papr_p95_db"]),
        "papr_max_db": float(papr_row["papr_max_db"]),
        "lambda_spec": float(lambda_spec) if lambda_spec is not None else np.nan,
        "lambda_papr": float(lambda_papr) if lambda_papr is not None else np.nan,
    }
    for col in summary_row.index:
        if isinstance(col, str) and col.startswith("robust_window_"):
            row[col] = float(summary_row[col])
    return row


def _relabel_method_frame(df: pd.DataFrame, source_method: str, output_method: str) -> pd.DataFrame:
    frame = df[df["method"] == source_method].copy()
    frame["method"] = output_method
    frame["method_label"] = method_display_name(output_method)
    return frame


def build_spectral_constraint_trial_summary(
    modulation: str,
    constrained_results: list[tuple[float, ExperimentResult]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for lambda_spec, result in constrained_results:
        learned = _summary_row_for_method(
            result=result,
            source_method="Learned",
            output_method="LearnedSpectral",
            lambda_spec=lambda_spec,
        )
        learned["modulation"] = modulation
        rows.append(learned)
    return pd.DataFrame(rows).sort_values("lambda_spec", kind="stable").reset_index(drop=True)


def build_spectral_constraint_comparison_summary(
    baseline_result: ExperimentResult,
    best_constrained_result: ExperimentResult | None,
    best_lambda_spec: float | None,
) -> pd.DataFrame:
    rows = [
        _summary_row_for_method(baseline_result, "OFDM", "OFDM"),
        _summary_row_for_method(baseline_result, "Learned", "Learned"),
    ]
    if best_constrained_result is not None and best_lambda_spec is not None:
        rows.append(
            _summary_row_for_method(
                best_constrained_result,
                "Learned",
                "LearnedSpectral",
                lambda_spec=best_lambda_spec,
            )
        )
    return pd.DataFrame(rows)


def build_papr_constraint_trial_summary(
    modulation: str,
    constrained_results: list[tuple[float, ExperimentResult]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for lambda_papr, result in constrained_results:
        learned = _summary_row_for_method(
            result=result,
            source_method="Learned",
            output_method="LearnedPAPR",
            lambda_papr=lambda_papr,
        )
        learned["modulation"] = modulation
        rows.append(learned)
    return pd.DataFrame(rows).sort_values("lambda_papr", kind="stable").reset_index(drop=True)


def build_papr_constraint_comparison_summary(
    baseline_result: ExperimentResult,
    best_constrained_result: ExperimentResult | None,
    best_lambda_papr: float | None,
) -> pd.DataFrame:
    rows = [
        _summary_row_for_method(baseline_result, "OFDM", "OFDM"),
        _summary_row_for_method(baseline_result, "Learned", "Learned"),
    ]
    if best_constrained_result is not None and best_lambda_papr is not None:
        rows.append(
            _summary_row_for_method(
                best_constrained_result,
                "Learned",
                "LearnedPAPR",
                lambda_papr=best_lambda_papr,
            )
        )
    return pd.DataFrame(rows)


def build_papr_constraint_tradeoff_df(
    trial_summary_df: pd.DataFrame,
    comparison_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if trial_summary_df.empty:
        return trial_summary_df.copy()

    learned_baseline = comparison_summary_df[comparison_summary_df["method"] == "Learned"].iloc[0]
    ofdm_baseline = comparison_summary_df[comparison_summary_df["method"] == "OFDM"].iloc[0]
    tradeoff_df = trial_summary_df.copy()
    tradeoff_df["delta_integrated_log10_ber_vs_learned"] = (
        tradeoff_df["integrated_log10_ber"] - float(learned_baseline["integrated_log10_ber"])
    )
    tradeoff_df["delta_papr_p95_db_vs_learned"] = (
        tradeoff_df["papr_p95_db"] - float(learned_baseline["papr_p95_db"])
    )
    tradeoff_df["delta_papr_mean_db_vs_learned"] = (
        tradeoff_df["papr_mean_db"] - float(learned_baseline["papr_mean_db"])
    )
    tradeoff_df["delta_ber_at_0_vs_learned"] = tradeoff_df["ber_at_0"] - float(learned_baseline["ber_at_0"])
    tradeoff_df["delta_ber_at_0p05_vs_learned"] = (
        tradeoff_df["ber_at_0p05"] - float(learned_baseline["ber_at_0p05"])
    )
    tradeoff_df["delta_ber_at_0p10_vs_learned"] = (
        tradeoff_df["ber_at_0p10"] - float(learned_baseline["ber_at_0p10"])
    )
    tradeoff_df["delta_oob_power_vs_learned"] = (
        tradeoff_df["out_of_band_power_ratio"] - float(learned_baseline["out_of_band_power_ratio"])
    )
    tradeoff_df["delta_occupied_bw_vs_learned"] = (
        tradeoff_df["occupied_bandwidth"] - float(learned_baseline["occupied_bandwidth"])
    )
    tradeoff_df["delta_clean_identity_vs_learned"] = (
        tradeoff_df["clean_identity_loss"] - float(learned_baseline["clean_identity_loss"])
    )
    tradeoff_df["delta_clean_offdiag_vs_learned"] = (
        tradeoff_df["clean_offdiag_leakage"] - float(learned_baseline["clean_offdiag_leakage"])
    )
    tradeoff_df["delta_papr_p95_db_vs_ofdm"] = (
        tradeoff_df["papr_p95_db"] - float(ofdm_baseline["papr_p95_db"])
    )
    return tradeoff_df.sort_values("lambda_papr", kind="stable").reset_index(drop=True)


def plot_papr_constraint_tradeoff(
    tradeoff_df: pd.DataFrame,
    comparison_summary_df: pd.DataFrame,
    path: Path,
) -> Path | None:
    if tradeoff_df.empty:
        return None

    learned_baseline = comparison_summary_df[comparison_summary_df["method"] == "Learned"].iloc[0]
    ofdm_baseline = comparison_summary_df[comparison_summary_df["method"] == "OFDM"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=130, constrained_layout=True)
    axes = axes.ravel()

    accepted = tradeoff_df["stage1_acceptance_passed"].to_numpy(dtype=bool)
    accepted_color = "#2D8A5F"
    failed_color = "#C74B50"
    point_colors = np.where(accepted, accepted_color, failed_color)

    axes[0].plot(
        tradeoff_df["lambda_papr"],
        tradeoff_df["integrated_log10_ber"],
        color="#444444",
        linewidth=1.2,
        alpha=0.8,
    )
    axes[0].scatter(
        tradeoff_df["lambda_papr"],
        tradeoff_df["integrated_log10_ber"],
        c=point_colors,
        s=42,
        zorder=3,
    )
    axes[0].axhline(float(learned_baseline["integrated_log10_ber"]), color=method_color("Learned"), linestyle="--", linewidth=1.2)
    axes[0].axhline(float(ofdm_baseline["integrated_log10_ber"]), color=method_color("OFDM"), linestyle=":", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_title("BER robustness vs lambda")
    axes[0].set_xlabel(r"$\lambda_{\mathrm{papr}}$")
    axes[0].set_ylabel("Integrated log10 BER")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(
        tradeoff_df["lambda_papr"],
        tradeoff_df["papr_p95_db"],
        color="#444444",
        linewidth=1.2,
        alpha=0.8,
    )
    axes[1].scatter(
        tradeoff_df["lambda_papr"],
        tradeoff_df["papr_p95_db"],
        c=point_colors,
        s=42,
        zorder=3,
    )
    axes[1].axhline(float(learned_baseline["papr_p95_db"]), color=method_color("Learned"), linestyle="--", linewidth=1.2)
    axes[1].axhline(float(ofdm_baseline["papr_p95_db"]), color=method_color("OFDM"), linestyle=":", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_title("PAPR p95 vs lambda")
    axes[1].set_xlabel(r"$\lambda_{\mathrm{papr}}$")
    axes[1].set_ylabel("PAPR p95 [dB]")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].scatter(
        tradeoff_df["papr_p95_db"],
        tradeoff_df["integrated_log10_ber"],
        c=point_colors,
        s=52,
        zorder=3,
    )
    axes[2].scatter(
        [float(learned_baseline["papr_p95_db"])],
        [float(learned_baseline["integrated_log10_ber"])],
        color=method_color("Learned"),
        marker="s",
        s=64,
        label="Learned baseline",
        zorder=4,
    )
    axes[2].scatter(
        [float(ofdm_baseline["papr_p95_db"])],
        [float(ofdm_baseline["integrated_log10_ber"])],
        color=method_color("OFDM"),
        marker="^",
        s=64,
        label="OFDM",
        zorder=4,
    )
    for _, row in tradeoff_df.iterrows():
        axes[2].annotate(
            f"{float(row['lambda_papr']):.0e}",
            (float(row["papr_p95_db"]), float(row["integrated_log10_ber"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[2].set_title("PAPR-BER tradeoff")
    axes[2].set_xlabel("PAPR p95 [dB]")
    axes[2].set_ylabel("Integrated log10 BER")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(
        tradeoff_df["lambda_papr"],
        tradeoff_df["out_of_band_power_ratio"],
        color=method_color("LearnedPAPR"),
        linewidth=1.8,
        marker="o",
    )
    axes[3].axhline(float(learned_baseline["out_of_band_power_ratio"]), color=method_color("Learned"), linestyle="--", linewidth=1.2)
    axes[3].axhline(float(ofdm_baseline["out_of_band_power_ratio"]), color=method_color("OFDM"), linestyle=":", linewidth=1.2)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_title("Spectral spillover vs lambda")
    axes[3].set_xlabel(r"$\lambda_{\mathrm{papr}}$")
    axes[3].set_ylabel("Out-of-band power ratio")
    axes[3].grid(True, which="both", alpha=0.3)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ber_curve_multi(
    ber_df: pd.DataFrame,
    method_order: tuple[str, ...],
    title: str,
    path: Path,
) -> Path | None:
    if ber_df.empty:
        return None
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in method_order:
        group = ber_df[ber_df["method"] == method].sort_values("eps")
        if group.empty:
            continue
        plt.semilogy(
            group["eps"],
            group["ber"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(title)
    plt.xlabel("Normalized residual CFO")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectral_fairness_multi(
    n: int,
    m: int,
    spectral_df: pd.DataFrame,
    papr_df: pd.DataFrame,
    method_order: tuple[str, ...],
    title: str,
    path: Path,
) -> Path | None:
    if spectral_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=130, constrained_layout=True)

    nominal_edge = n / (2.0 * m)
    for method in method_order:
        group = spectral_df[spectral_df["method"] == method].sort_values("frequency")
        if group.empty:
            continue
        axes[0].plot(
            group["frequency"],
            group["psd_db"],
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    axes[0].axvline(nominal_edge, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axvline(-nominal_edge, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_title("Average transmit PSD")
    axes[0].set_xlabel("Normalized frequency")
    axes[0].set_ylabel("PSD [dBr]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for method in method_order:
        values = np.sort(papr_df[papr_df["method"] == method]["papr_db"].to_numpy())
        if values.size == 0:
            continue
        ccdf = 1.0 - np.arange(1, values.size + 1, dtype=float) / values.size
        ccdf = np.maximum(ccdf, 1.0 / values.size)
        axes[1].semilogy(
            values,
            ccdf,
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    axes[1].set_xlabel("PAPR [dB]")
    axes[1].set_ylabel("CCDF")
    axes[1].set_title("PAPR CCDF")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_spectral_constraint_artifacts(
    output_dir: Path,
    baseline_result: ExperimentResult,
    constrained_results: list[tuple[float, ExperimentResult]],
    constrained_trial_df: pd.DataFrame,
    comparison_summary_df: pd.DataFrame,
    best_constrained_result: ExperimentResult | None,
    best_lambda_spec: float | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trial_summary_csv": output_dir / "spectral_constraint_trial_summary.csv",
        "comparison_summary_csv": output_dir / "spectral_constraint_comparison_summary.csv",
    }
    constrained_trial_df.to_csv(paths["trial_summary_csv"], index=False)
    comparison_summary_df.to_csv(paths["comparison_summary_csv"], index=False)

    all_trial_rows = []
    for lambda_spec, result in constrained_results:
        frame = result.summary_df.copy()
        frame["lambda_spec"] = lambda_spec
        all_trial_rows.append(frame)
    if all_trial_rows:
        paths["trial_raw_csv"] = output_dir / "spectral_constraint_trial_raw_summary.csv"
        pd.concat(all_trial_rows, ignore_index=True).to_csv(paths["trial_raw_csv"], index=False)

    if best_constrained_result is None or best_lambda_spec is None:
        return paths

    comparison_ber_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.ber_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.ber_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.ber_df, "Learned", "LearnedSpectral"),
        ],
        ignore_index=True,
    )
    comparison_spectral_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.spectral_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.spectral_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.spectral_df, "Learned", "LearnedSpectral"),
        ],
        ignore_index=True,
    )
    comparison_papr_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.papr_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.papr_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.papr_df, "Learned", "LearnedSpectral"),
        ],
        ignore_index=True,
    )

    paths["comparison_ber_csv"] = output_dir / "spectral_constraint_comparison_ber_vs_cfo.csv"
    comparison_ber_df.to_csv(paths["comparison_ber_csv"], index=False)
    paths["comparison_spectral_csv"] = output_dir / "spectral_constraint_comparison_spectral_psd.csv"
    comparison_spectral_df.to_csv(paths["comparison_spectral_csv"], index=False)
    paths["comparison_papr_csv"] = output_dir / "spectral_constraint_comparison_papr_samples.csv"
    comparison_papr_df.to_csv(paths["comparison_papr_csv"], index=False)

    ber_plot = plot_ber_curve_multi(
        ber_df=comparison_ber_df,
        method_order=("OFDM", "Learned", "LearnedSpectral"),
        title=(
            f"BER vs residual CFO | {baseline_result.summary_df.iloc[0]['modulation']}, "
            f"N={baseline_result.summary_df.iloc[0]['N']}, M={baseline_result.summary_df.iloc[0]['M']}"
        ),
        path=output_dir / "spectral_constraint_comparison_ber_vs_cfo.png",
    )
    if ber_plot is not None:
        paths["comparison_ber_plot"] = ber_plot

    spectral_plot = plot_spectral_fairness_multi(
        n=int(baseline_result.summary_df.iloc[0]["N"]),
        m=int(baseline_result.summary_df.iloc[0]["M"]),
        spectral_df=comparison_spectral_df,
        papr_df=comparison_papr_df,
        method_order=("OFDM", "Learned", "LearnedSpectral"),
        title=(
            f"Spectral fairness comparison | {baseline_result.summary_df.iloc[0]['modulation']}, "
            f"best lambda={best_lambda_spec:.2f}"
        ),
        path=output_dir / "spectral_constraint_comparison_spectral_fairness.png",
    )
    if spectral_plot is not None:
        paths["comparison_spectral_plot"] = spectral_plot
    return paths


def save_papr_constraint_artifacts(
    output_dir: Path,
    baseline_result: ExperimentResult,
    constrained_results: list[tuple[float, ExperimentResult]],
    constrained_trial_df: pd.DataFrame,
    comparison_summary_df: pd.DataFrame,
    best_constrained_result: ExperimentResult | None,
    best_lambda_papr: float | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trial_summary_csv": output_dir / "papr_constraint_trial_summary.csv",
        "comparison_summary_csv": output_dir / "papr_constraint_comparison_summary.csv",
    }
    constrained_trial_df.to_csv(paths["trial_summary_csv"], index=False)
    comparison_summary_df.to_csv(paths["comparison_summary_csv"], index=False)
    tradeoff_df = build_papr_constraint_tradeoff_df(constrained_trial_df, comparison_summary_df)
    paths["tradeoff_csv"] = output_dir / "papr_constraint_tradeoff_summary.csv"
    tradeoff_df.to_csv(paths["tradeoff_csv"], index=False)
    tradeoff_plot = plot_papr_constraint_tradeoff(
        tradeoff_df=tradeoff_df,
        comparison_summary_df=comparison_summary_df,
        path=output_dir / "papr_constraint_tradeoff.png",
    )
    if tradeoff_plot is not None:
        paths["tradeoff_plot"] = tradeoff_plot

    all_trial_rows = []
    for lambda_papr, result in constrained_results:
        frame = result.summary_df.copy()
        frame["lambda_papr"] = lambda_papr
        all_trial_rows.append(frame)
    if all_trial_rows:
        paths["trial_raw_csv"] = output_dir / "papr_constraint_trial_raw_summary.csv"
        pd.concat(all_trial_rows, ignore_index=True).to_csv(paths["trial_raw_csv"], index=False)

    if best_constrained_result is None or best_lambda_papr is None:
        return paths

    comparison_ber_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.ber_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.ber_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.ber_df, "Learned", "LearnedPAPR"),
        ],
        ignore_index=True,
    )
    comparison_spectral_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.spectral_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.spectral_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.spectral_df, "Learned", "LearnedPAPR"),
        ],
        ignore_index=True,
    )
    comparison_papr_df = pd.concat(
        [
            _relabel_method_frame(baseline_result.papr_df, "OFDM", "OFDM"),
            _relabel_method_frame(baseline_result.papr_df, "Learned", "Learned"),
            _relabel_method_frame(best_constrained_result.papr_df, "Learned", "LearnedPAPR"),
        ],
        ignore_index=True,
    )

    paths["comparison_ber_csv"] = output_dir / "papr_constraint_comparison_ber_vs_cfo.csv"
    comparison_ber_df.to_csv(paths["comparison_ber_csv"], index=False)
    paths["comparison_spectral_csv"] = output_dir / "papr_constraint_comparison_spectral_psd.csv"
    comparison_spectral_df.to_csv(paths["comparison_spectral_csv"], index=False)
    paths["comparison_papr_csv"] = output_dir / "papr_constraint_comparison_papr_samples.csv"
    comparison_papr_df.to_csv(paths["comparison_papr_csv"], index=False)

    ber_plot = plot_ber_curve_multi(
        ber_df=comparison_ber_df,
        method_order=("OFDM", "Learned", "LearnedPAPR"),
        title=(
            f"BER vs residual CFO | {baseline_result.summary_df.iloc[0]['modulation']}, "
            f"N={baseline_result.summary_df.iloc[0]['N']}, M={baseline_result.summary_df.iloc[0]['M']}"
        ),
        path=output_dir / "papr_constraint_comparison_ber_vs_cfo.png",
    )
    if ber_plot is not None:
        paths["comparison_ber_plot"] = ber_plot

    spectral_plot = plot_spectral_fairness_multi(
        n=int(baseline_result.summary_df.iloc[0]["N"]),
        m=int(baseline_result.summary_df.iloc[0]["M"]),
        spectral_df=comparison_spectral_df,
        papr_df=comparison_papr_df,
        method_order=("OFDM", "Learned", "LearnedPAPR"),
        title=(
            f"PAPR-regularization comparison | {baseline_result.summary_df.iloc[0]['modulation']}, "
            f"best lambda={best_lambda_papr:.1e}"
        ),
        path=output_dir / "papr_constraint_comparison_spectral_fairness.png",
    )
    if spectral_plot is not None:
        paths["comparison_spectral_plot"] = spectral_plot
    return paths


def plot_training_diagnostics(config: ExperimentConfig, history_df: pd.DataFrame) -> Path:
    path = config.output_dir / "training_diagnostics.png"
    if "clean_offdiag_leakage" not in history_df.columns:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=130, constrained_layout=True, sharex=True)
        for scheme, group in history_df.groupby("scheme"):
            axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=scheme)
            axes[1].plot(group["global_epoch"], group["val_total"], linewidth=2, label=scheme)
            axes[2].plot(group["global_epoch"], group["val_ber"], linewidth=2, label=scheme)
        axes[0].set_title(
            f"{config.modulation} Stage 2 training diagnostics | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
        )
        axes[0].set_ylabel("Train total")
        axes[0].legend()
        axes[1].set_title("Validation total loss")
        axes[1].set_ylabel("Val total")
        axes[2].set_title("Validation BER")
        axes[2].set_xlabel("Global epoch")
        axes[2].set_ylabel("Val BER")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    if config.stage2_enabled and "val_ber" in history_df.columns and "train_Ldecision" in history_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=130, constrained_layout=True)
        axes = axes.ravel()

        for stage, group in history_df.groupby("stage"):
            axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=stage)
            axes[1].plot(group["global_epoch"], group["val_total"], linewidth=2, label=stage)
            axes[2].plot(group["global_epoch"], group["val_ber"], linewidth=2, label=stage)
            axes[3].plot(group["global_epoch"], group["clean_offdiag_leakage"], linewidth=2, label=stage)
        axes[0].set_title(
            f"{config.modulation} Stage 2 diagnostics | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
        )
        axes[0].set_xlabel("Global epoch")
        axes[0].set_ylabel("Train total")
        axes[0].legend()
        axes[1].set_title("Validation total loss")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("Val total")
        axes[1].legend()
        axes[2].set_title("Validation BER")
        axes[2].set_xlabel("Global epoch")
        axes[2].set_ylabel("Val BER")
        axes[2].legend()
        axes[3].set_title("Clean-operator off-diagonal leakage")
        axes[3].set_xlabel("Global epoch")
        axes[3].set_ylabel("Leakage ratio")
        axes[3].legend()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=130, constrained_layout=True)
    axes = axes.ravel()

    for stage, group in history_df.groupby("stage"):
        axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=stage)
    axes[0].set_title(
        f"{config.modulation} training diagnostics | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
    )
    axes[0].set_xlabel("Global epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    component_keys = ["train_L0", "train_Loff", "train_Ldiag", "train_LV", "train_Lsym"]
    if "train_Lnn" in history_df.columns and np.any(np.abs(history_df["train_Lnn"].to_numpy()) > 1e-12):
        component_keys.append("train_Lnn")
    if "train_Lspec" in history_df.columns and np.any(np.abs(history_df["train_Lspec"].to_numpy()) > 1e-12):
        component_keys.append("train_Lspec")
    if "train_Lpapr" in history_df.columns and np.any(np.abs(history_df["train_Lpapr"].to_numpy()) > 1e-12):
        component_keys.append("train_Lpapr")
    for key in component_keys:
        axes[1].plot(history_df["global_epoch"], history_df[key], linewidth=2, label=key.replace("train_", ""))
    axes[1].set_title("Loss components")
    axes[1].set_xlabel("Global epoch")
    axes[1].set_ylabel("Component value")
    axes[1].legend()

    axes[2].plot(history_df["global_epoch"], history_df["clean_offdiag_leakage"], linewidth=2)
    axes[2].set_title("Clean-operator off-diagonal leakage")
    axes[2].set_xlabel("Global epoch")
    axes[2].set_ylabel("Leakage ratio")

    axes[3].plot(history_df["global_epoch"], history_df["rx_fro_norm_sq"], linewidth=2)
    axes[3].set_title("Receiver Frobenius norm")
    axes[3].set_xlabel("Global epoch")
    axes[3].set_ylabel(r"$||V||_F^2$")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_loss_components(config: ExperimentConfig, history_df: pd.DataFrame) -> Path:
    path = config.output_dir / "training_loss_components.png"
    if "stage" not in history_df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=130, constrained_layout=True, sharex=True)
        for scheme, group in history_df.groupby("scheme"):
            axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=scheme)
        axes[0].set_title(
            f"{config.modulation} Stage 2 loss curves | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
        )
        axes[0].set_ylabel("Total loss")
        axes[0].legend()

        component_specs = [
            ("train_Lce", r"$L_{\mathrm{CE}}$"),
            ("train_Lmse", r"$L_{\mathrm{MSE}}$"),
            ("residual_scale", r"$\rho$"),
        ]
        for key, label in component_specs:
            if key not in history_df.columns:
                continue
            axes[1].plot(history_df["global_epoch"], history_df[key], linewidth=2, label=label)
        axes[1].set_title("Receiver loss components")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("Value")
        axes[1].legend()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    if config.stage2_enabled and "train_Ldecision" in history_df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=130, constrained_layout=True, sharex=True)

        for stage, group in history_df.groupby("stage"):
            axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=stage)
        axes[0].set_title(
            f"{config.modulation} Stage 2 loss curves | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
        )
        axes[0].set_ylabel("Total loss")
        axes[0].legend()

        component_specs = [
            ("train_Ldecision", _stage2_decision_loss_label(config)),
            ("train_Lbce", r"$L_{\mathrm{BCE}}$"),
            ("train_Lce", r"$L_{\mathrm{CE}}$"),
            ("train_Lmse", r"$L_{\mathrm{MSE}}$"),
            ("train_L0", r"$L_0$"),
            ("train_Loff", r"$L_{\mathrm{off}}$"),
            ("train_Ldiag", r"$L_{\mathrm{diag}}$"),
            ("train_Lnn", r"$L_{\mathrm{nn}}$"),
            ("train_Lspec", r"$L_{\mathrm{spec}}$"),
            ("residual_scale", r"$\rho$"),
        ]
        for key, label in component_specs:
            if key not in history_df.columns:
                continue
            axes[1].plot(history_df["global_epoch"], history_df[key], linewidth=2, label=label)
        axes[1].set_title("Stage 2 loss components")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("Value")
        axes[1].legend(ncol=2)

        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=130, constrained_layout=True, sharex=True)

    for stage, group in history_df.groupby("stage"):
        axes[0].plot(group["global_epoch"], group["train_total"], linewidth=2, label=stage)
    axes[0].set_title(
        f"{config.modulation} supplementary training curves | N={config.N}, M={config.M} | train {config.train_ebn0_db:.0f} dB"
    )
    axes[0].set_ylabel("Total loss")
    axes[0].legend()

    component_specs = [
        ("train_L0", r"$L_0$"),
        ("train_Loff", r"$L_{\mathrm{off}}$"),
        ("train_Ldiag", r"$L_{\mathrm{diag}}$"),
        ("train_Lnn", r"$L_{\mathrm{nn}}$"),
        ("train_Lsym", r"$L_{\mathrm{sym}}$"),
        ("train_Lspec", r"$L_{\mathrm{spec}}$"),
        ("train_Lpapr", r"$L_{\mathrm{papr}}$"),
        ("rx_fro_norm_sq", r"$||V||_F^2$"),
    ]
    for key, label in component_specs:
        if key not in history_df.columns:
            continue
        axes[1].plot(history_df["global_epoch"], history_df[key], linewidth=2, label=label)
    axes[1].set_title("Loss components and receiver norm")
    axes[1].set_xlabel("Global epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend(ncol=2)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_operator_heatmaps(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    linear_methods = linear_method_order(schemes, preferred_order)
    if not linear_methods:
        return None
    eps_tensor = torch.tensor(config.heatmap_cfo, device=config.device, dtype=torch.float32)
    operators = {
        method: effective_operator(schemes[method].tx_basis, schemes[method].rx_basis, eps_tensor).detach().cpu().numpy()
        for method in linear_methods
    }

    fig, axes = plt.subplots(
        len(linear_methods),
        len(config.heatmap_cfo),
        figsize=(4.5 * len(config.heatmap_cfo), 4.0 * len(linear_methods)),
        dpi=130,
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, method in enumerate(linear_methods):
        for col_idx, eps in enumerate(config.heatmap_cfo):
            ax = axes[row_idx][col_idx]
            image = np.abs(operators[method][col_idx])
            im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"{method_display_name(method)} | " + rf"$|A(\delta)|$, $\delta={eps:.2f}$")
            ax.set_xlabel("Output stream")
            ax.set_ylabel("Input stream")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = config.output_dir / "operator_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_offdiag_leakage(
    config: ExperimentConfig,
    operator_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path:
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in ordered_methods(operator_df["method"].unique().tolist(), preferred_order):
        group = operator_df[operator_df["method"] == method].sort_values("eps")
        plt.plot(
            group["eps"],
            group["offdiag_leakage"],
            linewidth=2.1,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(
        f"Off-diagonal leakage vs residual CFO | {config.modulation}, N={config.N}, M={config.M}"
    )
    plt.xlabel("Normalized residual CFO")
    plt.ylabel(r"$\eta_{off}(\delta)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = config.output_dir / "offdiag_leakage_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nearest_neighbor_leakage(
    config: ExperimentConfig,
    operator_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if operator_df.empty or "nearest_neighbor_leakage" not in operator_df.columns:
        return None
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in ordered_methods(operator_df["method"].unique().tolist(), preferred_order):
        group = operator_df[operator_df["method"] == method].sort_values("eps")
        plt.plot(
            group["eps"],
            group["nearest_neighbor_leakage"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(
        f"Nearest-neighbor leakage vs CFO | {config.modulation}, N={config.N}, M={config.M}"
    )
    plt.xlabel("Normalized residual CFO")
    plt.ylabel(r"$\eta_{nn}(\delta)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = config.output_dir / "nearest_neighbor_leakage_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ber_curve(
    config: ExperimentConfig,
    ber_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if ber_df.empty:
        return None
    plot_df = average_metrics_by_abs_cfo(ber_df) if config.stage2_enabled else ber_df
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in ordered_methods(plot_df["method"].unique().tolist(), preferred_order):
        group = plot_df[plot_df["method"] == method].sort_values("eps")
        plt.semilogy(
            group["eps"],
            group["ber"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(
        f"BER vs residual CFO | {config.modulation}, N={config.N}, M={config.M}, eval {config.eval_ebn0_db:.0f} dB"
    )
    plt.xlabel("Absolute residual CFO" if config.stage2_enabled else "Normalized residual CFO")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    path = config.output_dir / "ber_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if config.stage2_enabled:
        signed_fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
        for method in ordered_methods(ber_df["method"].unique().tolist(), preferred_order):
            group = ber_df[ber_df["method"] == method].sort_values("eps")
            plt.semilogy(
                group["eps"],
                group["ber"],
                marker="o",
                linewidth=2.0,
                color=method_color(method),
                label=method_display_name(method),
            )
        plt.title(
            f"BER vs signed residual CFO | {config.modulation}, N={config.N}, M={config.M}, eval {config.eval_ebn0_db:.0f} dB"
        )
        plt.xlabel("Normalized residual CFO")
        plt.ylabel("BER")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        signed_fig.savefig(config.output_dir / "ber_vs_cfo_signed.png", bbox_inches="tight")
        plt.close(signed_fig)
    return path


def plot_evm_curve(
    config: ExperimentConfig,
    ber_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if ber_df.empty:
        return None
    plot_df = average_metrics_by_abs_cfo(ber_df) if config.stage2_enabled else ber_df
    fig = plt.figure(figsize=(7.8, 4.4), dpi=130)
    for method in ordered_methods(plot_df["method"].unique().tolist(), preferred_order):
        group = plot_df[plot_df["method"] == method].sort_values("eps")
        plt.plot(
            group["eps"],
            group["evm"],
            marker="o",
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(
        f"EVM vs residual CFO | {config.modulation}, N={config.N}, M={config.M}, eval {config.eval_ebn0_db:.0f} dB"
    )
    plt.xlabel("Absolute residual CFO" if config.stage2_enabled else "Normalized residual CFO")
    plt.ylabel("EVM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = config.output_dir / "evm_vs_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ber_vs_snr_by_cfo(
    config: ExperimentConfig,
    ber_snr_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if ber_snr_df.empty:
        return None
    cfo_values = [float(eps) for eps in sorted(ber_snr_df["eps"].unique())]
    fig, axes = plt.subplots(
        1,
        len(cfo_values),
        figsize=(5.0 * len(cfo_values), 4.2),
        dpi=130,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, eps in zip(axes[0], cfo_values):
        cfo_group = ber_snr_df[ber_snr_df["eps"] == eps]
        for method in ordered_methods(cfo_group["method"].unique().tolist(), preferred_order):
            method_group = cfo_group[cfo_group["method"] == method].sort_values("eval_ebn0_db")
            ax.semilogy(
                method_group["eval_ebn0_db"],
                method_group["ber"],
                marker="o",
                linewidth=2.0,
                color=method_color(method),
                label=method_display_name(method),
            )
        ax.set_title(rf"$\delta = {eps:.2f}$")
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend()
    fig.suptitle(
        f"BER vs SNR at fixed residual CFO | {config.modulation}, N={config.N}, M={config.M}",
        fontsize=12,
    )
    path = config.output_dir / "ber_vs_snr_by_cfo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_redundancy_ablation_ber(
    config: ExperimentConfig,
    ablation_df: pd.DataFrame,
    fixed_n: int,
    fixed_eps: float,
    fixed_eval_ebn0_db: float,
) -> Path | None:
    if ablation_df.empty:
        return None

    fig = plt.figure(figsize=(8.2, 4.6), dpi=130)
    line_styles = {"OFDM": "--", "Learned": "-"}
    x_values = sorted(float(value) for value in ablation_df["M_over_N"].unique())
    for method in ordered_methods(ablation_df["method"].unique().tolist()):
        group = ablation_df[ablation_df["method"] == method].sort_values(["M_over_N", "M"])
        plt.semilogy(
            group["M_over_N"],
            group["ber"],
            marker="o",
            linestyle=line_styles.get(method, "-"),
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(
        f"BER vs redundancy ratio | {config.modulation}, N={fixed_n}, "
        + rf"$\delta={fixed_eps:.2f}$, $E_b/N_0={fixed_eval_ebn0_db:.0f}$ dB"
    )
    plt.xlabel("Redundancy ratio M/N")
    plt.ylabel("BER")
    plt.xticks(x_values, [redundancy_ratio_tick_label(value) for value in x_values])
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    path = config.output_dir / "redundancy_ablation_ber.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_constellation_snapshots(
    config: ExperimentConfig,
    constellation_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if constellation_df.empty:
        return None
    cfo_values = [float(eps) for eps in sorted(constellation_df["eps"].unique())]
    method_names = ordered_methods(constellation_df["method"].unique().tolist(), preferred_order)
    fig, axes = plt.subplots(
        len(method_names),
        len(cfo_values),
        figsize=(4.4 * len(cfo_values), 4.0 * len(method_names)),
        dpi=130,
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ideal_points = np.unique(
        constellation_df["ref_real"].round(6).astype(str) + "," + constellation_df["ref_imag"].round(6).astype(str)
    )
    ideal_coords = np.array([[float(v) for v in point.split(",")] for point in ideal_points], dtype=float)
    for row_idx, method in enumerate(method_names):
        for col_idx, eps in enumerate(cfo_values):
            ax = axes[row_idx][col_idx]
            points = constellation_df[
                (constellation_df["method"] == method) & (constellation_df["eps"] == eps)
            ]
            ax.scatter(points["est_real"], points["est_imag"], s=7, alpha=0.26, color=method_color(method))
            ax.scatter(
                ideal_coords[:, 0],
                ideal_coords[:, 1],
                s=40,
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.8,
            )
            ax.set_title(f"{method_display_name(method)} | " + rf"$\delta={eps:.2f}$")
            ax.set_xlabel("In-phase")
            ax.set_ylabel("Quadrature")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.22)
    path = config.output_dir / "constellation_snapshots.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frequency_domain_all(config: ExperimentConfig, learned_tx: torch.Tensor) -> Path:
    learned_plot = normalize_columns(learned_tx).detach().cpu()
    freq_axis = np.linspace(-0.5, 0.5, config.fft_len, endpoint=False)
    spectra_db = basis_spectra_db(learned_plot, config.fft_len)

    fig = plt.figure(figsize=(9.2, 4.8), dpi=130)
    for idx in range(spectra_db.shape[1]):
        plt.plot(freq_axis, spectra_db[:, idx], linewidth=0.9, alpha=0.65)
    plt.title(
        f"All learned frequency-domain basis spectra | {config.modulation}, N={config.N}, M={config.M}"
    )
    plt.xlabel("Normalized frequency")
    plt.ylabel("Magnitude [dB]")
    plt.ylim(-70, 3)
    plt.grid(True, alpha=0.25)
    path = config.output_dir / "frequency_domain_bases_all.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frequency_domain_random(config: ExperimentConfig, learned_tx: torch.Tensor) -> Path:
    learned_plot = normalize_columns(learned_tx).detach().cpu()
    freq_axis = np.linspace(-0.5, 0.5, config.fft_len, endpoint=False)
    spectra_db = basis_spectra_db(learned_plot, config.fft_len)
    indices = random_basis_plot_indices(config, learned_plot.shape[1])

    fig = plt.figure(figsize=(9.2, 4.8), dpi=130)
    for idx in indices:
        plt.plot(freq_axis, spectra_db[:, idx], linewidth=1.7, label=f"k={idx}")
    plt.title(
        f"Random learned frequency-domain basis spectra | {config.modulation}, seed {config.random_basis_seed}"
    )
    plt.xlabel("Normalized frequency")
    plt.ylabel("Magnitude [dB]")
    plt.ylim(-70, 3)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    path = config.output_dir / "frequency_domain_bases_random.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_domain_waveform(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path:
    _, symbols = sample_training_symbols(1, config)
    time = np.arange(config.M)
    method_names = transmitter_plot_method_order(schemes, preferred_order)

    fig, axes = plt.subplots(len(method_names), 1, figsize=(9.0, 3.2 * len(method_names)), dpi=130, constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(len(method_names), 1)
    for row_idx, method in enumerate(method_names):
        waveform = transmit_symbols(symbols, schemes[method].tx_basis)[0].detach().cpu().numpy()
        axes[row_idx, 0].plot(time, np.real(waveform), linewidth=1.8, label="Real")
        axes[row_idx, 0].plot(time, np.imag(waveform), linewidth=1.8, linestyle="--", label="Imag")
        axes[row_idx, 0].set_title(f"{method_display_name(method)} waveform samples")
        axes[row_idx, 0].set_xlabel("Time sample")
        axes[row_idx, 0].set_ylabel("Amplitude")
        axes[row_idx, 0].grid(True, alpha=0.25)
        if row_idx == 0:
            axes[row_idx, 0].legend()

    path = config.output_dir / "time_domain_waveform.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_domain_envelope_phase(
    config: ExperimentConfig,
    schemes: dict[str, EvaluationScheme],
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path:
    _, symbols = sample_training_symbols(1, config)
    time = np.arange(config.M)
    method_names = transmitter_plot_method_order(schemes, preferred_order)

    fig, axes = plt.subplots(len(method_names), 1, figsize=(9.0, 3.2 * len(method_names)), dpi=130, constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(len(method_names), 1)
    for row_idx, method in enumerate(method_names):
        waveform = transmit_symbols(symbols, schemes[method].tx_basis)[0].detach().cpu().numpy()
        axes[row_idx, 0].plot(time, np.abs(waveform), linewidth=1.8, color=method_color(method), label="Magnitude")
        axes[row_idx, 0].plot(time, np.unwrap(np.angle(waveform)), linewidth=1.2, linestyle=":", color="#444444", label="Phase")
        axes[row_idx, 0].set_title(f"{method_display_name(method)} waveform envelope / phase")
        axes[row_idx, 0].set_xlabel("Time sample")
        axes[row_idx, 0].set_ylabel("Value")
        axes[row_idx, 0].grid(True, alpha=0.25)
        if row_idx == 0:
            axes[row_idx, 0].legend()

    path = config.output_dir / "time_domain_envelope_phase.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_papr_ccdf(
    config: ExperimentConfig,
    papr_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if papr_df.empty:
        return None
    fig = plt.figure(figsize=(7.6, 4.4), dpi=130)
    for method in ordered_methods(papr_df["method"].unique().tolist(), preferred_order):
        values = np.sort(papr_df[papr_df["method"] == method]["papr_db"].to_numpy())
        if values.size == 0:
            continue
        ccdf = 1.0 - np.arange(1, values.size + 1, dtype=float) / values.size
        ccdf = np.maximum(ccdf, 1.0 / values.size)
        plt.semilogy(
            values,
            ccdf,
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    plt.title(f"PAPR CCDF | {config.modulation}, N={config.N}, M={config.M}")
    plt.xlabel("PAPR [dB]")
    plt.ylabel("CCDF")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    path = config.output_dir / "papr_ccdf.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectral_fairness(
    config: ExperimentConfig,
    spectral_df: pd.DataFrame,
    papr_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    if spectral_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=130, constrained_layout=True)

    nominal_width = config.payload_bin_count if config.frame_structure_enabled else config.N
    nominal_edge = nominal_width / (2.0 * config.M)
    for method in ordered_methods(spectral_df["method"].unique().tolist(), preferred_order):
        group = spectral_df[spectral_df["method"] == method].sort_values("frequency")
        axes[0].plot(
            group["frequency"],
            group["psd_db"],
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    axes[0].axvline(nominal_edge, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axvline(-nominal_edge, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_title("Average transmit PSD")
    axes[0].set_xlabel("Normalized frequency")
    axes[0].set_ylabel("PSD [dBr]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for method in ordered_methods(papr_df["method"].unique().tolist(), preferred_order):
        values = np.sort(papr_df[papr_df["method"] == method]["papr_db"].to_numpy())
        if values.size == 0:
            continue
        ccdf = 1.0 - np.arange(1, values.size + 1, dtype=float) / values.size
        ccdf = np.maximum(ccdf, 1.0 / values.size)
        axes[1].semilogy(
            values,
            ccdf,
            linewidth=2.0,
            color=method_color(method),
            label=method_display_name(method),
        )
    axes[1].set_xlabel("PAPR [dB]")
    axes[1].set_ylabel("CCDF")
    axes[1].set_title("PAPR CCDF")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Spectral and implementation fairness | {config.modulation}, N={config.N}, M={config.M}",
        fontsize=12,
    )
    path = config.output_dir / "spectral_fairness.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_mapping_markdown(config: ExperimentConfig) -> str:
    if config.stage2_enabled:
        if getattr(config, "stage2_sideinfo_enabled", False):
            return "\n".join(
                [
                    "**Stage 2 Post-V Adaptive Refinement**",
                    "",
                    rf"$z_0 = V y,\quad \hat{{\delta}}_{{\mathrm{{coarse}}}} = {config.stage2_sideinfo_scale:.2f}\delta,\quad \hat{{\delta}} = \hat{{\delta}}_{{\mathrm{{coarse}}}} + g_\theta(z_0, \hat{{\delta}}_{{\mathrm{{coarse}}}}),\quad \hat{{s}} = G(\hat{{\delta}}) z_0$",
                    "",
                    "- The Stage 1 linear receiver remains the interpretable front-end.",
                    "- Stage 2 starts from a conservative coarse CFO seed and trains a compact residual estimator on top of `z_0` before the MMSE solve.",
                    "- The correction matrix is built from the fixed front-end operator `A(hat(delta)) = V Phi_hat(delta) W` and applied through a regularized symbol-domain solve.",
                    "- `W` and `V` remain frozen in the main Stage 2 experiment so any BER gain is attributable to post-`V` adaptation rather than front-end retuning.",
                    "- BER is decoded from the shared complex-symbol slicer so all methods use identical hard-decision semantics.",
                ]
            )
        return "\n".join(
            [
                "**Stage 2 Post-V Adaptive Refinement**",
                "",
                rf"$z_0 = V y,\quad \hat{{\delta}} = g_\theta(z_0),\quad \hat{{s}} = G(\hat{{\delta}}) z_0$",
                "",
                "- The Stage 1 linear receiver remains the interpretable front-end.",
                "- Stage 2 uses only observable soft-symbol features from `z_0` to estimate the residual CFO; it never receives the true CFO at test time.",
                "- The correction matrix is built from the fixed front-end operator `A(hat(delta)) = V Phi_hat(delta) W` and applied through a regularized symbol-domain solve.",
                "- `W` and `V` remain frozen in the main Stage 2 experiment so any BER gain is attributable to post-`V` adaptation rather than front-end retuning.",
                "- BER is decoded from geometry-derived bit logits thresholded at zero, while EVM continues to use the corrected complex-symbol estimates.",
            ]
        )
    if config.payload_region_enabled:
        return "\n".join(
            [
                "**Structured K-Bin Symbol-to-Waveform Mapping**",
                "",
                rf"$x = W s = \sum_{{k=0}}^{{{config.N - 1}}} s_k w_k$",
                "",
                f"- `s` is a length-`{config.N}` vector of unknown complex `{config.modulation}` data symbols.",
                f"- `W` has shape `({config.M}, {config.N})`, and each learned basis column is constrained to the same `{config.K}`-bin structured subspace.",
                f"- The `{config.M}`-bin frame is split into `{config.K}` learned-data bins, `{config.N_pilots}` reserved pilot bins, and `{config.N_guard}` guard bins.",
                f"- Classical OFDM uses `{config.N}` fixed contiguous data tones inside that `{config.K}`-bin structured subspace, while the learned waveform spreads the same `{config.N}` data symbols across the full `{config.K}`-dimensional subspace.",
                f"- The in-band redundancy is `R = K - N = {config.redundancy_dimensions}`.",
                "- Reserved pilot tones are held aside for future work and are not used for estimation in this notebook.",
            ]
        )
    if config.frame_structure_enabled:
        return "\n".join(
            [
                "**Frame-Structured Symbol-to-Waveform Mapping**",
                "",
                rf"$x = W s = \sum_{{k=0}}^{{{config.N_active - 1}}} s_k w_k$",
                "",
                f"- `s` is a length-`{config.N_active}` active-stream vector with `{config.N_data}` unknown `{config.modulation}` data symbols and `{config.N_pilots}` fixed known pilot symbols.",
                f"- `W` has shape `({config.M}, {config.N_active})`, so each column `w_k` is a transmit basis atom with `{config.M}` time samples.",
                f"- The `64`-bin resource budget is split into `{config.N_guard}` guard bins, `{config.N_pilots}` pilot bins, and `{config.N_data}` payload data bins.",
                "- BER and SER are reported on payload data symbols only; pilot symbols are used for residual-CFO estimation and compensation.",
                "- Classical OFDM and the learned basis share the same gross active/guard resource allocation.",
            ]
        )
    return "\n".join(
        [
            "**Symbol-to-Waveform Mapping**",
            "",
            rf"$x = W s = \sum_{{k=0}}^{{{config.N - 1}}} s_k w_k$",
            "",
            f"- `s` is a length-`{config.N}` vector of complex `{config.modulation}` symbols.",
            f"- `W` has shape `({config.M}, {config.N})`, so each column `w_k` is a learned time-domain basis atom with `{config.M}` samples.",
            f"- The transmitted waveform `x` has `{config.M}` time samples and is formed by weighting and summing all `{config.N}` learned columns of `W`.",
            f"- Classical OFDM uses `{config.N}` contiguous active subcarriers embedded in an `{config.M}`-point IFFT with matched receive basis.",
            "- The paper comparison axis is only `Classical OFDM` versus the learned residual-CFO-robust basis.",
        ]
    )


def build_summary_markdown(
    config: ExperimentConfig,
    training_result: TrainingResult,
    summary_df: pd.DataFrame,
    stage_summary_df: pd.DataFrame,
    snr_summary_df: pd.DataFrame,
    spectral_summary_df: pd.DataFrame,
    papr_summary_df: pd.DataFrame,
    preferred_order: list[str] | tuple[str, ...] | None = None,
) -> str:
    if config.stage2_enabled:
        available_methods = set(summary_df["method"].astype(str).tolist())
        required_four_way = {"OFDM", "OFDMNonlinear", "Learned", "LearnedNonlinear"}
        if not required_four_way.issubset(available_methods):
            lines = ["**Stage 2 Diagnostic Summary**"]
            lines.append(
                f"- Configuration: `{config.modulation}`, `N={config.N}`, `M={config.M}`, train `{config.train_ebn0_db:.0f} dB`, eval `{config.eval_ebn0_db:.0f} dB`."
            )
            lines.append(f"- Stage 1 checkpoint source: `{config.stage2_checkpoint_source_path or config.stage2_checkpoint_path}`.")
            lines.append(
                f"- Methods: `{', '.join(ordered_methods(summary_df['method'].astype(str).unique().tolist(), preferred_order))}`."
            )
            if not stage_summary_df.empty:
                for _, row in stage_summary_df.iterrows():
                    lines.append(
                        f"- {row['scheme']} {row['stage']}: best epoch `{int(row.get('best_stage_epoch', row['best_global_epoch']))}`, "
                        f"val BER `{row['val_ber']:.4e}`, hard-CFO BER `{row['hard_cfo_weighted_ber']:.4e}`."
                    )
            for method in ordered_methods(summary_df["method"].astype(str).unique().tolist(), preferred_order):
                row = summary_df[summary_df["method"] == method].iloc[0]
                lines.append(
                    f"- {method_display_name(method)}: BER(0) `{row['ber_at_0']:.3e}`, "
                    f"BER(0.05) `{row['ber_at_0p05']:.3e}`, BER(0.10) `{row['ber_at_0p10']:.3e}`."
                )
            return "\n".join(lines)

        lines = ["**Stage 2 Four-Way Comparison Summary**"]
        lines.append(
            f"- Configuration: `{config.modulation}`, `N={config.N}`, `M={config.M}`, train `{config.train_ebn0_db:.0f} dB`, eval `{config.eval_ebn0_db:.0f} dB`."
        )
        lines.append(f"- Stage 1 checkpoint source: `{config.stage2_checkpoint_source_path or config.stage2_checkpoint_path}`.")
        lines.append(
            f"- Stage 1 checkpoint snapshot: `{config.stage2_checkpoint_snapshot_path}`, "
            f"format `{config.stage2_checkpoint_format}`, payload hash `{config.stage2_checkpoint_hash_sha256}`, "
            f"file hash `{config.stage2_checkpoint_file_sha256}`, acceptance `{config.stage2_checkpoint_acceptance_passed}`."
        )
        if not stage_summary_df.empty and "identity_bit_mismatch_rate_vs_stage1" in stage_summary_df.columns:
            identity_row = stage_summary_df.iloc[0]
            lines.append(
                f"- Stage 2 identity diagnostic: symbol-MSE-to-`z0` `{identity_row['identity_symbol_mse_to_z0']:.4e}`, "
                f"max-logit diff `{identity_row['identity_max_logit_abs_diff']:.4e}`, "
                f"bit mismatch vs Stage 1 `{identity_row['identity_bit_mismatch_rate_vs_stage1']:.4e}`."
            )
        for _, row in stage_summary_df.iterrows():
            lines.append(
                f"- {row['scheme']} {row['stage']}: best epoch `{int(row.get('best_stage_epoch', row['best_global_epoch']))}`, "
                f"val BER `{row['val_ber']:.4e}`, hard-CFO BER `{row['hard_cfo_weighted_ber']:.4e}`, "
                f"eps-hat MAE `{row.get('eps_hat_mae', float('nan')):.4e}`, blend `{row['residual_scale']:.4f}`."
            )

        ofdm_linear = summary_df[summary_df["method"] == "OFDM"].iloc[0]
        ofdm_stage2 = summary_df[summary_df["method"] == "OFDMNonlinear"].iloc[0]
        learned_linear = summary_df[summary_df["method"] == "Learned"].iloc[0]
        learned_nonlinear = summary_df[summary_df["method"] == "LearnedNonlinear"].iloc[0]
        lines.append(
            f"- BER at `|delta|=0.10`: OFDM `{ofdm_linear['ber_at_0p10']:.3e}`, OFDM + Stage 2 `{ofdm_stage2['ber_at_0p10']:.3e}`, "
            f"Learned `{learned_linear['ber_at_0p10']:.3e}`, Learned + Stage 2 `{learned_nonlinear['ber_at_0p10']:.3e}`."
        )
        lines.append(
            f"- BER at `|delta|=0.05`: OFDM `{ofdm_linear['ber_at_0p05']:.3e}`, OFDM + Stage 2 `{ofdm_stage2['ber_at_0p05']:.3e}`, "
            f"Learned `{learned_linear['ber_at_0p05']:.3e}`, Learned + Stage 2 `{learned_nonlinear['ber_at_0p05']:.3e}`."
        )
        lines.append(
            f"- Robustness window BER <= `0.01`: OFDM `{ofdm_linear['robust_window_ber_le_0.01']:.3f}`, "
            f"OFDM + Stage 2 `{ofdm_stage2['robust_window_ber_le_0.01']:.3f}`, Learned `{learned_linear['robust_window_ber_le_0.01']:.3f}`, "
            f"Learned + Stage 2 `{learned_nonlinear['robust_window_ber_le_0.01']:.3f}`."
        )
        lines.append(
            f"- Stage 2 CFO-estimate MAE at `|delta|=0.10`: OFDM + Stage 2 `{ofdm_stage2['stage2_eps_hat_mae_at_0p10']:.4e}`, "
            f"Learned + Stage 2 `{learned_nonlinear['stage2_eps_hat_mae_at_0p10']:.4e}`."
        )
        return "\n".join(lines)

    learned = summary_df[summary_df["method"] == "Learned"].iloc[0]
    ofdm = summary_df[summary_df["method"] == "OFDM"].iloc[0]

    lines = []
    lines.append("**Stage 1 Linear Comparison Summary**")
    lines.append(
        f"- Configuration: `{config.modulation}`, `N={config.N}`, `M={config.M}`, "
        f"training `{config.train_ebn0_db:.0f} dB`, evaluation `{config.eval_ebn0_db:.0f} dB`."
    )
    if config.payload_region_enabled:
        lines.append(
            f"- Resource split: `K={config.K}` learned-data bins, `P={config.N_pilots}` reserved pilot bins, `G={config.N_guard}` guard bins, "
            f"`R={config.redundancy_dimensions}` redundancy, structured occupancy `{config.K}/{config.M} = {config.payload_region_fraction:.3f}`, "
            f"information occupancy `{config.N}/{config.M} = {config.payload_fraction:.3f}`."
        )
    elif config.frame_structure_enabled:
        lines.append(
            f"- Frame structure: `{config.N_data}` data + `{config.N_pilots}` pilots + `{config.N_guard}` guards, "
            f"payload fraction `{config.N_data}/{config.M} = {config.N_data / config.M:.3f}`."
        )
    acceptance_passed = bool(learned.get("stage1_acceptance_passed", not training_result.stage_failed))
    acceptance_reason = str(learned.get("stage1_acceptance_stop_reason", training_result.stop_reason))
    acceptance_label = "PASS" if acceptance_passed else "FAIL"
    lines.append(f"- Training status: `{training_result.stop_reason}`")
    lines.append(
        f"- Stage 1 acceptance: `{acceptance_label}`. Clean identity <= `{learned['stage1_clean_identity_tol']:.1e}`, "
        f"clean leakage <= `{learned['stage1_clean_leakage_tol']:.1e}`, "
        f"learned BER(0) <= `{learned['stage1_zero_cfo_ber_upper_bound']:.3e}`. "
        f"Result: `{acceptance_reason}`"
    )
    for _, row in stage_summary_df.iterrows():
        lines.append(
            f"- {row['stage']}: clean identity `{row['clean_identity_loss']:.4e}`, "
            f"clean leakage `{row['clean_offdiag_leakage']:.4e}`, "
            f"symbol loss `{row['validation_symbol_loss']:.4e}`, "
            f"`||V||_F^2={row['receiver_fro_norm_sq']:.4e}`."
        )
    lines.append(
        f"- Zero CFO BER: learned `{learned['ber_at_0']:.3e}` vs classical OFDM `{ofdm['ber_at_0']:.3e}`."
    )
    lines.append(
        f"- CFO-robust BER score: learned `{learned['integrated_log10_ber']:.4f}` vs classical OFDM `{ofdm['integrated_log10_ber']:.4f}`."
    )
    lines.append(
        f"- Integrated off-diagonal leakage: learned `{learned['integrated_offdiag_leakage']:.4e}` "
        f"vs classical OFDM `{ofdm['integrated_offdiag_leakage']:.4e}`."
    )
    lines.append(
        f"- Small/mid-CFO nearest-neighbor leakage: learned `{learned['small_mid_integrated_nearest_neighbor_leakage']:.4e}` "
        f"vs classical OFDM `{ofdm['small_mid_integrated_nearest_neighbor_leakage']:.4e}`."
    )
    lines.append(
        f"- EVM at CFO `0.10`: learned `{learned['evm_at_0p10']:.4f}` vs classical OFDM `{ofdm['evm_at_0p10']:.4f}`."
    )
    if config.frame_structure_enabled and config.pilot_estimation_enabled and not config.payload_region_enabled:
        lines.append(
            f"- Pilot CFO MAE at `0.10`: learned `{learned['cfo_est_mae_at_0p10']:.4e}` "
            f"vs classical OFDM `{ofdm['cfo_est_mae_at_0p10']:.4e}`."
        )
        lines.append(
            f"- Pilot MSE at `0.10`: learned `{learned['pilot_symbol_mse_at_0p10']:.4e}` "
            f"vs classical OFDM `{ofdm['pilot_symbol_mse_at_0p10']:.4e}`."
        )
    window_cols = [col for col in summary_df.columns if isinstance(col, str) and col.startswith("robust_window_")]
    for col in sorted(window_cols, key=lambda value: float(value.split("_")[-1])):
        threshold = col.replace("robust_window_ber_le_", "")
        learned_window = float(learned[col])
        ofdm_window = float(ofdm[col])
        if ofdm_window > 0.0:
            pct_gain = 100.0 * (learned_window - ofdm_window) / ofdm_window
            gain_text = f"{pct_gain:+.1f}%"
        else:
            gain_text = "n/a"
        lines.append(
            f"- Robustness window BER <= `{threshold}`: learned `|delta|<={learned_window:.3f}` "
            f"vs classical OFDM `|delta|<={ofdm_window:.3f}` (gain `{gain_text}`)."
        )
    if not snr_summary_df.empty:
        for eps in sorted(snr_summary_df["eps"].unique()):
            learned_row = snr_summary_df[
                (snr_summary_df["method"] == "Learned") & (snr_summary_df["eps"] == eps)
            ].iloc[0]
            ofdm_row = snr_summary_df[
                (snr_summary_df["method"] == "OFDM") & (snr_summary_df["eps"] == eps)
            ].iloc[0]
            lines.append(
                f"- BER-vs-SNR at CFO `{eps:.2f}`: learned `{learned_row['integrated_log10_ber_vs_snr']:.4f}` "
                f"vs classical OFDM `{ofdm_row['integrated_log10_ber_vs_snr']:.4f}`."
            )
    if not spectral_summary_df.empty:
        learned_spec = spectral_summary_df[spectral_summary_df["method"] == "Learned"].iloc[0]
        ofdm_spec = spectral_summary_df[spectral_summary_df["method"] == "OFDM"].iloc[0]
        lines.append(
            f"- 99% occupied bandwidth: learned `{learned_spec['occupied_bandwidth']:.4f}` "
            f"vs classical OFDM `{ofdm_spec['occupied_bandwidth']:.4f}`."
        )
        lines.append(
            f"- Out-of-band power ratio: learned `{learned_spec['out_of_band_power_ratio']:.4e}` "
            f"vs classical OFDM `{ofdm_spec['out_of_band_power_ratio']:.4e}`."
        )
    if not papr_summary_df.empty:
        learned_papr = papr_summary_df[papr_summary_df["method"] == "Learned"].iloc[0]
        ofdm_papr = papr_summary_df[papr_summary_df["method"] == "OFDM"].iloc[0]
        lines.append(
            f"- PAPR 95th percentile: learned `{learned_papr['papr_p95_db']:.2f} dB` "
            f"vs classical OFDM `{ofdm_papr['papr_p95_db']:.2f} dB`."
        )
    return "\n".join(lines)


def run_final_evaluation(
    config: ExperimentConfig,
    training_result: TrainingResult,
    schemes: dict[str, EvaluationScheme] | None = None,
    method_order: tuple[str, ...] | None = None,
) -> ExperimentResult:
    schemes = build_scheme_dict(config, training_result) if schemes is None else schemes
    _log_progress(config, "[Eval] Building operator diagnostics")
    operator_df, diagonal_df = build_operator_df(config, schemes, preferred_order=method_order)
    _log_progress(config, "[Eval] BER vs CFO sweep")
    ber_df = evaluate_scheme_set(
        config=config,
        schemes=schemes,
        cfo_points=config.ber_eval_cfo,
        ebn0_db=config.eval_ebn0_db,
        num_blocks=config.ber_blocks,
        batch_size=config.ber_batch_size,
        progress_label="BER vs CFO",
    )
    ber_df["method_label"] = ber_df["method"].map(method_display_name)
    _log_progress(config, "[Eval] BER vs SNR slices")
    ber_snr_df = build_ber_snr_df(config, schemes)
    _log_progress(config, "[Eval] Constellation snapshots")
    constellation_df = build_constellation_df(config, schemes, preferred_order=method_order)
    _log_progress(config, "[Eval] Spectral and PAPR outputs")
    spectral_df, spectral_summary_df, papr_df, papr_summary_df = build_spectral_outputs(
        config,
        schemes,
        preferred_order=method_order,
    )
    summary_df = build_summary_df(config, training_result, operator_df, ber_df, preferred_order=method_order)
    snr_summary_df = build_snr_summary_df(ber_snr_df, preferred_order=method_order)
    _log_progress(config, "[Eval] Writing CSV artifacts")
    artifact_paths = save_artifacts(
        config,
        training_result,
        operator_df,
        diagonal_df,
        ber_df,
        ber_snr_df,
        constellation_df,
        spectral_df,
        spectral_summary_df,
        papr_df,
        papr_summary_df,
        summary_df,
        snr_summary_df,
    )
    _log_progress(config, "[Eval] Rendering plots")
    artifact_paths["training_plot"] = plot_training_diagnostics(config, training_result.history_df)
    artifact_paths["training_loss_components_plot"] = plot_training_loss_components(config, training_result.history_df)
    operator_heatmap = plot_operator_heatmaps(config, schemes, preferred_order=method_order)
    if operator_heatmap is not None:
        artifact_paths["operator_heatmap_plot"] = operator_heatmap
    if not operator_df.empty:
        artifact_paths["offdiag_plot"] = plot_offdiag_leakage(config, operator_df, preferred_order=method_order)
        nn_plot = plot_nearest_neighbor_leakage(config, operator_df, preferred_order=method_order)
        if nn_plot is not None:
            artifact_paths["nearest_neighbor_plot"] = nn_plot
    ber_plot = plot_ber_curve(config, ber_df, preferred_order=method_order)
    if ber_plot is not None:
        artifact_paths["ber_plot"] = ber_plot
    evm_plot = plot_evm_curve(config, ber_df, preferred_order=method_order)
    if evm_plot is not None:
        artifact_paths["evm_plot"] = evm_plot
    ber_snr_plot = plot_ber_vs_snr_by_cfo(config, ber_snr_df, preferred_order=method_order)
    if ber_snr_plot is not None:
        artifact_paths["ber_snr_plot"] = ber_snr_plot
    constellation_plot = plot_constellation_snapshots(config, constellation_df, preferred_order=method_order)
    if constellation_plot is not None:
        artifact_paths["constellation_plot"] = constellation_plot
    artifact_paths["freq_all_plot"] = plot_frequency_domain_all(config, training_result.learned_tx)
    artifact_paths["freq_random_plot"] = plot_frequency_domain_random(config, training_result.learned_tx)
    artifact_paths["time_waveform_plot"] = plot_time_domain_waveform(config, schemes, preferred_order=method_order)
    artifact_paths["time_envelope_phase_plot"] = plot_time_domain_envelope_phase(
        config,
        schemes,
        preferred_order=method_order,
    )
    papr_ccdf_plot = plot_papr_ccdf(config, papr_df, preferred_order=method_order)
    if papr_ccdf_plot is not None:
        artifact_paths["papr_ccdf_plot"] = papr_ccdf_plot
    spectral_plot = plot_spectral_fairness(config, spectral_df, papr_df, preferred_order=method_order)
    if spectral_plot is not None:
        artifact_paths["spectral_fairness_plot"] = spectral_plot

    mapping_markdown = build_mapping_markdown(config)
    summary_markdown = build_summary_markdown(
        config=config,
        training_result=training_result,
        summary_df=summary_df,
        stage_summary_df=training_result.stage_summary_df,
        snr_summary_df=snr_summary_df,
        spectral_summary_df=spectral_summary_df,
        papr_summary_df=papr_summary_df,
        preferred_order=method_order,
    )
    return ExperimentResult(
        learned_tx=training_result.learned_tx,
        learned_rx=training_result.learned_rx,
        history_df=training_result.history_df,
        stage_summary_df=training_result.stage_summary_df,
        operator_df=operator_df,
        diagonal_df=diagonal_df,
        ber_df=ber_df,
        ber_snr_df=ber_snr_df,
        constellation_df=constellation_df,
        spectral_df=spectral_df,
        spectral_summary_df=spectral_summary_df,
        papr_df=papr_df,
        papr_summary_df=papr_summary_df,
        summary_df=summary_df,
        snr_summary_df=snr_summary_df,
        artifact_paths=artifact_paths,
        mapping_markdown=mapping_markdown,
        summary_markdown=summary_markdown,
        stage_failed=training_result.stage_failed,
        failed_stage=training_result.failed_stage,
        stop_reason=training_result.stop_reason,
    )
