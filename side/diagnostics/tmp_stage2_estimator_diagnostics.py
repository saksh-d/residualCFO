from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channel import propagate
from comm_core import StageSpec, load_stage1_checkpoint, set_seed
from receiver import (
    CfoEstimatorMmseReceiver,
    _sample_training_cfo_values,
    _sample_training_ebn0_values,
    _stage2_estimator_max_abs_cfo,
    _stage2_training_abs_cfo_scale,
    decode_symbols,
)
from run_stage2_blind_mmse import build_stage2_blind_mmse_config
from transmitter import sample_training_symbols, transmit_symbols


OUTPUT_ROOT = SCRIPT_DIR / "outputs"
OUTPUT_SUBDIR = "tmp_estimator_diagnostics_n45_r9"
REFRESH_OUTPUT_DIR = True

SCATTER_POINTS_PER_ARCH = 4000
EVAL_BLOCKS_PER_CFO = 1024
EVAL_BATCH_SIZE = 256
TRAIN_ARCHITECTURES = ("LOCAL", "DENSE")
EVAL_CFO_GRID = np.linspace(-0.20, 0.20, 41)


def build_stage2_estimator_diagnostic_config(
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


def _prepare_output_dir(output_dir: Path, refresh: bool) -> None:
    if refresh and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                for nested in sorted((p for p in child.rglob("*") if p.is_dir()), reverse=True):
                    nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_stage1_bases(config: object) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint_path = Path(config.stage2_checkpoint_path)
    print(f"[EstimatorDiag] Loading Stage 1 checkpoint -> {checkpoint_path}", flush=True)
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
    return checkpoint["learned_tx"].to(config.device), checkpoint["learned_rx"].to(config.device)


def _build_receiver(config: object, architecture: str, tx_basis: torch.Tensor, rx_basis: torch.Tensor) -> CfoEstimatorMmseReceiver:
    return CfoEstimatorMmseReceiver(
        tx_basis=tx_basis,
        rx_basis=rx_basis,
        num_symbols=config.N,
        bits_per_symbol=config.bits_per_symbol,
        architecture=architecture,
        channels=config.stage2_local_channels,
        hidden_multiplier=config.stage2_hidden_multiplier,
        kernel_size=config.stage2_local_kernel_size,
        max_abs_eps=_stage2_estimator_max_abs_cfo(config),
    ).to(config.device)


def _estimator_features_batch(
    config: object,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    stage: StageSpec,
    batch_size: int,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, symbols = sample_training_symbols(batch_size, config)
    eps_values = _sample_training_cfo_values(config, stage, batch_size, deterministic=deterministic)
    ebn0_values = _sample_training_ebn0_values(config, batch_size, deterministic=deterministic)
    tx_signal = transmit_symbols(symbols, tx_basis)
    rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_values)
    z0 = decode_symbols(rx_signal, rx_basis)
    return z0, eps_values.to(torch.float32)


def _train_estimator_only(
    config: object,
    architecture: str,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
    seed_offset: int,
) -> tuple[CfoEstimatorMmseReceiver, pd.DataFrame, dict[str, float]]:
    set_seed(config.base_seed + seed_offset)
    receiver = _build_receiver(config, architecture, tx_basis, rx_basis)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=config.stage2_nonlinear_only_learning_rate)
    stage = StageSpec("EstimatorOnly", config.stage2_nonlinear_only_epochs, config.stage_c_cfo, True)
    eps_scale = _stage2_training_abs_cfo_scale(config)
    val_batch_size = min(config.train_symbol_batch_size, 256)
    val_z0, val_eps = _estimator_features_batch(
        config,
        tx_basis,
        rx_basis,
        stage,
        val_batch_size,
        deterministic=True,
    )
    history: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"[EstimatorDiag] {architecture} start | epochs={config.stage2_nonlinear_only_epochs}", flush=True)
    for epoch in range(config.stage2_nonlinear_only_epochs):
        train_z0, train_eps = _estimator_features_batch(
            config,
            tx_basis,
            rx_basis,
            stage,
            config.train_symbol_batch_size,
            deterministic=False,
        )
        eps_hat = receiver._estimate_eps(train_z0)
        train_loss = torch.mean(((eps_hat - train_eps) / eps_scale) ** 2)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(receiver.parameters(), 5.0)
        optimizer.step()

        should_log = (
            epoch == 0
            or epoch + 1 == config.stage2_nonlinear_only_epochs
            or (epoch + 1) % config.history_log_interval == 0
        )
        if not should_log:
            continue

        with torch.no_grad():
            val_eps_hat = receiver._estimate_eps(val_z0)
            val_loss = torch.mean(((val_eps_hat - val_eps) / eps_scale) ** 2)
            val_mae = torch.mean(torch.abs(val_eps_hat - val_eps)).item()
            val_bias = torch.mean(val_eps_hat - val_eps).item()
            row = {
                "architecture": architecture,
                "epoch": epoch + 1,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "val_mae": float(val_mae),
                "val_bias": float(val_bias),
                "eps_hat_mean": float(torch.mean(val_eps_hat).item()),
                "eps_hat_min": float(torch.min(val_eps_hat).item()),
                "eps_hat_max": float(torch.max(val_eps_hat).item()),
            }
            history.append(row)
            print(
                (
                    f"[EstimatorDiag] {architecture} {epoch + 1}/{config.stage2_nonlinear_only_epochs} "
                    f"| train_loss={row['train_loss']:.4e} val_loss={row['val_loss']:.4e} "
                    f"val_mae={row['val_mae']:.4e} eps_hat_mean={row['eps_hat_mean']:.4e}"
                ),
                flush=True,
            )
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_epoch = epoch + 1
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in receiver.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No validation state captured for architecture {architecture}.")
    receiver.load_state_dict(best_state)
    receiver.eval()
    summary = {
        "architecture": architecture,
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    return receiver, pd.DataFrame(history), summary


def _evaluate_estimator(
    config: object,
    architecture: str,
    receiver: CfoEstimatorMmseReceiver,
    tx_basis: torch.Tensor,
    rx_basis: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    scatter_rng = np.random.default_rng(config.base_seed + (0 if architecture == "LOCAL" else 1000))
    scatter_target = SCATTER_POINTS_PER_ARCH

    for eps in EVAL_CFO_GRID:
        remaining = EVAL_BLOCKS_PER_CFO
        while remaining > 0:
            batch_size = min(EVAL_BATCH_SIZE, remaining)
            _, symbols = sample_training_symbols(batch_size, config)
            eps_values = torch.full((batch_size,), float(eps), device=config.device, dtype=torch.float32)
            ebn0_values = _sample_training_ebn0_values(config, batch_size, deterministic=False)
            tx_signal = transmit_symbols(symbols, tx_basis)
            rx_signal, _ = propagate(tx_signal, eps_values, config, ebn0_db=ebn0_values)
            z0 = decode_symbols(rx_signal, rx_basis)
            with torch.no_grad():
                eps_hat = receiver._estimate_eps(z0).detach().cpu().numpy()
            keep_mask = scatter_rng.random(batch_size) < min(1.0, scatter_target / (len(EVAL_CFO_GRID) * EVAL_BLOCKS_PER_CFO))
            for idx, value in enumerate(eps_hat):
                rows.append(
                    {
                        "architecture": architecture,
                        "eps_true": float(eps),
                        "eps_hat": float(value),
                        "scatter_keep": int(keep_mask[idx]),
                    }
                )
            remaining -= batch_size

    raw_df = pd.DataFrame(rows)
    grid_df = (
        raw_df.groupby(["architecture", "eps_true"], as_index=False)["eps_hat"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "eps_hat_mean",
                "std": "eps_hat_std",
                "min": "eps_hat_min",
                "max": "eps_hat_max",
            }
        )
    )
    return raw_df, grid_df


def _summary_from_eval(architecture: str, raw_df: pd.DataFrame, grid_df: pd.DataFrame) -> dict[str, float | str]:
    eps_true = raw_df["eps_true"].to_numpy(dtype=float)
    eps_hat = raw_df["eps_hat"].to_numpy(dtype=float)
    in_range_mask = np.abs(eps_true) <= 0.10 + 1e-12
    sign_mask = np.abs(eps_true) > 0.03
    zero_mask = np.isclose(eps_true, 0.0)
    sign_accuracy = float(np.mean(np.sign(eps_true[sign_mask]) == np.sign(eps_hat[sign_mask]))) if np.any(sign_mask) else float("nan")
    slope, intercept = np.polyfit(
        grid_df["eps_true"].to_numpy(dtype=float),
        grid_df["eps_hat_mean"].to_numpy(dtype=float),
        deg=1,
    )
    return {
        "architecture": architecture,
        "mae_full_range": float(np.mean(np.abs(eps_hat - eps_true))),
        "mae_in_range_le_0p10": float(np.mean(np.abs(eps_hat[in_range_mask] - eps_true[in_range_mask]))),
        "bias_full_range": float(np.mean(eps_hat - eps_true)),
        "sign_accuracy_abs_delta_gt_0p03": sign_accuracy,
        "zero_bias_mean": float(np.mean(eps_hat[zero_mask])),
        "zero_bias_std": float(np.std(eps_hat[zero_mask])),
        "zero_bias_p10": float(np.quantile(eps_hat[zero_mask], 0.10)),
        "zero_bias_p50": float(np.quantile(eps_hat[zero_mask], 0.50)),
        "zero_bias_p90": float(np.quantile(eps_hat[zero_mask], 0.90)),
        "mean_curve_slope": float(slope),
        "mean_curve_intercept": float(intercept),
        "eps_hat_min": float(np.min(eps_hat)),
        "eps_hat_max": float(np.max(eps_hat)),
    }


def _save_plots(output_dir: Path, raw_df: pd.DataFrame, grid_df: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    combined_fig = plt.figure(figsize=(7.4, 5.6), dpi=140)
    for architecture, color in (("LOCAL", "#2E6F95"), ("DENSE", "#C8553D")):
        arch_scatter = raw_df[(raw_df["architecture"] == architecture) & (raw_df["scatter_keep"] == 1)]
        arch_grid = grid_df[grid_df["architecture"] == architecture]
        plt.scatter(
            arch_scatter["eps_true"],
            arch_scatter["eps_hat"],
            s=10,
            alpha=0.15,
            color=color,
            label=f"{architecture} samples",
        )
        plt.plot(
            arch_grid["eps_true"],
            arch_grid["eps_hat_mean"],
            linewidth=2.2,
            color=color,
            label=f"{architecture} mean",
        )
    plt.plot(EVAL_CFO_GRID, EVAL_CFO_GRID, linestyle="--", linewidth=1.2, color="#222222", label="ideal")
    plt.xlabel("True residual CFO")
    plt.ylabel("Estimated residual CFO")
    plt.title(r"Estimator diagnostics: $\hat{\delta}$ vs $\delta$")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    combined_path = output_dir / "delta_hat_scatter_combined.png"
    combined_fig.savefig(combined_path, bbox_inches="tight")
    plt.close(combined_fig)
    paths["combined_scatter"] = combined_path

    mean_fig = plt.figure(figsize=(7.4, 5.2), dpi=140)
    for architecture, color in (("LOCAL", "#2E6F95"), ("DENSE", "#C8553D")):
        arch_grid = grid_df[grid_df["architecture"] == architecture]
        plt.plot(
            arch_grid["eps_true"],
            arch_grid["eps_hat_mean"],
            marker="o",
            linewidth=2.1,
            color=color,
            label=architecture,
        )
    plt.plot(EVAL_CFO_GRID, EVAL_CFO_GRID, linestyle="--", linewidth=1.2, color="#222222", label="ideal")
    plt.xlabel("True residual CFO")
    plt.ylabel("Mean estimated residual CFO")
    plt.title(r"Mean $\hat{\delta}$ versus true $\delta$")
    plt.grid(True, alpha=0.25)
    plt.legend()
    mean_path = output_dir / "delta_hat_mean_vs_true.png"
    mean_fig.savefig(mean_path, bbox_inches="tight")
    plt.close(mean_fig)
    paths["mean_curve"] = mean_path

    for architecture, color in (("LOCAL", "#2E6F95"), ("DENSE", "#C8553D")):
        arch_scatter = raw_df[(raw_df["architecture"] == architecture) & (raw_df["scatter_keep"] == 1)]
        fig = plt.figure(figsize=(7.0, 5.2), dpi=140)
        plt.scatter(arch_scatter["eps_true"], arch_scatter["eps_hat"], s=10, alpha=0.18, color=color)
        plt.plot(EVAL_CFO_GRID, EVAL_CFO_GRID, linestyle="--", linewidth=1.2, color="#222222")
        plt.xlabel("True residual CFO")
        plt.ylabel("Estimated residual CFO")
        plt.title(f"{architecture}: " + r"$\hat{\delta}$ vs $\delta$")
        plt.grid(True, alpha=0.25)
        scatter_path = output_dir / f"{architecture.lower()}_delta_hat_scatter.png"
        fig.savefig(scatter_path, bbox_inches="tight")
        plt.close(fig)
        paths[f"{architecture.lower()}_scatter"] = scatter_path

        zero_df = raw_df[(raw_df["architecture"] == architecture) & np.isclose(raw_df["eps_true"], 0.0)]
        hist_fig = plt.figure(figsize=(7.0, 4.8), dpi=140)
        plt.hist(zero_df["eps_hat"], bins=40, color=color, alpha=0.88)
        plt.xlabel(r"$\hat{\delta}$ when true $\delta = 0$")
        plt.ylabel("Count")
        plt.title(f"{architecture}: " + r"distribution of $\hat{\delta}$ at true $\delta = 0$")
        plt.grid(True, alpha=0.25)
        hist_path = output_dir / f"{architecture.lower()}_delta_hat_zero_hist.png"
        hist_fig.savefig(hist_path, bbox_inches="tight")
        plt.close(hist_fig)
        paths[f"{architecture.lower()}_zero_hist"] = hist_path

    return paths


def _write_report(
    output_dir: Path,
    config: object,
    architecture_summary_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> Path:
    lines = [
        "# Temporary Stage 2 Estimator Diagnostics",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Stage 1 checkpoint: `{config.stage2_checkpoint_path}`",
        f"- Training support: `|delta| <= {config.stage_c_cfo:.2f}`",
        f"- Evaluation grid: `[{EVAL_CFO_GRID[0]:+.2f}, {EVAL_CFO_GRID[-1]:+.2f}]` with `{len(EVAL_CFO_GRID)}` points",
        f"- Architectures: `{', '.join(TRAIN_ARCHITECTURES)}`",
        f"- Estimator loss: normalized CFO MSE only",
        "",
        "## Architecture Summary",
        "",
        "```text",
        architecture_summary_df.to_string(index=False),
        "```",
        "",
        "## Figures",
        "",
    ]
    for name, path in plot_paths.items():
        lines.append(f"- `{name}`: [{path.name}]({path.name})")
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def run_stage2_estimator_diagnostics(config: object | None = None) -> dict[str, object]:
    config = build_stage2_estimator_diagnostic_config() if config is None else config
    output_dir = Path(config.output_dir)
    _prepare_output_dir(output_dir, config.refresh_output_dir)
    learned_tx, learned_rx = _load_stage1_bases(config)

    architecture_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    grid_frames: list[pd.DataFrame] = []

    for seed_offset, architecture in enumerate(TRAIN_ARCHITECTURES):
        receiver, history_df, train_summary = _train_estimator_only(
            config,
            architecture,
            learned_tx,
            learned_rx,
            seed_offset=seed_offset,
        )
        raw_df, grid_df = _evaluate_estimator(config, architecture, receiver, learned_tx, learned_rx)
        architecture_summary = _summary_from_eval(architecture, raw_df, grid_df)
        architecture_summary.update(train_summary)
        architecture_frames.append(pd.DataFrame([architecture_summary]))
        history_frames.append(history_df)
        raw_frames.append(raw_df)
        grid_frames.append(grid_df)

    architecture_summary_df = pd.concat(architecture_frames, ignore_index=True).sort_values(
        ["mae_full_range", "mae_in_range_le_0p10"],
        kind="stable",
    )
    history_df = pd.concat(history_frames, ignore_index=True)
    raw_df = pd.concat(raw_frames, ignore_index=True)
    grid_df = pd.concat(grid_frames, ignore_index=True)

    architecture_summary_df.to_csv(output_dir / "architecture_summary.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    raw_df.to_csv(output_dir / "delta_hat_samples.csv", index=False)
    grid_df.to_csv(output_dir / "delta_hat_grid_summary.csv", index=False)
    plot_paths = _save_plots(output_dir, raw_df, grid_df)
    report_path = _write_report(output_dir, config, architecture_summary_df, plot_paths)

    print("\n=== architecture summary ===")
    print(architecture_summary_df.to_string(index=False))
    print("\n=== report ===")
    print(report_path)
    return {
        "architecture_summary_df": architecture_summary_df,
        "history_df": history_df,
        "raw_df": raw_df,
        "grid_df": grid_df,
        "plot_paths": plot_paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_stage2_estimator_diagnostics()
