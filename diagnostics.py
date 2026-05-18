from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from channel import effective_operator
from comm_core import load_stage1_checkpoint
from reporting import method_color, method_display_name
from run_stage2_usrnet import build_stage2_usrnet_config
from transmitter import make_ofdm_baseline_transceiver


SCRIPT_DIR = Path(__file__).resolve().parent

DELTA_VALUES = (0.0, 0.05, 0.10, 0.15)
SUMMARY_RADII = (1, 3, 7, 15, 22)
OUTPUT_ROOT = SCRIPT_DIR / "operator_locality_diagnostics"
OUTPUT_SUBDIR = "usrnet_locality_n45_r9"
METHOD_ORDER = ("OFDM", "Learned")
REFERENCE_OPERATOR_CSV = SCRIPT_DIR / "stage2_usrnet_outputs" / "usrnet_n45_r9" / "operator_diagnostics.csv"
ENERGY_EPS = 1.0e-12
RELATIVE_ZERO_OFFDIAG_TOL = 1.0e-10
PROFILE_TOL = 1.0e-6
ETA_CROSSCHECK_TOL = 1.0e-6
ORDERING_COMPARISON_METHOD = "Learned"


def build_locality_config():
    return build_stage2_usrnet_config(
        output_root=OUTPUT_ROOT,
        output_subdir=OUTPUT_SUBDIR,
        refresh_output_dir=False,
    )


def _distance_masks(size: int) -> dict[int, np.ndarray]:
    indices = np.arange(size, dtype=int)
    distances = np.abs(indices[:, None] - indices[None, :])
    return {distance: distances == distance for distance in range(1, size)}


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > ENERGY_EPS,
    )


def compute_locality_metrics(operator: np.ndarray, distance_masks: dict[int, np.ndarray]) -> dict[str, object]:
    energy = np.abs(operator) ** 2
    diagonal = np.diag(np.diag(operator))
    offdiag = operator - diagonal
    offdiag_energy_matrix = np.abs(offdiag) ** 2

    total_energy = float(energy.sum())
    offdiag_energy = float(offdiag_energy_matrix.sum())
    eta_off = 0.0 if total_energy <= ENERGY_EPS else offdiag_energy / total_energy

    row_total = energy.sum(axis=1)
    row_offdiag = offdiag_energy_matrix.sum(axis=1)
    row_leakage = _safe_ratio(row_offdiag, row_total)

    col_total = energy.sum(axis=0)
    col_offdiag = offdiag_energy_matrix.sum(axis=0)
    col_leakage = _safe_ratio(col_offdiag, col_total)

    num_streams = operator.shape[0]
    zero_offdiag = offdiag_energy <= max(ENERGY_EPS, RELATIVE_ZERO_OFFDIAG_TOL * max(total_energy, ENERGY_EPS))
    if zero_offdiag:
        leakage_by_distance = np.zeros(num_streams - 1, dtype=np.float64)
        cumulative_by_radius = np.ones(num_streams - 1, dtype=np.float64)
        d90 = 0
    else:
        leakage_by_distance = np.asarray(
            [
                float(offdiag_energy_matrix[distance_masks[distance]].sum() / offdiag_energy)
                for distance in range(1, num_streams)
            ],
            dtype=np.float64,
        )
        cumulative_by_radius = np.cumsum(leakage_by_distance)
        d90 = int(np.searchsorted(cumulative_by_radius, 0.90, side="left") + 1)

    return {
        "heatmap": energy,
        "offdiag_heatmap": offdiag_energy_matrix,
        "eta_off": float(eta_off),
        "row_leakage": row_leakage,
        "col_leakage": col_leakage,
        "leakage_by_distance": leakage_by_distance,
        "cumulative_by_radius": cumulative_by_radius,
        "d90": int(d90),
        "zero_offdiag": bool(zero_offdiag),
        "offdiag_energy": float(offdiag_energy),
        "total_energy": float(total_energy),
    }


def compute_spectral_centroid_permutation(tx_basis: torch.Tensor) -> tuple[np.ndarray, pd.DataFrame]:
    spectrum = torch.fft.fft(tx_basis, dim=0).detach().cpu().numpy()
    spectral_energy = np.abs(spectrum) ** 2
    frequency_index = np.arange(spectrum.shape[0], dtype=np.float64)[:, None]
    centroids = np.divide(
        np.sum(frequency_index * spectral_energy, axis=0),
        np.maximum(np.sum(spectral_energy, axis=0), ENERGY_EPS),
    )
    permutation = np.argsort(centroids, kind="stable")
    centroid_df = pd.DataFrame(
        {
            "original_index": np.arange(tx_basis.shape[1], dtype=int),
            "spectral_centroid": centroids.astype(np.float64),
            "sorted_rank": np.argsort(permutation).astype(int),
        }
    ).sort_values("sorted_rank").reset_index(drop=True)
    return permutation.astype(int), centroid_df


def _build_ordering_comparison(
    learned_operators: np.ndarray,
    learned_permutation: np.ndarray,
    distance_masks: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    orderings = {
        "original": None,
        "frequency_sorted": learned_permutation,
    }
    for ordering_name, permutation in orderings.items():
        for delta_index, delta in enumerate(DELTA_VALUES):
            operator = learned_operators[delta_index]
            if permutation is not None:
                operator = operator[np.ix_(permutation, permutation)]
            metrics = compute_locality_metrics(operator, distance_masks)
            summary_row = {
                "method": ORDERING_COMPARISON_METHOD,
                "method_label": method_display_name(ORDERING_COMPARISON_METHOD),
                "ordering": ordering_name,
                "delta": float(delta),
                "eta_off": float(metrics["eta_off"]),
                "d90": int(metrics["d90"]),
                "offdiag_energy": float(metrics["offdiag_energy"]),
                "total_energy": float(metrics["total_energy"]),
                "zero_offdiag": bool(metrics["zero_offdiag"]),
            }
            cumulative = np.asarray(metrics["cumulative_by_radius"], dtype=np.float64)
            for radius in SUMMARY_RADII:
                summary_row[f"C_{radius}"] = float(cumulative[radius - 1])
            summary_rows.append(summary_row)

            leakage_by_distance = np.asarray(metrics["leakage_by_distance"], dtype=np.float64)
            for distance, (leakage_value, cumulative_value) in enumerate(zip(leakage_by_distance, cumulative), start=1):
                distance_rows.append(
                    {
                        "method": ORDERING_COMPARISON_METHOD,
                        "method_label": method_display_name(ORDERING_COMPARISON_METHOD),
                        "ordering": ordering_name,
                        "delta": float(delta),
                        "distance": int(distance),
                        "L_d": float(leakage_value),
                        "C_d": float(cumulative_value),
                    }
                )
    return (
        pd.DataFrame(summary_rows).sort_values(["ordering", "delta"]).reset_index(drop=True),
        pd.DataFrame(distance_rows).sort_values(["ordering", "delta", "distance"]).reset_index(drop=True),
    )


def _plot_heatmaps(
    output_dir: Path,
    metrics_by_method_delta: dict[str, dict[float, dict[str, object]]],
    *,
    field_name: str,
    filename: str,
    title_prefix: str,
) -> Path:
    vmax = max(
        float(np.max(metrics_by_method_delta[method][delta][field_name]))
        for method in METHOD_ORDER
        for delta in DELTA_VALUES
    )
    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        len(DELTA_VALUES),
        figsize=(4.1 * len(DELTA_VALUES), 3.8 * len(METHOD_ORDER)),
        dpi=150,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    for row_idx, method in enumerate(METHOD_ORDER):
        for col_idx, delta in enumerate(DELTA_VALUES):
            ax = axes[row_idx, col_idx]
            image = metrics_by_method_delta[method][delta][field_name]
            last_im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"{method_display_name(method)}\n" + rf"$\delta={delta:.2f}$")
            if row_idx == len(METHOD_ORDER) - 1:
                ax.set_xlabel("Input stream")
            if col_idx == 0:
                ax.set_ylabel("Output stream")
    assert last_im is not None
    fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    path = output_dir / filename
    fig.suptitle(title_prefix, fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_distance_lines(
    output_dir: Path,
    metrics_by_method_delta: dict[str, dict[float, dict[str, object]]],
    *,
    field_name: str,
    filename: str,
    ylabel: str,
    title_prefix: str,
) -> Path:
    fig, axes = plt.subplots(1, len(DELTA_VALUES), figsize=(4.2 * len(DELTA_VALUES), 4.0), dpi=150, sharey=True)
    for axis, delta in zip(axes, DELTA_VALUES):
        for method in METHOD_ORDER:
            values = np.asarray(metrics_by_method_delta[method][delta][field_name], dtype=np.float64)
            distance = np.arange(1, values.size + 1, dtype=int)
            axis.plot(
                distance,
                values,
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                color=method_color(method),
                label=method_display_name(method),
            )
        axis.set_title(rf"$\delta={delta:.2f}$")
        axis.set_xlabel("Distance" if field_name == "leakage_by_distance" else "Radius")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title_prefix, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_rowwise_leakage(
    output_dir: Path,
    metrics_by_method_delta: dict[str, dict[float, dict[str, object]]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), dpi=150, sharex=True, sharey=True)
    for axis, delta in zip(axes.flat, DELTA_VALUES):
        for method in METHOD_ORDER:
            values = np.asarray(metrics_by_method_delta[method][delta]["row_leakage"], dtype=np.float64)
            stream_index = np.arange(values.size, dtype=int)
            axis.plot(
                stream_index,
                values,
                linewidth=2.0,
                color=method_color(method),
                label=method_display_name(method),
            )
        axis.set_title(rf"$\delta={delta:.2f}$")
        axis.set_xlabel("Row index")
        axis.set_ylabel("Row leakage")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Row-wise leakage by residual CFO", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / "rowwise_leakage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _sanity_checks(
    summary_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    metrics_by_method_delta: dict[str, dict[float, dict[str, object]]],
) -> None:
    for method in METHOD_ORDER:
        for delta in DELTA_VALUES:
            metrics = metrics_by_method_delta[method][delta]
            leakage = np.asarray(metrics["leakage_by_distance"], dtype=np.float64)
            cumulative = np.asarray(metrics["cumulative_by_radius"], dtype=np.float64)
            if metrics["zero_offdiag"]:
                if not np.allclose(leakage, 0.0, atol=1e-12):
                    raise ValueError(f"Expected zero leakage profile for {method} delta={delta:.2f}.")
                if not np.allclose(cumulative, 1.0, atol=1e-12):
                    raise ValueError(f"Expected unit cumulative profile for zero-offdiag case {method} delta={delta:.2f}.")
            else:
                if np.any(leakage < -1e-12):
                    raise ValueError(f"Negative distance leakage for {method} delta={delta:.2f}.")
                if abs(float(leakage.sum()) - 1.0) > PROFILE_TOL:
                    raise ValueError(f"Distance leakage must sum to 1 for {method} delta={delta:.2f}.")
                if np.any(np.diff(cumulative) < -PROFILE_TOL):
                    raise ValueError(f"Cumulative leakage must be monotone for {method} delta={delta:.2f}.")
                if np.any(cumulative < -PROFILE_TOL) or np.any(cumulative > 1.0 + PROFILE_TOL):
                    raise ValueError(f"Cumulative leakage out of range for {method} delta={delta:.2f}.")
                d90 = int(metrics["d90"])
                if d90 < 1 or cumulative[d90 - 1] < 0.90 - PROFILE_TOL:
                    raise ValueError(f"Invalid d90 for {method} delta={delta:.2f}.")
                if d90 > 1 and cumulative[d90 - 2] >= 0.90 - PROFILE_TOL:
                    raise ValueError(f"d90 is not minimal for {method} delta={delta:.2f}.")
    if summary_df.empty or distance_df.empty:
        raise ValueError("Expected non-empty CSV outputs.")


def _cross_check_reference(summary_df: pd.DataFrame) -> pd.DataFrame:
    if not REFERENCE_OPERATOR_CSV.exists():
        return pd.DataFrame()
    reference_df = pd.read_csv(REFERENCE_OPERATOR_CSV)
    reference_df = reference_df[reference_df["method"].isin(METHOD_ORDER)][["method", "eps", "offdiag_leakage"]].copy()
    reference_df["delta_key"] = reference_df["eps"].round(8)
    reference_df = reference_df[reference_df["delta_key"].isin([round(delta, 8) for delta in DELTA_VALUES])]
    reference_df = reference_df.rename(columns={"eps": "delta", "offdiag_leakage": "eta_off_reference"})
    summary_with_key = summary_df.copy()
    summary_with_key["delta_key"] = summary_with_key["delta"].round(8)
    merged = summary_with_key.merge(reference_df[["method", "delta_key", "eta_off_reference"]], on=["method", "delta_key"], how="left")
    merged["eta_off_abs_diff"] = np.abs(merged["eta_off"] - merged["eta_off_reference"])
    if merged["eta_off_reference"].isna().any():
        raise ValueError("Missing reference eta_off values for the requested method/delta pairs.")
    max_abs_diff = float(merged["eta_off_abs_diff"].max())
    if max_abs_diff > ETA_CROSSCHECK_TOL:
        raise ValueError(f"eta_off cross-check against operator_diagnostics.csv failed: max abs diff={max_abs_diff:.3e}")
    return merged.drop(columns=["delta_key"])


def _print_summary(summary_df: pd.DataFrame, cross_check_df: pd.DataFrame) -> None:
    print("Operator locality summary")
    for method in METHOD_ORDER:
        print(f"\n{method_display_name(method)}")
        method_df = summary_df[summary_df["method"] == method].sort_values("delta")
        for _, row in method_df.iterrows():
            print(
                "  "
                f"delta={row['delta']:.2f} "
                f"eta_off={row['eta_off']:.10f} "
                f"C(1)={row['C_1']:.10f} "
                f"C(3)={row['C_3']:.10f} "
                f"C(7)={row['C_7']:.10f} "
                f"C(15)={row['C_15']:.10f} "
                f"C(22)={row['C_22']:.10f} "
                f"d90={int(row['d90'])}"
            )
    if not cross_check_df.empty:
        print("\neta_off cross-check against stage2_usrnet_outputs/usrnet_n45_r9/operator_diagnostics.csv")
        for _, row in cross_check_df.sort_values(["method", "delta"]).iterrows():
            print(
                "  "
                f"{row['method']} delta={row['delta']:.2f} "
                f"new={row['eta_off']:.10f} ref={row['eta_off_reference']:.10f} "
                f"abs_diff={row['eta_off_abs_diff']:.3e}"
            )
    print("\nInterpretation:")
    print("  If Learned has much lower eta_off than OFDM, Stage 1 reduces total smearing.")
    print("  If Learned has smaller d90 than OFDM, Stage 1 localizes the remaining smearing.")
    print("  If Learned has low eta_off but large d90, Stage 1 reduces global leakage magnitude but does not localize it.")


def _print_ordering_comparison(centroid_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    print(f"\n{method_display_name(ORDERING_COMPARISON_METHOD)} frequency-centroid ordering")
    print(
        "  sorted permutation = "
        + repr(centroid_df["original_index"].astype(int).tolist())
    )
    print(
        "  sorted centroids = "
        + repr([round(float(value), 4) for value in centroid_df["spectral_centroid"].tolist()])
    )
    print("\nLearned ordering comparison")
    for delta in DELTA_VALUES:
        original = comparison_df[
            (comparison_df["ordering"] == "original") & np.isclose(comparison_df["delta"], delta)
        ].iloc[0]
        reordered = comparison_df[
            (comparison_df["ordering"] == "frequency_sorted") & np.isclose(comparison_df["delta"], delta)
        ].iloc[0]
        print(
            "  "
            f"delta={delta:.2f} "
            f"original: eta_off={original['eta_off']:.10f}, d90={int(original['d90'])}, "
            f"C(1)={original['C_1']:.10f}, C(3)={original['C_3']:.10f}, C(7)={original['C_7']:.10f}, "
            f"C(15)={original['C_15']:.10f}, C(22)={original['C_22']:.10f}"
        )
        print(
            "  "
            f"delta={delta:.2f} "
            f"sorted:   eta_off={reordered['eta_off']:.10f}, d90={int(reordered['d90'])}, "
            f"C(1)={reordered['C_1']:.10f}, C(3)={reordered['C_3']:.10f}, C(7)={reordered['C_7']:.10f}, "
            f"C(15)={reordered['C_15']:.10f}, C(22)={reordered['C_22']:.10f}"
        )


def run_operator_locality_diagnostic() -> Path:
    config = build_locality_config()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_stage1_checkpoint(config.stage2_checkpoint_path, device=config.device, require_v2=True)
    learned_tx = checkpoint["learned_tx"].to(config.device)
    learned_rx = checkpoint["learned_rx"].to(config.device)
    ofdm_tx, ofdm_rx = make_ofdm_baseline_transceiver(config)

    schemes = {
        "OFDM": (ofdm_tx, ofdm_rx),
        "Learned": (learned_tx, learned_rx),
    }
    learned_permutation, centroid_df = compute_spectral_centroid_permutation(learned_tx)
    distance_masks = _distance_masks(config.N)
    delta_tensor = torch.tensor(DELTA_VALUES, device=config.device, dtype=torch.float32)

    metrics_by_method_delta: dict[str, dict[float, dict[str, object]]] = {method: {} for method in METHOD_ORDER}
    operators_by_method: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for method in METHOD_ORDER:
        tx_basis, rx_basis = schemes[method]
        operators = effective_operator(tx_basis, rx_basis, delta_tensor).detach().cpu().numpy()
        operators_by_method[method] = operators
        for delta_index, delta in enumerate(DELTA_VALUES):
            metrics = compute_locality_metrics(operators[delta_index], distance_masks)
            metrics_by_method_delta[method][float(delta)] = metrics

            summary_row = {
                "method": method,
                "method_label": method_display_name(method),
                "delta": float(delta),
                "eta_off": float(metrics["eta_off"]),
                "d90": int(metrics["d90"]),
                "offdiag_energy": float(metrics["offdiag_energy"]),
                "total_energy": float(metrics["total_energy"]),
                "zero_offdiag": bool(metrics["zero_offdiag"]),
            }
            cumulative = np.asarray(metrics["cumulative_by_radius"], dtype=np.float64)
            for radius in SUMMARY_RADII:
                summary_row[f"C_{radius}"] = float(cumulative[radius - 1])
            summary_rows.append(summary_row)

            leakage_by_distance = np.asarray(metrics["leakage_by_distance"], dtype=np.float64)
            for distance, (leakage_value, cumulative_value) in enumerate(zip(leakage_by_distance, cumulative), start=1):
                distance_rows.append(
                    {
                        "method": method,
                        "method_label": method_display_name(method),
                        "delta": float(delta),
                        "distance": int(distance),
                        "L_d": float(leakage_value),
                        "C_d": float(cumulative_value),
                    }
                )

            for axis_name, values in (("row", metrics["row_leakage"]), ("column", metrics["col_leakage"])):
                leakage_values = np.asarray(values, dtype=np.float64)
                for index, leakage_value in enumerate(leakage_values):
                    leakage_rows.append(
                        {
                            "method": method,
                            "method_label": method_display_name(method),
                            "delta": float(delta),
                            "axis": axis_name,
                            "index": int(index),
                            "leakage": float(leakage_value),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "delta"]).reset_index(drop=True)
    distance_df = pd.DataFrame(distance_rows).sort_values(["method", "delta", "distance"]).reset_index(drop=True)
    leakage_df = pd.DataFrame(leakage_rows).sort_values(["method", "delta", "axis", "index"]).reset_index(drop=True)
    ordering_summary_df, ordering_distance_df = _build_ordering_comparison(
        operators_by_method[ORDERING_COMPARISON_METHOD],
        learned_permutation,
        distance_masks,
    )

    _sanity_checks(summary_df, distance_df, metrics_by_method_delta)
    cross_check_df = _cross_check_reference(summary_df)

    summary_df.to_csv(output_dir / "operator_locality_summary.csv", index=False)
    distance_df.to_csv(output_dir / "leakage_distance_profile.csv", index=False)
    leakage_df.to_csv(output_dir / "row_column_leakage.csv", index=False)
    centroid_df.to_csv(output_dir / "learned_spectral_centroids.csv", index=False)
    ordering_summary_df.to_csv(output_dir / "learned_ordering_locality_summary.csv", index=False)
    ordering_distance_df.to_csv(output_dir / "learned_ordering_distance_profile.csv", index=False)

    _plot_heatmaps(
        output_dir,
        metrics_by_method_delta,
        field_name="heatmap",
        filename="operator_heatmaps_locality.png",
        title_prefix=r"Full operator heatmaps: $|A(\delta)|^2$",
    )
    _plot_heatmaps(
        output_dir,
        metrics_by_method_delta,
        field_name="offdiag_heatmap",
        filename="offdiag_heatmaps_locality.png",
        title_prefix=r"Off-diagonal operator heatmaps: $|A(\delta) - \mathrm{diag}(A(\delta))|^2$",
    )
    _plot_distance_lines(
        output_dir,
        metrics_by_method_delta,
        field_name="leakage_by_distance",
        filename="leakage_vs_distance.png",
        ylabel=r"$L(d)$",
        title_prefix="Distance leakage profiles",
    )
    _plot_distance_lines(
        output_dir,
        metrics_by_method_delta,
        field_name="cumulative_by_radius",
        filename="cumulative_leakage_vs_radius.png",
        ylabel=r"$C(R)$",
        title_prefix="Cumulative leakage locality",
    )
    _plot_rowwise_leakage(output_dir, metrics_by_method_delta)

    _print_summary(summary_df, cross_check_df)
    _print_ordering_comparison(centroid_df, ordering_summary_df)
    print(f"\nSaved outputs to {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_operator_locality_diagnostic()
