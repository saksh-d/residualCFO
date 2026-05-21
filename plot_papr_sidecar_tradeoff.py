from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reporting import build_papr_constraint_tradeoff_df, plot_papr_constraint_tradeoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PAPR sidecar tradeoff artifacts from existing CSVs.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Sidecar output root containing papr_constraint_trial_summary.csv and papr_constraint_comparison_summary.csv",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    trial_summary_path = root / "papr_constraint_trial_summary.csv"
    comparison_summary_path = root / "papr_constraint_comparison_summary.csv"
    if not trial_summary_path.exists():
        raise FileNotFoundError(f"Missing trial summary: {trial_summary_path}")
    if not comparison_summary_path.exists():
        raise FileNotFoundError(f"Missing comparison summary: {comparison_summary_path}")

    trial_summary_df = pd.read_csv(trial_summary_path)
    comparison_summary_df = pd.read_csv(comparison_summary_path)
    tradeoff_df = build_papr_constraint_tradeoff_df(trial_summary_df, comparison_summary_df)

    tradeoff_csv_path = root / "papr_constraint_tradeoff_summary.csv"
    tradeoff_df.to_csv(tradeoff_csv_path, index=False)
    tradeoff_plot_path = root / "papr_constraint_tradeoff.png"
    plot_papr_constraint_tradeoff(
        tradeoff_df=tradeoff_df,
        comparison_summary_df=comparison_summary_df,
        path=tradeoff_plot_path,
    )

    print(f"Wrote tradeoff CSV -> {tradeoff_csv_path}")
    print(f"Wrote tradeoff plot -> {tradeoff_plot_path}")


if __name__ == "__main__":
    main()
