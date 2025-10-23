"""
Clean labels.csv by removing row-level outliers and suspicious flatlined values.

Rules (configurable via CLI flags):
- Six Sigma filter: drop any row where at least one numeric feature (excluding
  'filename', 'session_id', and 'blood_glucose') lies outside mean ± 6*std for
  that feature. Columns with zero/NaN std are ignored for sigma filtering.
- Flatline filter: for each numeric feature, if the most frequent value is 0 or
  equals the column maximum and it occurs in >= threshold proportion of rows,
  drop rows where the feature equals that value. This is aimed at artifacts
  where many samples are exactly 0 or clipped to the same max value.

Outputs:
- eye_glucose_data/labels_clean.csv      (kept rows)
- eye_glucose_data/labels_removed.csv    (removed rows with reasons)
- eye_glucose_data/cleaning_report.json  (summary stats)

Usage examples:
  python clean_labels_sixsigma.py \
    --input eye_glucose_data/labels.csv \
    --output eye_glucose_data/labels_clean.csv

  # Adjust flatline sensitivity to 5%
  python clean_labels_sixsigma.py --flatline-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd


EXCLUDE_COLS = {"filename", "session_id", "blood_glucose"}


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns suitable for outlier checks, excluding known ID/label cols."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def sixsigma_bounds(series: pd.Series) -> Tuple[float, float]:
    """Compute mean ± 6*std bounds, handling degenerate std (0 or NaN)."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0 or np.isnan(sigma):
        # Degenerate distribution; return infinities so it won't filter anything for this column
        return -math.inf, math.inf
    return float(mu - 6 * sigma), float(mu + 6 * sigma)


def detect_flatline_value(series: pd.Series, threshold: float) -> Tuple[bool, float | None, str | None]:
    """Detect if a column has a suspicious flatlined value (0 or max) occurring >= threshold proportion.

    This checks zero and max frequencies independently (not only the modal value),
    then returns (is_flatlined, value, reason) where reason is 'zero' or 'max'.
    """
    s = series.dropna()
    n = len(s)
    if n == 0:
        return False, None, None

    # Zero frequency
    zero_mask = np.isclose(s.astype(float), 0.0, rtol=0, atol=1e-9)
    zero_count = int(zero_mask.sum())
    zero_prop = zero_count / n

    if zero_prop >= threshold and zero_count > 0:
        return True, 0.0, "zero"

    # Max frequency
    max_value = float(s.max())
    max_mask = np.isclose(s.astype(float), max_value, rtol=0, atol=1e-9)
    max_count = int(max_mask.sum())
    max_prop = max_count / n

    if max_prop >= threshold and max_count > 0:
        return True, max_value, "max"

    return False, None, None


def clean_dataframe(
    df: pd.DataFrame,
    flatline_threshold: float = 0.02,
    zero_threshold: float | None = None,
    max_threshold: float | None = None,
    drop_any_zero: bool = False,
    drop_any_max: bool = False,
    min_nonnull_ratio: float = 0.2,
    min_nonnull_count: int = 100,
    ignore_columns: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Return (clean_df, removed_df, report)."""
    ignore_set = set([c.strip() for c in (ignore_columns or []) if str(c).strip()])
    num_cols = [c for c in numeric_feature_columns(df) if c not in ignore_set]

    # Build sigma masks per-column
    sigma_bounds: Dict[str, Tuple[float, float]] = {c: sixsigma_bounds(df[c]) for c in num_cols}
    sigma_outlier_mask = pd.Series(False, index=df.index)
    per_col_sigma_outliers: Dict[str, int] = {}
    for c in num_cols:
        lo, hi = sigma_bounds[c]
        mask = (df[c] < lo) | (df[c] > hi)
        per_col_sigma_outliers[c] = int(mask.sum())
        sigma_outlier_mask |= mask

    # Flatline detection
    flatline_values: Dict[str, Dict] = {}
    flatline_mask = pd.Series(False, index=df.index)
    zero_max_counts: Dict[str, Dict[str, int]] = {}
    for c in num_cols:
        col = df[c]
        is_flat, v, reason = detect_flatline_value(col, flatline_threshold)
        # Track zero/max counts regardless of threshold for reporting
        s = col.dropna().astype(float)
        zero_ct = int(np.isclose(s, 0.0, rtol=0, atol=1e-9).sum())
        max_val = float(s.max()) if len(s) > 0 else float("nan")
        max_ct = int(np.isclose(s, max_val, rtol=0, atol=1e-9).sum()) if len(s) > 0 else 0
        zero_max_counts[c] = {"zero": zero_ct, "max": max_ct}

        # Column eligibility (sufficient data) guard
        eligible = (len(s) >= min_nonnull_count) and (len(s) / max(1, len(col)) >= min_nonnull_ratio)

        # Flatline threshold-driven removal
        if eligible and is_flat and v is not None:
            m = np.isclose(col.astype(float), float(v), rtol=0, atol=1e-9)
            flatline_mask |= m
            flatline_values[c] = {
                "value": float(v),
                "reason": reason,
                "count": int(m.sum()),
            }

        # Optional zero/max thresholds
        if eligible and (drop_any_zero or (zero_threshold is not None and zero_ct / max(1, len(s)) >= zero_threshold)):
            flatline_mask |= np.isclose(col.astype(float), 0.0, rtol=0, atol=1e-9)
            if c not in flatline_values:
                flatline_values[c] = {"value": 0.0, "reason": "zero", "count": int(np.isclose(col.astype(float), 0.0, rtol=0, atol=1e-9).sum())}

        if eligible and len(s) > 0 and (drop_any_max or (max_threshold is not None and max_ct / max(1, len(s)) >= max_threshold)):
            max_value = float(s.max())
            flatline_mask |= np.isclose(col.astype(float), max_value, rtol=0, atol=1e-9)
            if c not in flatline_values:
                flatline_values[c] = {"value": max_value, "reason": "max", "count": int(np.isclose(col.astype(float), max_value, rtol=0, atol=1e-9).sum())}

    # Combine masks
    removed_mask = sigma_outlier_mask | flatline_mask
    removed_df = df.loc[removed_mask].copy()
    kept_df = df.loc[~removed_mask].copy()

    report = {
        "rows_total": int(len(df.index)),
        "rows_removed": int(removed_mask.sum()),
        "rows_kept": int((~removed_mask).sum()),
        "sigma_bounds": {c: {"low": b[0], "high": b[1]} for c, b in sigma_bounds.items()},
        "per_column_sigma_outliers": per_col_sigma_outliers,
        "flatline_values": flatline_values,
        "zero_max_counts": zero_max_counts,
        "flatline_threshold": flatline_threshold,
        "zero_threshold": zero_threshold,
        "max_threshold": max_threshold,
        "drop_any_zero": drop_any_zero,
        "drop_any_max": drop_any_max,
        "min_nonnull_ratio": min_nonnull_ratio,
        "min_nonnull_count": min_nonnull_count,
        "ignored_columns": sorted(ignore_set),
        "exclude_columns": sorted(EXCLUDE_COLS),
        "numeric_columns": num_cols,
    }

    return kept_df, removed_df, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean labels.csv by removing six-sigma and flatlined-value rows")
    p.add_argument("--input", default="eye_glucose_data/labels.csv", help="Path to input labels CSV")
    p.add_argument(
        "--output", default="eye_glucose_data/labels_clean.csv", help="Path to write cleaned CSV"
    )
    p.add_argument(
        "--removed-output",
        default="eye_glucose_data/labels_removed.csv",
        help="Path to write removed rows",
    )
    p.add_argument(
        "--report",
        default="eye_glucose_data/cleaning_report.json",
        help="Path to write JSON report",
    )
    p.add_argument(
        "--flatline-threshold",
        type=float,
        default=0.02,
        help="Proportion (0-1) at which a dominant value of 0 or max is considered a flatline and removed",
    )
    p.add_argument("--zero-threshold", type=float, default=None, help="If set, drop rows where a feature equals 0 when zeros >= this proportion in that column.")
    p.add_argument("--max-threshold", type=float, default=None, help="If set, drop rows where a feature equals column max when that max occurs >= this proportion in that column.")
    p.add_argument("--drop-any-zero", action="store_true", help="Drop any row that has a zero in any numeric feature (strong).")
    p.add_argument("--drop-any-max", action="store_true", help="Drop any row that has a max value in any numeric feature (strong).")
    p.add_argument("--min-nonnull-ratio", type=float, default=0.2, help="Minimum non-null ratio (0-1) a column must have to apply zero/max removal rules.")
    p.add_argument("--min-nonnull-count", type=int, default=100, help="Minimum non-null count a column must have to apply zero/max removal rules.")
    p.add_argument("--ignore-cols", type=str, default="", help="Comma-separated list of columns to ignore for outlier and flatline checks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    ignore_cols = [c for c in (args.ignore_cols.split(",") if args.ignore_cols else []) if c.strip()]
    kept_df, removed_df, report = clean_dataframe(
        df,
        flatline_threshold=args.flatline_threshold,
        zero_threshold=args.zero_threshold,
        max_threshold=args.max_threshold,
        drop_any_zero=args.drop_any_zero,
        drop_any_max=args.drop_any_max,
        min_nonnull_ratio=args.min_nonnull_ratio,
        min_nonnull_count=args.min_nonnull_count,
        ignore_columns=ignore_cols,
    )

    # Ensure output dirs exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.removed_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    kept_df.to_csv(args.output, index=False)
    removed_df.to_csv(args.removed_output, index=False)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"Cleaned CSV written to {args.output} | kept {report['rows_kept']}/{report['rows_total']} rows; "
        f"removed {report['rows_removed']} rows."
    )
    # Quick console summary of flatlines
    if report["flatline_values"]:
        print("Detected flatlined values:")
        for c, info in report["flatline_values"].items():
            print(f"  - {c}: {info['reason']}={info['value']} (rows: {info['count']})")


if __name__ == "__main__":
    main()
