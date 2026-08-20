"""
Standalone preprocessing preview for the 20-city long AQI dataset.

Loads the daily long-format CSV, aggregates to monthly city AQI, builds
the 4×5 spatial frame tensor, creates supervised sequences, and prints a
full data-quality report. No model training — use experiments.py for that.

Usage
-----
    python csv_preprocessing.py
    python csv_preprocessing.py --data-file aqi_20cities_long.csv --seq-len 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_utils import (
    CITY_GRID_LAYOUT,
    build_grid,
    create_sequences,
    load_city_data,
    per_city_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview monthly AQI preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-file", default="aqi_20cities_long.csv",
        help="Long CSV with columns: Date, City, AQI",
    )
    parser.add_argument(
        "--seq-len", type=int, default=6,
        help="Input sequence length (months of history per training sample)",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for preview CSV output",
    )
    return parser.parse_args()


def _sep(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n {title}\n{line}")
    else:
        print(line)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _sep("1. Load & preprocess")
    df, report = load_city_data(args.data_file, return_report=True)

    print(f"  Source      : {report['source_file']}")
    print(f"  Format      : {report['source_format']}")
    print(f"  Date range  : {report['start_month']} → {report['end_month']}")
    print(f"  Daily rows  : {report.get('daily_rows', 'N/A'):,}")
    print(f"  Missing AQI : {report.get('missing_daily_aqi', 'N/A'):,} daily values")
    print(f"  Cities      : {len(report['cities'])} — {', '.join(report['cities'])}")
    print(f"  Monthly shape: {df.shape}  (months × cities)")
    mb = sum(report.get("missing_before", {}).values())
    ma = sum(report.get("missing_after", {}).values())
    print(f"  Missing monthly → before interpolation: {mb}  |  after: {ma}")

    _sep("2. Per-city statistics (monthly AQI)")
    print(per_city_stats(df).to_string())

    _sep("3. Spatial frame construction")
    frames, valid_mask, city_positions = build_grid(df, layout=CITY_GRID_LAYOUT)
    print(f"  Frame tensor shape : {frames.shape}  (months, H=4, W=5, C=1)")
    print(f"  Valid city cells   : {int(valid_mask.sum())} / {valid_mask.size}")
    print(f"  Grid layout (4×5)  :")
    for i, row in enumerate(CITY_GRID_LAYOUT):
        print(f"    Row {i}: {list(row)}")

    _sep("4. Sequence preparation")
    X, y = create_sequences(frames, seq_len=args.seq_len)
    print(f"  Input window   : {args.seq_len} months")
    print(f"  X shape        : {X.shape}  (samples, seq_len, H, W, C)")
    print(f"  y shape        : {y.shape}  (samples, H, W, C)")
    print(f"  Total sequences: {len(X)}")

    # Quick sanity check on value range
    aqi_min = float(df.min().min())
    aqi_max = float(df.max().max())
    print(f"  AQI range in dataset: [{aqi_min:.1f}, {aqi_max:.1f}]")

    _sep("5. Saving preview CSV")
    out_path = output_dir / "monthly_interpolated_city_aqi.csv"
    df.to_csv(out_path)
    print(f"  Saved: {out_path}")

    report_path = output_dir / "preprocessing_report.json"
    report_out = {k: v for k, v in report.items() if k != "grid_layout"}
    report_out["grid_layout"] = [list(row) for row in CITY_GRID_LAYOUT]
    report_path.write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    print(f"  Saved: {report_path}")

    _sep("Done — run experiments.py to train and evaluate the ConvLSTM model.")


if __name__ == "__main__":
    main()
