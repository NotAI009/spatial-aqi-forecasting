"""Data loading, preprocessing, and spatial frame utilities for the AQI forecasting pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed 20-city layout used to transform each month into a map-like AQI frame.
# The arrangement is intentionally stable for reproducibility and roughly follows
# west-to-east/north-to-south regional grouping for the available cities.
CITY_GRID_LAYOUT = (
    ("Jaipur", "Delhi", "Ghaziabad", "Noida", "Lucknow"),
    ("Ahmedabad", "Gurugram", "Faridabad", "Agra", "Kanpur"),
    ("Mumbai", "Pune", "Hyderabad", "Varanasi", "Patna"),
    ("Thiruvananthapuram", "Bengaluru", "Chennai", "Visakhapatnam", "Kolkata"),
)


def get_city_order(layout=CITY_GRID_LAYOUT) -> list[str]:
    return [city for row in layout for city in row if city is not None]


def _clean_wide_monthly_data(
    df: pd.DataFrame,
    interpolation_method: str = "time",
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df.index = df.index.to_period("M").to_timestamp(how="start")
    df = df.groupby(df.index).mean(numeric_only=True).sort_index()

    ordered_cities = get_city_order()
    extra_cities = [city for city in df.columns if city not in ordered_cities]
    present_cities = [city for city in ordered_cities if city in df.columns]
    if not present_cities:
        raise ValueError("No expected AQI city columns were found in the dataset.")

    df = df[present_cities].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(full_index)

    missing_before = df.isna().sum()
    if interpolation_method == "time":
        df = df.interpolate(method="time", limit_direction="both")
    else:
        df = df.interpolate(method=interpolation_method, limit_direction="both")
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    missing_after = df.isna().sum()

    df = df.astype(np.float32)
    report = {
        "rows": int(df.shape[0]),
        "cities": present_cities,
        "extra_cities_ignored": extra_cities,
        "missing_before": missing_before.to_dict(),
        "missing_after": missing_after.to_dict(),
        "start_month": df.index.min().strftime("%Y-%m"),
        "end_month": df.index.max().strftime("%Y-%m"),
    }
    return df, report


def load_long_aqi_csv(
    csv_file: str | Path,
    date_column: str = "Date",
    city_column: str = "City",
    value_column: str = "AQI",
    aggregation: str = "mean",
    interpolation_method: str = "time",
    return_report: bool = False,
):
    """
    Load daily long-format AQI data and convert it to monthly wide city data.

    Expected input columns:
      Date, City, AQI

    Output:
      DataFrame indexed by month-start timestamps, columns are city names, values
      are monthly aggregate AQI. Missing monthly values are interpolated.
    """
    raw_df = pd.read_csv(csv_file, encoding="utf-8-sig")
    required = {date_column, city_column, value_column}
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    raw_df = raw_df[[date_column, city_column, value_column]].copy()
    raw_df[date_column] = pd.to_datetime(raw_df[date_column], errors="coerce")
    raw_df[city_column] = raw_df[city_column].astype(str).str.strip()
    raw_df[value_column] = pd.to_numeric(raw_df[value_column], errors="coerce")
    raw_df = raw_df.dropna(subset=[date_column, city_column])
    raw_df["Month"] = raw_df[date_column].dt.to_period("M").dt.to_timestamp(how="start")

    daily_rows = int(len(raw_df))
    missing_daily_aqi = int(raw_df[value_column].isna().sum())
    city_counts = raw_df.groupby(city_column)[value_column].count().to_dict()

    monthly = raw_df.pivot_table(
        index="Month",
        columns=city_column,
        values=value_column,
        aggfunc=aggregation,
    )
    monthly, report = _clean_wide_monthly_data(
        monthly,
        interpolation_method=interpolation_method,
    )
    report.update(
        {
            "source_file": str(csv_file),
            "source_format": "daily_long_csv",
            "daily_rows": daily_rows,
            "missing_daily_aqi": missing_daily_aqi,
            "monthly_aggregation": aggregation,
            "raw_city_observation_counts": city_counts,
        }
    )

    if return_report:
        return monthly, report
    return monthly


def load_city_data(
    data_file: str | Path,
    date_column: str = "Month & Year/States",
    interpolation_method: str = "time",
    return_report: bool = False,
):
    """
    Load AQI data from either the new long CSV or the legacy wide Excel file.

    New preferred format:
      aqi_20cities_long.csv with columns Date, City, AQI.

    Legacy format:
      Excel sheet with a month/date column and one column per city.
    """
    data_file = Path(data_file)
    if data_file.suffix.lower() == ".csv":
        return load_long_aqi_csv(
            data_file,
            interpolation_method=interpolation_method,
            return_report=return_report,
        )

    raw_df = pd.read_excel(data_file)
    if date_column not in raw_df.columns:
        raise ValueError(
            f"Date column '{date_column}' not found. Available columns: {list(raw_df.columns)}"
        )
    raw_df[date_column] = pd.to_datetime(raw_df[date_column], errors="coerce")
    raw_df = raw_df.dropna(subset=[date_column]).set_index(date_column).sort_index()
    df, report = _clean_wide_monthly_data(
        raw_df,
        interpolation_method=interpolation_method,
    )
    report.update({"source_file": str(data_file), "source_format": "legacy_wide_excel"})

    if return_report:
        return df, report
    return df


def build_grid(df: pd.DataFrame, layout=CITY_GRID_LAYOUT, fill_value: float = 0.0):
    """
    Convert monthly city AQI data to frames shaped (time, rows, cols, channels).
    """
    rows = len(layout)
    cols = len(layout[0])
    timesteps = len(df)

    grid = np.full((timesteps, rows, cols, 1), fill_value, dtype=np.float32)
    valid_mask = np.zeros((rows, cols), dtype=bool)
    city_positions: dict[str, tuple[int, int]] = {}

    for r, row in enumerate(layout):
        for c, city in enumerate(row):
            if city is not None and city in df.columns:
                grid[:, r, c, 0] = df[city].to_numpy(dtype=np.float32)
                valid_mask[r, c] = True
                city_positions[city] = (r, c)

    return grid, valid_mask, city_positions


def create_sequences(frames: np.ndarray, seq_len: int):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if len(frames) <= seq_len:
        raise ValueError(
            f"Need more than {seq_len} monthly frames to create sequences; got {len(frames)}."
        )

    X, y = [], []
    for i in range(len(frames) - seq_len):
        X.append(frames[i : i + seq_len])
        y.append(frames[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Time-ordered split for forecasting. No shuffling is used.
    """
    if len(X) < 10:
        raise ValueError("Not enough sequences for train/val/test split.")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Split ratios must be within [0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= len(X):
        raise ValueError("Invalid split ratios for current data size.")

    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def train_val_split(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def per_city_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics (mean, std, min, max, count) for each city.
    Useful for reporting in the methods section of a paper.
    """
    stats = df.describe().T[["count", "mean", "std", "min", "max"]]
    stats.index.name = "City"
    return stats.round(2)


def load_and_describe(data_file, **kwargs) -> None:
    """
    Load the dataset and print a compact quality report to stdout.
    """
    df, report = load_city_data(data_file, return_report=True, **kwargs)
    print(f"Source : {report.get('source_file')}")
    print(f"Format : {report.get('source_format')}")
    print(f"Range  : {report.get('start_month')} → {report.get('end_month')}")
    print(f"Months : {df.shape[0]}  |  Cities: {df.shape[1]}")
    print(f"Daily rows processed : {report.get('daily_rows', 'N/A'):,}")
    print(f"Missing daily AQI    : {report.get('missing_daily_aqi', 'N/A'):,}")
    print(f"Missing monthly (before interpolation): {sum(report.get('missing_before', {}).values())}")
    print(f"Missing monthly (after  interpolation): {sum(report.get('missing_after',  {}).values())}")
    print("\nPer-city statistics (monthly AQI):")
    print(per_city_stats(df).to_string())
