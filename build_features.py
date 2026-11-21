import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR / "data/historical/historical_hourly_2025-11_2026-10.csv"
)
OUTPUT_DIR = BASE_DIR / "data/features"

METRICS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "rain_mm",
    "wind_speed_ms",
    "us_aqi",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "uv_index",
]

LAGS = [1, 3, 6, 24, 48, 168]
WINDOWS = [3, 6, 24]


def read_history(path):
    frame = pd.read_csv(path)
    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"], utc=True, errors="raise"
    )
    return frame.set_index("hour_utc").sort_index()


def make_features(history):
    """Build features using observations at or before each row's timestamp."""
    frame = history.copy()

    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.tz is None
        or frame.index.has_duplicates
        or frame.index.hasnans
        or frame.empty
    ):
        raise ValueError("History needs unique, timezone-aware hourly timestamps.")

    frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    frame.index.name = "hour_utc"

    if not frame.index.equals(frame.index.floor("h")):
        raise ValueError("History timestamps must fall exactly on UTC hours.")

    expected = pd.date_range(
        frame.index.min(), frame.index.max(), freq="h", tz="UTC"
    )
    if not frame.index.equals(expected):
        raise ValueError("History contains missing hours.")

    required = [*METRICS, "wind_direction_deg"]
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite historical values.")

    if len(frame) < max(LAGS) + 1:
        raise ValueError("At least 169 consecutive hourly rows are required.")

    features = {}

    for column in METRICS:
        values = frame[column]
        features[f"{column}_current"] = values

        for lag in LAGS:
            features[f"{column}_lag_{lag}h"] = values.shift(lag)

        past = values.shift(1)
        for window in WINDOWS:
            rolling = past.rolling(window, min_periods=window)
            features[f"{column}_mean_previous_{window}h"] = rolling.mean()
            features[f"{column}_std_previous_{window}h"] = rolling.std(ddof=0)

        features[f"{column}_change_1h"] = values.diff()

    radians = np.deg2rad(frame["wind_direction_deg"] % 360)
    moving = frame["wind_speed_ms"].gt(0)

    direction_components = {
        "wind_direction_sin": pd.Series(
            np.sin(radians), index=frame.index
        ).where(moving, 0.0),
        "wind_direction_cos": pd.Series(
            np.cos(radians), index=frame.index
        ).where(moving, 0.0),
    }

    for name, values in direction_components.items():
        features[f"{name}_current"] = values
        for lag in LAGS:
            features[f"{name}_lag_{lag}h"] = values.shift(lag)

    features["wind_calm_current"] = (~moving).astype(int)

    local_time = frame.index.tz_convert("Asia/Karachi")
    hour = local_time.hour
    weekday = local_time.dayofweek
    day_of_year = local_time.dayofyear - 1
    year_length = np.where(local_time.is_leap_year, 366, 365)

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["year_sin"] = np.sin(2 * np.pi * day_of_year / year_length)
    features["year_cos"] = np.cos(2 * np.pi * day_of_year / year_length)

    result = pd.DataFrame(features, index=frame.index)
    result = result.iloc[max(LAGS):].copy()

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Invalid generated features.")

    return result


def verify_training_compatibility():
    """Check the refactored calculations against existing training features."""
    reference_path = OUTPUT_DIR / "train_features.csv"
    reference = read_history(reference_path)

    # Read only the training-period portion of the annual CSV.
    cutoff = pd.Timestamp("2026-08-01", tz="UTC")
    chunks = []

    for chunk in pd.read_csv(INPUT_PATH, chunksize=1000):
        chunk["hour_utc"] = pd.to_datetime(
            chunk["hour_utc"], utc=True, errors="raise"
        )
        chunks.append(chunk.loc[chunk["hour_utc"] < cutoff])

    history = (
        pd.concat(chunks, ignore_index=True)
        .set_index("hour_utc")
        .sort_index()
    )
    rebuilt = make_features(history)

    if not rebuilt.index.equals(reference.index):
        raise ValueError("Rebuilt training feature timestamps differ.")
    if list(rebuilt.columns) != list(reference.columns):
        raise ValueError("Rebuilt training feature names/order differ.")

    actual = rebuilt.to_numpy(dtype=float)
    expected = reference.to_numpy(dtype=float)

    if not np.isfinite(expected).all():
        raise ValueError("Existing training features contain invalid values.")

    if not np.allclose(actual, expected, atol=1e-8, rtol=1e-8):
        raise ValueError(
            "Rebuilt features differ from existing training features. "
            "No inference file was saved."
        )

    print(
        "Training feature compatibility: PASS "
        f"({len(rebuilt)} rows, {len(rebuilt.columns)} features)"
    )
    return list(reference.columns)


def build_inference(history_path, output_path):
    expected_columns = verify_training_compatibility()
    history = read_history(history_path)
    features = make_features(history)
    latest = features.tail(1)

    if list(latest.columns) != expected_columns:
        raise ValueError("Inference feature names/order differ from training.")

    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    latest.to_csv(temporary, index_label="hour_utc")
    temporary.replace(output_path)

    print("\nAgentWeather — Inference Features")
    print("Prediction time:", latest.index[0])
    print("Feature rows:", len(latest))
    print("Input features:", latest.shape[1])
    print("Missing values:", int(latest.isna().sum().sum()))
    print("Saved to:", output_path)
    print("Existing training and test feature files were not changed.")
    print("No model fitting or Hopsworks upload.")
    print("These inputs are historical, not live.")


def build_training_splits():
    history = read_history(INPUT_PATH)
    expected = pd.date_range(
        "2025-11-01", "2026-11-01",
        inclusive="left", freq="h", tz="UTC",
    )
    if not history.index.equals(expected):
        raise ValueError("Unexpected annual-history timestamps.")

    features = make_features(history)
    validation_start = pd.Timestamp("2026-08-01", tz="UTC")
    test_start = pd.Timestamp("2026-10-01", tz="UTC")

    splits = {
        "train": features.loc[features.index < validation_start],
        "validation": features.loc[
            (features.index >= validation_start)
            & (features.index < test_start)
        ],
        "test": features.loc[features.index >= test_start],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        path = OUTPUT_DIR / f"{name}_features.csv"
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index_label="hour_utc")
        temporary.replace(path)
        print(f"{name.upper()}: {len(frame)} rows, {frame.shape[1]} features")

    print("Feature splits saved. No model fitting or Hopsworks upload.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["training", "inference"], default="training"
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=BASE_DIR / "data/inference/history_through_2026-11-15.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/inference/features_2026-11-15T230000Z.csv",
    )
    args = parser.parse_args()

    if args.mode == "inference":
        build_inference(args.history, args.output)
    else:
        build_training_splits()


if __name__ == "__main__":
    main()