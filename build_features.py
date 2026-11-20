from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "historical_hourly_2025-11_2026-10.csv"
)
OUTPUT_DIR = BASE_DIR / "data" / "features"

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


def main():
    try:
        df = pd.read_csv(INPUT_PATH)
        df["hour_utc"] = pd.to_datetime(
            df["hour_utc"], utc=True, errors="raise"
        )
        df = df.sort_values("hour_utc").set_index("hour_utc")

        expected = pd.date_range(
            "2025-11-01",
            "2026-11-01",
            inclusive="left",
            freq="h",
            tz="UTC",
        )

        if not df.index.equals(expected):
            raise ValueError("Unexpected, duplicate or missing timestamps.")

        for column in [*METRICS, "wind_direction_deg"]:
            df[column] = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(
            df[[*METRICS, "wind_direction_deg"]].to_numpy(dtype=float)
        ).all():
            raise ValueError("Missing or non-finite input values.")

        features = {}

        for column in METRICS:
            values = df[column]

            # Value available at the feature timestamp.
            features[f"{column}_current"] = values

            for lag in LAGS:
                features[f"{column}_lag_{lag}h"] = values.shift(lag)

            # These windows use only earlier hours, excluding the current hour.
            past = values.shift(1)

            for window in WINDOWS:
                rolling = past.rolling(window, min_periods=window)
                features[f"{column}_mean_previous_{window}h"] = (
                    rolling.mean()
                )
                features[f"{column}_std_previous_{window}h"] = (
                    rolling.std(ddof=0)
                )

            features[f"{column}_change_1h"] = values.diff()

        # Direction is circular: 359° and 1° are close together.
        radians = np.deg2rad(df["wind_direction_deg"] % 360)
        moving = df["wind_speed_ms"].gt(0)

        direction_components = {
            "wind_direction_sin": pd.Series(
                np.sin(radians), index=df.index
            ).where(moving, 0.0),
            "wind_direction_cos": pd.Series(
                np.cos(radians), index=df.index
            ).where(moving, 0.0),
        }

        for name, values in direction_components.items():
            features[f"{name}_current"] = values
            for lag in LAGS:
                features[f"{name}_lag_{lag}h"] = values.shift(lag)

        features["wind_calm_current"] = (~moving).astype(int)

        # Local time is appropriate for Karachi's daily cycle.
        local_time = df.index.tz_convert("Asia/Karachi")
        hour = local_time.hour
        weekday = local_time.dayofweek
        day_of_year = local_time.dayofyear - 1
        year_length = np.where(local_time.is_leap_year, 366, 365)

        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        features["year_sin"] = np.sin(
            2 * np.pi * day_of_year / year_length
        )
        features["year_cos"] = np.cos(
            2 * np.pi * day_of_year / year_length
        )

        result = pd.DataFrame(features, index=df.index)

        # The first seven days lack the required 168-hour history.
        warmup_hours = max(LAGS)
        result = result.iloc[warmup_hours:].copy()

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Unexpected missing or non-finite features.")

        validation_start = pd.Timestamp("2026-08-01", tz="UTC")
        test_start = pd.Timestamp("2026-10-01", tz="UTC")

        splits = {
            "train": result.loc[result.index < validation_start],
            "validation": result.loc[
                (result.index >= validation_start)
                & (result.index < test_start)
            ],
            "test": result.loc[result.index >= test_start],
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for name, frame in splits.items():
            path = OUTPUT_DIR / f"{name}_features.csv"
            temp_path = path.with_suffix(".tmp")
            frame.to_csv(temp_path, index_label="hour_utc")
            temp_path.replace(path)

            print(f"\n{name.upper()}")
            print(f"Rows: {len(frame)}")
            print(f"Features: {len(frame.columns)}")
            print(f"First hour: {frame.index.min()}")
            print(f"Last hour: {frame.index.max()}")

        print(f"\nWarm-up hours excluded: {warmup_hours}")
        print(f"Saved to: {OUTPUT_DIR}")
        print("All lag and rolling features use earlier timestamps.")
        print("No scaling, imputation or model fitting performed.")
        print("Targets have not been created yet.")
        print("Original data unchanged. No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Feature generation failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()