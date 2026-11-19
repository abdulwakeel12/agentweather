import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from build_historical_dataset import AIR_COLUMNS, BOUNDS, WEATHER_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
OUTPUT_PATH = DATA_DIR / "historical_hourly_2025-11_2026-10.csv"

METRICS = WEATHER_COLUMNS + AIR_COLUMNS

SOURCE_LABELS = {
    "weather_source": "open_meteo_era5",
    "weather_data_type": "reanalysis",
    "air_source": "open_meteo_cams_global",
    "air_data_type": "modelled_estimate",
}


def main():
    frames = []
    expected_columns = None

    try:
        # Explicit monthly filenames exclude the earlier seven-day sample.
        months = pd.period_range("2025-11", "2026-10", freq="M")

        for month in months:
            year = month.year
            number = month.month
            last_day = calendar.monthrange(year, number)[1]

            start = f"{year:04d}-{number:02d}-01"
            end = f"{year:04d}-{number:02d}-{last_day:02d}"

            path = DATA_DIR / f"historical_hourly_{start}_{end}.csv"

            if not path.is_file():
                raise ValueError(f"Missing monthly file: {path.name}")

            df = pd.read_csv(path)

            required = {"hour_utc", *METRICS, *SOURCE_LABELS}
            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{path.name}: missing columns {sorted(missing)}"
                )

            if expected_columns is None:
                expected_columns = list(df.columns)
            elif list(df.columns) != expected_columns:
                raise ValueError(f"{path.name}: column layout differs.")

            df["hour_utc"] = pd.to_datetime(
                df["hour_utc"], utc=True, errors="raise"
            )
            df = df.sort_values("hour_utc").reset_index(drop=True)

            expected_hours = pd.date_range(
                start=start,
                periods=last_day * 24,
                freq="h",
                tz="UTC",
            )

            if not pd.DatetimeIndex(df["hour_utc"]).equals(expected_hours):
                raise ValueError(
                    f"{path.name}: missing, duplicate or unexpected hours."
                )

            frames.append(df)
            print(f"Loaded {month}: {len(df)} rows")

        annual = (
            pd.concat(frames, ignore_index=True)
            .sort_values("hour_utc")
            .reset_index(drop=True)
        )

        expected_year = pd.date_range(
            start="2025-11-01",
            end="2026-11-01",
            inclusive="left",
            freq="h",
            tz="UTC",
        )

        if not pd.DatetimeIndex(annual["hour_utc"]).equals(expected_year):
            raise ValueError("Annual timestamps do not match the full year.")

        for column, expected in SOURCE_LABELS.items():
            if not annual[column].eq(expected).all():
                raise ValueError(f"Unexpected source label in {column}.")

        for column in METRICS:
            values = pd.to_numeric(annual[column], errors="raise")
            annual[column] = values

            if values.isna().any():
                raise ValueError(f"Missing values in {column}.")

            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite values in {column}.")

            lower, upper = BOUNDS[column]

            if lower is not None and values.lt(lower).any():
                raise ValueError(f"{column} contains values below {lower}.")

            if upper is not None and values.gt(upper).any():
                raise ValueError(f"{column} contains values above {upper}.")

        temp_path = OUTPUT_PATH.with_suffix(".tmp")
        annual.to_csv(temp_path, index=False)
        temp_path.replace(OUTPUT_PATH)

    except (ValueError, KeyError, TypeError) as error:
        print(f"\nValidation failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("\nFile access failed. Check paths and close open CSV files.")
        raise SystemExit(1) from None

    print("\nAgentWeather — Annual Historical Dataset")
    print(f"Monthly files combined: {len(frames)}")
    print(f"Total rows: {len(annual)}")
    print(f"First hour: {annual['hour_utc'].min()}")
    print(f"Last hour: {annual['hour_utc'].max()}")
    print(f"Duplicate hours: {annual['hour_utc'].duplicated().sum()}")
    print("Missing hours: 0")

    print("\nMissing metric values:")
    print(annual[METRICS].isna().sum().to_string())

    print("\nValue ranges:")
    print(annual[METRICS].agg(["min", "max"]).T.round(3).to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Monthly source files preserved.")
    print("No data uploaded to Hopsworks.")


if __name__ == "__main__":
    main()