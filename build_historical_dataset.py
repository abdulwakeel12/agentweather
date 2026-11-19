from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"

START_DATE = "2026-10-01"
END_DATE = "2026-10-07"

WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "rain_mm",
    "wind_speed_ms",
    "wind_direction_deg",
]

AIR_COLUMNS = [
    "us_aqi",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "uv_index",
]

# Basic validity checks, not hazard-alert thresholds.
BOUNDS = {
    "temperature_c": (None, None),
    "humidity_pct": (0, 100),
    "precipitation_mm": (0, None),
    "rain_mm": (0, None),
    "wind_speed_ms": (0, None),
    "wind_direction_deg": (0, 360),
    "us_aqi": (0, None),
    "pm2_5_ug_m3": (0, None),
    "pm10_ug_m3": (0, None),
    "uv_index": (0, None),
}


def load_source(path, columns, expected_source, expected_type):
    df = pd.read_csv(path)

    required = {
        "hour_utc",
        "source",
        "data_type",
        *columns,
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns: {', '.join(sorted(missing))}"
        )

    df["hour_utc"] = pd.to_datetime(
        df["hour_utc"], utc=True, errors="raise"
    )

    if df["hour_utc"].isna().any():
        raise ValueError(f"{path.name}: missing timestamps.")

    if df["hour_utc"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate timestamps.")

    df = df.sort_values("hour_utc").reset_index(drop=True)

    expected_hours = pd.date_range(
        start=START_DATE,
        end=pd.Timestamp(END_DATE) + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
        tz="UTC",
    )

    if not pd.DatetimeIndex(df["hour_utc"]).equals(expected_hours):
        raise ValueError(f"{path.name}: unexpected dates or missing hours.")

    if not df["source"].eq(expected_source).all():
        raise ValueError(f"{path.name}: unexpected source label.")

    if not df["data_type"].eq(expected_type).all():
        raise ValueError(f"{path.name}: unexpected data-type label.")

    for column in columns:
        values = pd.to_numeric(df[column], errors="raise")
        df[column] = values

        if values.isna().any():
            raise ValueError(f"{path.name}: missing values in {column}.")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: non-finite values in {column}.")

        lower, upper = BOUNDS[column]

        if lower is not None and values.lt(lower).any():
            raise ValueError(f"{path.name}: {column} is below {lower}.")

        if upper is not None and values.gt(upper).any():
            raise ValueError(f"{path.name}: {column} is above {upper}.")

    return df


def main():
    weather_path = DATA_DIR / (
        f"weather_era5_{START_DATE}_{END_DATE}.csv"
    )
    air_path = DATA_DIR / (
        f"air_quality_cams_global_{START_DATE}_{END_DATE}.csv"
    )
    output_path = DATA_DIR / (
        f"historical_hourly_{START_DATE}_{END_DATE}.csv"
    )

    try:
        weather = load_source(
            weather_path,
            WEATHER_COLUMNS,
            "open_meteo_era5",
            "reanalysis",
        )
        air = load_source(
            air_path,
            AIR_COLUMNS,
            "open_meteo_cams_global",
            "modelled_estimate",
        )

        # Prefix metadata so both sources remain identifiable.
        weather = weather.rename(
            columns={
                column: f"weather_{column}"
                for column in weather.columns
                if column not in ["hour_utc", *WEATHER_COLUMNS]
            }
        )
        air = air.rename(
            columns={
                column: f"air_{column}"
                for column in air.columns
                if column not in ["hour_utc", *AIR_COLUMNS]
            }
        )

        combined = weather.merge(
            air,
            on="hour_utc",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not combined["_merge"].eq("both").all():
            raise ValueError("Some hours do not have both sources.")

        combined = (
            combined.drop(columns="_merge")
            .sort_values("hour_utc")
            .reset_index(drop=True)
        )

        metrics = WEATHER_COLUMNS + AIR_COLUMNS

        print("\nAgentWeather — Historical Dataset")
        print(f"Weather rows: {len(weather)}")
        print(f"Air-quality rows: {len(air)}")
        print(f"Matched hourly rows: {len(combined)}")
        print(f"First hour: {combined['hour_utc'].min()}")
        print(f"Last hour: {combined['hour_utc'].max()}")

        print("\nValue ranges:")
        summary = combined[metrics].agg(["min", "max"]).T
        print(summary.round(3).to_string())

        constant_columns = [
            column
            for column in metrics
            if combined[column].nunique() == 1
        ]
        if constant_columns:
            print(
                "\nConstant columns in this sample: "
                + ", ".join(constant_columns)
            )
            print("Review these during EDA; a constant week is not an error.")

        temp_path = output_path.with_suffix(".tmp")
        combined.to_csv(temp_path, index=False)
        temp_path.replace(output_path)

    except (ValueError, KeyError, TypeError) as error:
        print(f"Validation failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print(
            "File access failed. Check that both historical CSVs exist "
            "and close the output CSV if it is open in Excel."
        )
        raise SystemExit(1) from None

    print(f"\nSaved to: {output_path}")
    print("Timestamp and basic numeric checks passed.")
    print("Source CSVs were preserved. No Hopsworks upload.")


if __name__ == "__main__":
    main()