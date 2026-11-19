from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CONFIGS = {
    "weather": {
        "file": "weather_observations.csv",
        "event_time": "observed_at_utc",
        "keys": ["source", "latitude", "longitude", "observed_at_utc"],
        "numbers": [
            "latitude",
            "longitude",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "wind_direction_deg",
            "rain_api_1h",
            "snow_api_1h",
        ],
        "flags": ["rain_field_supplied", "snow_field_supplied"],
        "source": "openweather",
    },
    "air_quality": {
        "file": "air_quality_estimates.csv",
        "event_time": "valid_time_utc",
        "keys": [
            "source",
            "grid_latitude",
            "grid_longitude",
            "valid_time_utc",
        ],
        "numbers": [
            "requested_latitude",
            "requested_longitude",
            "grid_latitude",
            "grid_longitude",
            "us_aqi",
            "pm2_5_ug_m3",
            "pm10_ug_m3",
            "uv_index",
        ],
        "flags": [],
        "source": "open_meteo_cams_global",
    },
}


def prepare_source(name):
    config = CONFIGS[name]
    path = DATA_DIR / config["file"]

    # Read as text first, then explicitly convert each column.
    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig")

    required = {
        *config["numbers"],
        *config["flags"],
        *config["keys"],
        "retrieved_at_utc",
    }

    if name == "air_quality":
        required.add("data_type")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{name}: no records found")

    for column in df.columns:
        df[column] = df[column].str.strip().replace("", pd.NA)

    for column in (config["event_time"], "retrieved_at_utc"):
        df[column] = pd.to_datetime(
            df[column],
            format="ISO8601",
            utc=True,
            errors="raise",
        )

    for column in config["numbers"]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        ).astype("Float64")

    for column in config["flags"]:
        values = df[column].str.lower()

        if not values.dropna().isin(["true", "false"]).all():
            raise ValueError(f"{name}: invalid boolean in {column}")

        df[column] = values.map(
            {"true": True, "false": False}
        ).astype("boolean")

    if df[config["keys"]].isna().any().any():
        raise ValueError(f"{name}: a record identifier is missing")

    if not df["source"].eq(config["source"]).all():
        raise ValueError(f"{name}: unexpected source label")

    if name == "air_quality":
        labels = df["data_type"]
        if labels.isna().any() or not labels.eq("modelled_estimate").all():
            raise ValueError("Air quality must be labelled modelled_estimate")

    if df.duplicated(subset=config["keys"]).any():
        raise ValueError(f"{name}: duplicate record identifiers")

    return df.sort_values(config["event_time"]).reset_index(drop=True)


def main():
    try:
        # Validate both sources before reporting success.
        prepared = {
            name: prepare_source(name)
            for name in CONFIGS
        }

        for name, df in prepared.items():
            event_time = CONFIGS[name]["event_time"]

            print(f"\n{name.upper()}")
            print(f"Records: {len(df)}")
            print(f"First timestamp: {df[event_time].min()}")
            print(f"Latest timestamp: {df[event_time].max()}")

            print("\nColumn types:")
            print(df.dtypes.to_string())

            missing = df.isna().sum()
            missing = missing[missing > 0]

            print("\nMissing values:")
            print(missing.to_string() if not missing.empty else "None")

        print("\nLocal preparation passed.")
        print("Missing values were preserved, not replaced with zeros.")
        print("No files changed and no data uploaded to Hopsworks.")

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"\nPreparation failed: {error}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()