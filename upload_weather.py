import hashlib
import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv
from hsfs.feature import Feature

from prepare_data import prepare_source

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "rain_api_1h",
    "snow_api_1h",
]

BOOLEAN_COLUMNS = [
    "rain_field_supplied",
    "snow_field_supplied",
]

TIMESTAMP_COLUMNS = [
    "observed_at_utc",
    "retrieved_at_utc",
]


def make_record_id(row):
    identity = (
        f"{row['source']}|"
        f"{row['latitude']:.6f}|"
        f"{row['longitude']:.6f}|"
        f"{row['observed_at_utc'].isoformat()}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main():
    # Validate and prepare locally before connecting.
    df = prepare_source("weather")

    df.insert(0, "record_id", df.apply(make_record_id, axis=1))

    # Keep numeric missing values as NaN, never replace them with zero.
    for column in NUMERIC_COLUMNS:
        df[column] = df[column].astype("float64")

    # Preserve missing flags as None, rather than False.
    for column in BOOLEAN_COLUMNS:
        values = df[column]
        df[column] = values.astype(object).where(values.notna(), None)

    for column in ("conditions", "source"):
        values = df[column]
        df[column] = values.astype(object).where(values.notna(), None)

    # Define all feature types, including entirely empty rain/snow columns.
    features = []

    for column in df.columns:
        if column in NUMERIC_COLUMNS:
            feature_type = "double"
        elif column in BOOLEAN_COLUMNS:
            feature_type = "boolean"
        elif column in TIMESTAMP_COLUMNS:
            feature_type = "timestamp"
        else:
            feature_type = "string"

        features.append(Feature(column, type=feature_type))

    required = (
        "HOPSWORKS_API_KEY",
        "HOPSWORKS_PROJECT",
        "HOPSWORKS_HOST",
    )

    settings = {
        name: os.getenv(name, "").strip()
        for name in required
    }

    missing = [name for name, value in settings.items() if not value]

    if missing:
        print("Missing .env settings:", ", ".join(missing))
        raise SystemExit(1)

    print(f"\nPrepared {len(df)} weather records.")

    missing_counts = df.isna().sum()
    print("\nMissing values being preserved:")
    print(missing_counts[missing_counts > 0].to_string())

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        host=settings["HOPSWORKS_HOST"],
        port=443,
        project=settings["HOPSWORKS_PROJECT"],
        api_key_value=settings["HOPSWORKS_API_KEY"],
        engine="python",
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="karachi_weather_observations",
        version=1,
        description=(
            "Karachi weather records from OpenWeather. "
            "Original timestamps and missing values preserved. "
            "rain_api_1h and snow_api_1h retain the API values; "
            "supplied flags distinguish absent fields."
        ),
        primary_key=["record_id"],
        event_time="observed_at_utc",
        features=features,
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config={"enabled": False},
    )

    if fg.online_enabled:
        print("This group has online storage enabled. Stopping.")
        raise SystemExit(1)

    print("\nUploading weather records and waiting for ingestion...")
    fg.insert(df, wait=True)

    print("\nThe weather upload call completed.")
    print("Feature Group: karachi_weather_observations")
    print("Version: 1")
    print(f"Records submitted: {len(df)}")
    print("Next, verify the stored values and missing-value counts.")


if __name__ == "__main__":
    main()