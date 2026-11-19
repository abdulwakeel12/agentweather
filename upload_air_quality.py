import hashlib
import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv

from prepare_data import prepare_source

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def main():
    # Prepare and validate locally before connecting.
    df = prepare_source("air_quality")

    if df.isna().any().any():
        print("This first upload expects complete air-quality records.")
        print("Missing values detected. Nothing uploaded.")
        raise SystemExit(1)

    # Convert pandas nullable types to ordinary types for this complete batch.
    numeric_columns = [
        "requested_latitude",
        "requested_longitude",
        "grid_latitude",
        "grid_longitude",
        "us_aqi",
        "pm2_5_ug_m3",
        "pm10_ug_m3",
        "uv_index",
    ]

    for column in numeric_columns:
        df[column] = df[column].astype("float64")

    for column in ("source", "data_type"):
        df[column] = df[column].astype(str)

    # Stable identifiers allow the same records to be uploaded again
    # without creating separate observations.
    record_ids = []

    for _, row in df.iterrows():
        identity = (
            f"{row['source']}|"
            f"{row['grid_latitude']:.6f}|"
            f"{row['grid_longitude']:.6f}|"
            f"{row['valid_time_utc'].isoformat()}"
        )
        record_ids.append(
            hashlib.sha256(identity.encode("utf-8")).hexdigest()
        )

    df.insert(0, "record_id", record_ids)

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

    print(f"Prepared {len(df)} air-quality records.")
    print("Connecting to Hopsworks...")

    project = hopsworks.login(
        host=settings["HOPSWORKS_HOST"],
        port=443,
        project=settings["HOPSWORKS_PROJECT"],
        api_key_value=settings["HOPSWORKS_API_KEY"],
        engine="python",
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="karachi_air_quality_estimates",
        version=1,
        description=(
            "Karachi air-quality and UV estimates from Open-Meteo "
            "CAMS Global. Modelled estimates, not station measurements. "
            "AQI uses the US scale."
        ),
        primary_key=["record_id"],
        event_time="valid_time_utc",
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config={"enabled": False},
    )

    # Avoid silently using an existing group with different settings.
    if fg.online_enabled:
        print("This feature group has online storage enabled. Stopping.")
        raise SystemExit(1)

    print("Uploading records and waiting for ingestion...")
    fg.insert(df, wait=True)

    print("\nThe upload call completed.")
    print("Feature Group: karachi_air_quality_estimates")
    print("Version: 1")
    print(f"Records submitted: {len(df)}")
    print("Check the Feature Group in Hopsworks to verify stored data.")


if __name__ == "__main__":
    main()