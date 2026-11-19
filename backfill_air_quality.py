from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "historical"

START_DATE = "2026-10-01"
END_DATE = "2026-10-07"

VARIABLES = {
    "us_aqi": "us_aqi",
    "pm2_5": "pm2_5_ug_m3",
    "pm10": "pm10_ug_m3",
    "uv_index": "uv_index",
}


def main():
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "domains": "cams_global",
        "timezone": "UTC",
    }

    print(
        f"Downloading Karachi modelled air quality: "
        f"{START_DATE} to {END_DATE}"
    )

    try:
        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params=params,
            timeout=120,
        )

        if response.status_code != 200:
            print(f"Request failed: HTTP {response.status_code}")
            try:
                print(response.json().get("reason", "No details supplied."))
            except ValueError:
                pass
            raise SystemExit(1)

        data = response.json()

    except requests.exceptions.RequestException:
        print("Connection failed. Check your internet and retry.")
        raise SystemExit(1) from None
    except ValueError:
        print("The API returned invalid JSON.")
        raise SystemExit(1) from None

    try:
        hourly = data["hourly"]

        df = pd.DataFrame(
            {
                "hour_utc": hourly["time"],
                **{
                    column: hourly[variable]
                    for variable, column in VARIABLES.items()
                },
            }
        )

        df["hour_utc"] = pd.to_datetime(
            df["hour_utc"], utc=True, errors="raise"
        )

        expected_hours = pd.date_range(
            start=START_DATE,
            end=pd.Timestamp(END_DATE) + pd.Timedelta(days=1),
            freq="h",
            inclusive="left",
            tz="UTC",
        )

        if not pd.DatetimeIndex(df["hour_utc"]).equals(expected_hours):
            raise ValueError("Unexpected timestamps, gaps or duplicates.")

        for column in VARIABLES.values():
            df[column] = pd.to_numeric(df[column], errors="raise")

        print(f"\nRecords: {len(df)}")
        print(f"First hour: {df['hour_utc'].min()}")
        print(f"Last hour: {df['hour_utc'].max()}")
        print("\nMissing values:")
        print(df[list(VARIABLES.values())].isna().sum().to_string())

        empty_columns = [
            column
            for column in VARIABLES.values()
            if df[column].isna().all()
        ]

        if empty_columns:
            print("\nNo values returned for: " + ", ".join(empty_columns))
            print("Nothing saved. We need to investigate availability.")
            raise SystemExit(1)

        df["requested_latitude"] = params["latitude"]
        df["requested_longitude"] = params["longitude"]
        df["grid_latitude"] = data["latitude"]
        df["grid_longitude"] = data["longitude"]
        df["source"] = "open_meteo_cams_global"
        df["data_type"] = "modelled_estimate"
        df["retrieved_at_utc"] = datetime.now(UTC).isoformat()

    except (KeyError, ValueError, TypeError):
        print("Unexpected API data. No CSV was saved.")
        raise SystemExit(1) from None

    output_path = OUTPUT_DIR / (
        f"air_quality_cams_global_{START_DATE}_{END_DATE}.csv"
    )

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")
        df.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except OSError:
        print("Could not save the CSV. Check folder permissions.")
        raise SystemExit(1) from None

    print(f"\nSaved to: {output_path}")
    print("Source: Open-Meteo / CAMS Global")
    print("Data type: Archived model estimates, not station measurements.")
    print("No data uploaded to Hopsworks.")


if __name__ == "__main__":
    main()