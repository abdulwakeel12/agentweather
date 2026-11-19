from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "historical"

START_DATE = "2026-10-01"
END_DATE = "2026-10-07"

VARIABLES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
}


def main():
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    print(f"Downloading Karachi weather: {START_DATE} to {END_DATE}")

    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        print("Historical weather request failed. Check your connection.")
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

        if df[list(VARIABLES.values())].isna().all().any():
            raise ValueError("An entire weather column is missing.")

        df["requested_latitude"] = params["latitude"]
        df["requested_longitude"] = params["longitude"]
        df["grid_latitude"] = data["latitude"]
        df["grid_longitude"] = data["longitude"]
        df["source"] = "open_meteo_era5"
        df["data_type"] = "reanalysis"
        df["retrieved_at_utc"] = datetime.now(UTC).isoformat()

    except (KeyError, ValueError, TypeError):
        print("Unexpected API data. No CSV was saved.")
        raise SystemExit(1) from None

    output_path = OUTPUT_DIR / (
        f"weather_era5_{START_DATE}_{END_DATE}.csv"
    )

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")
        df.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except OSError:
        print("Could not save the CSV. Check folder permissions.")
        raise SystemExit(1) from None

    print(f"\nRecords: {len(df)}")
    print(f"First hour: {df['hour_utc'].min()}")
    print(f"Last hour: {df['hour_utc'].max()}")
    print("\nMissing values:")
    print(df[list(VARIABLES.values())].isna().sum().to_string())
    print(f"\nSaved to: {output_path}")
    print("Source: Open-Meteo / ERA5 reanalysis")
    print("No data uploaded to Hopsworks.")


if __name__ == "__main__":
    main()