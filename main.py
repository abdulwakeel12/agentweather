import csv
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def save_observation(record):
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "weather_observations.csv"
    fields = list(record)
    rows = []
    old_fields = fields

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            old_fields = reader.fieldnames or []
            rows = list(reader)

        if (
            "observed_at_utc" not in old_fields
            or not set(old_fields).issubset(fields)
        ):
            print("Unexpected CSV columns. Existing file left unchanged.")
            raise SystemExit(1)

    schema_changed = old_fields != fields

    if schema_changed:
        backup_path = csv_path.with_name(
            f"weather_observations_backup_"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.csv"
        )
        shutil.copy2(csv_path, backup_path)
        print(f"Backup saved to: {backup_path}")

    existing = next(
        (
            row
            for row in rows
            if row["observed_at_utc"] == record["observed_at_utc"]
        ),
        None,
    )

    if existing is not None:
        # Add newly introduced fields only when they were not collected.
        updated = False
        new_fields = (
            "retrieved_at_utc",
            "rain_api_1h",
            "snow_api_1h",
            "rain_field_supplied",
            "snow_field_supplied",
        )

        for field in new_fields:
            if existing.get(field, "") == "":
                value = record.get(field)
                if value is not None:
                    existing[field] = value
                    updated = True

        if not updated and not schema_changed:
            print("\nThis observation is already saved.")
            return
    else:
        rows.append(record)

    # Write to a temporary file before replacing the original.
    temp_path = csv_path.with_suffix(".tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(csv_path)
    print(f"\nWeather CSV saved to: {csv_path}")


def main():
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    if not api_key:
        print("Missing API key. Check your .env file.")
        raise SystemExit(1)

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": 24.8607,
        "lon": 67.0011,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException:
        # Do not print the request URL because it contains the API key.
        print("Connection failed. Check your internet connection and retry.")
        raise SystemExit(1) from None

    if response.status_code == 401:
        print("Key not accepted. Check the key or allow time for activation.")
        raise SystemExit(1)

    if response.status_code != 200:
        print(f"Weather request failed: HTTP {response.status_code}")
        raise SystemExit(1)

    try:
        data = response.json()
        observed_at = datetime.fromtimestamp(data["dt"], tz=UTC)

        record = {
            "observed_at_utc": observed_at.isoformat(),
            "latitude": data["coord"]["lat"],
            "longitude": data["coord"]["lon"],
            "temperature_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "wind_speed_ms": data.get("wind", {}).get("speed"),
            "wind_direction_deg": data.get("wind", {}).get("deg"),
            "conditions": data["weather"][0]["description"],
            "source": "openweather",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "rain_api_1h": (data.get("rain") or {}).get("1h"),
            "snow_api_1h": (data.get("snow") or {}).get("1h"),
            "rain_field_supplied": "1h" in (data.get("rain") or {}),
            "snow_field_supplied": "1h" in (data.get("snow") or {}),
        }

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
        OverflowError,
        OSError,
    ):
        print("The API returned an unexpected response.")
        raise SystemExit(1) from None

    wind_speed = record["wind_speed_ms"]
    wind_direction = record["wind_direction_deg"]

    print("\nKarachi — Current Weather")
    print(f"Observation time (UTC): {record['observed_at_utc']}")
    print(f"Temperature: {record['temperature_c']} °C")
    print(f"Humidity: {record['humidity_pct']} %")

    if wind_speed is None:
        print("Wind speed: N/A")
    else:
        print(f"Wind speed: {wind_speed} m/s")

    if wind_direction is None:
        print("Wind direction: N/A")
    else:
        print(f"Wind direction: {wind_direction} degrees")

    print(f"Conditions: {record['conditions']}")

    try:
        save_observation(record)
    except OSError:
        print(
            "Could not read or write the CSV file. "
            "Close it if open in Excel and check folder permissions."
        )
        raise SystemExit(1) from None
    except (csv.Error, ValueError, KeyError, TypeError):
        print("Invalid weather CSV data. Check the file before retrying.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()