import os
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def save_observation(record):
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "weather_observations.csv"
    has_data = csv_path.exists() and csv_path.stat().st_size > 0

    # Check whether this observation was already saved.
    if has_data:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            # Prevent appending incompatible columns to an existing file.
            if reader.fieldnames != list(record):
                print("CSV columns do not match. Existing file left unchanged.")
                return

            for row in reader:
                if row["observed_at_utc"] == record["observed_at_utc"]:
                    print("\nThis observation is already saved.")
                    return

    # Append the new observation.
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record))

        if not has_data:
            writer.writeheader()

        writer.writerow(record)

    print(f"\nObservation saved to: {csv_path}")


def main():
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    if not api_key:
        print("Missing API key. Check your .env file.")
        return

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
        return

    if response.status_code == 401:
        print("Key not accepted. Check the key or allow time for activation.")
        return

    if response.status_code != 200:
        print(f"Weather request failed: HTTP {response.status_code}")
        return

    try:
        data = response.json()

        observed_at = datetime.fromtimestamp(
            data["dt"], tz=timezone.utc
        )

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
        }

    except (ValueError, KeyError, IndexError, TypeError, OverflowError):
        print("The API returned an unexpected response.")
        return

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


if __name__ == "__main__":
    main()