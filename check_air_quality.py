import csv
from pathlib import Path
from datetime import datetime, timezone

import requests


BASE_DIR = Path(__file__).resolve().parent


def save_estimate(record):
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "air_quality_estimates.csv"
    has_data = csv_path.exists() and csv_path.stat().st_size > 0

    if has_data:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != list(record):
                print("CSV columns do not match. Existing file left unchanged.")
                return

            # Keep the first retrieved estimate for each valid time.
            for row in reader:
                if (
                    row["valid_time_utc"] == record["valid_time_utc"]
                    and row["source"] == record["source"]
                    and row["grid_latitude"] == str(record["grid_latitude"])
                    and row["grid_longitude"] == str(record["grid_longitude"])
                ):
                    print("\nAn estimate for this time is already saved.")
                    return

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record))

        if not has_data:
            writer.writeheader()

        writer.writerow(record)

    print(f"\nEstimate saved to: {csv_path}")


def main():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "current": "us_aqi,pm2_5,pm10,uv_index",
        "domains": "cams_global",
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException:
        print("Connection failed. Check your internet and retry.")
        return

    if response.status_code != 200:
        print(f"Request failed: HTTP {response.status_code}")
        return

    try:
        data = response.json()
        current = data["current"]

        valid_time = datetime.fromisoformat(
            current["time"].replace("Z", "+00:00")
        )

        # We explicitly requested UTC from the API.
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)

        record = {
            "valid_time_utc": valid_time.astimezone(timezone.utc).isoformat(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_latitude": params["latitude"],
            "requested_longitude": params["longitude"],
            "grid_latitude": data["latitude"],
            "grid_longitude": data["longitude"],
            "us_aqi": current.get("us_aqi"),
            "pm2_5_ug_m3": current.get("pm2_5"),
            "pm10_ug_m3": current.get("pm10"),
            "uv_index": current.get("uv_index"),
            "source": "open_meteo_cams_global",
            "data_type": "modelled_estimate",
        }

    except (ValueError, KeyError, TypeError, AttributeError):
        print("The API returned an unexpected response.")
        return

    print("\nKarachi — Modelled Air Quality")
    print("Source: Open-Meteo / CAMS Global")
    print("Data type: Model estimate, not a station measurement")
    print(f"Valid time (UTC): {record['valid_time_utc']}")
    print(f"Retrieved at (UTC): {record['retrieved_at_utc']}")

    fields = [
        ("us_aqi", "US AQI", ""),
        ("pm2_5_ug_m3", "PM2.5 concentration", "µg/m³"),
        ("pm10_ug_m3", "PM10 concentration", "µg/m³"),
        ("uv_index", "UV Index", ""),
    ]

    for key, label, unit in fields:
        value = record[key]

        if value is None:
            print(f"{label}: unavailable")
        else:
            print(f"{label}: {value} {unit}".rstrip())

    if all(record[key] is None for key, _, _ in fields):
        print("\nNo air-quality or UV values available. Nothing saved.")
        return

    try:
        save_estimate(record)
    except OSError:
        print(
            "Could not read or write the CSV. "
            "Close it if open in Excel and check folder permissions."
        )


if __name__ == "__main__":
    main()