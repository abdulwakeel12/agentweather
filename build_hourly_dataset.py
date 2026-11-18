import csv
from datetime import UTC, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def parse_time(value):
    timestamp = datetime.fromisoformat(value)

    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must include a timezone.")

    return timestamp.astimezone(UTC)


def load_hourly_records(path, time_column):
    """Select the latest timestamp within each UTC hour."""
    selected = {}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            timestamp = parse_time(row[time_column])
            hour = timestamp.replace(minute=0, second=0, microsecond=0)

            if hour not in selected or timestamp > selected[hour][0]:
                selected[hour] = (timestamp, row)

    return {hour: item[1] for hour, item in selected.items()}


def main():
    weather_path = DATA_DIR / "weather_observations.csv"
    air_path = DATA_DIR / "air_quality_estimates.csv"
    output_path = DATA_DIR / "hourly_dataset.csv"

    if not weather_path.exists() or not air_path.exists():
        print("Run main.py and check_air_quality.py first.")
        return

    try:
        weather = load_hourly_records(weather_path, "observed_at_utc")
        air = load_hourly_records(air_path, "valid_time_utc")

        # Keep hours from either source; missing values remain blank.
        hours = sorted(set(weather) | set(air))
        records = []

        for hour in hours:
            w = weather.get(hour, {})
            a = air.get(hour, {})

            records.append(
                {
                    "hour_utc": hour.isoformat(),
                    "weather_observed_at_utc": w.get("observed_at_utc", ""),
                    "air_valid_time_utc": a.get("valid_time_utc", ""),
                    "air_retrieved_at_utc": a.get("retrieved_at_utc", ""),
                    "temperature_c": w.get("temperature_c", ""),
                    "humidity_pct": w.get("humidity_pct", ""),
                    "wind_speed_ms": w.get("wind_speed_ms", ""),
                    "wind_direction_deg": w.get("wind_direction_deg", ""),
                    "conditions": w.get("conditions", ""),
                    "us_aqi": a.get("us_aqi", ""),
                    "pm2_5_ug_m3": a.get("pm2_5_ug_m3", ""),
                    "pm10_ug_m3": a.get("pm10_ug_m3", ""),
                    "uv_index": a.get("uv_index", ""),
                    "weather_source": w.get("source", ""),
                    "air_source": a.get("source", ""),
                    "air_data_type": a.get("data_type", ""),
                    "weather_available": bool(w),
                    "air_available": bool(a),
                }
            )

        if not records:
            print("No records available to combine.")
            return

        # Rebuild the derived dataset; preserve both source CSV files.
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        matched = sum(
            row["weather_available"] and row["air_available"]
            for row in records
        )

        print(f"\nHourly dataset saved to: {output_path}")
        print(f"Total hourly rows: {len(records)}")
        print(f"Hours with both sources: {matched}")
        print(f"Hours missing a source: {len(records) - matched}")

    except (ValueError, KeyError, TypeError):
        print("Invalid CSV format or timestamp. Check the source files.")
    except OSError:
        print("File access failed. Close the CSV files in Excel and retry.")


if __name__ == "__main__":
    main()